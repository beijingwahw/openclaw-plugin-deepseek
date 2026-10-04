/**
 * DeepSeek native agent harness.
 *
 * Claims `openclaw-deepseek/*` model refs and executes prepared OpenClaw turns
 * through a native DeepSeek function-calling loop with streamed reasoning and
 * text output, host tool execution, vision input (flash), and usage accounting.
 */
import type { AgentHarness } from "openclaw/plugin-sdk/agent-harness";
import { DeepSeekClient } from "./deepseek-api.js";
import { CASCADE_MODEL_ID, runCascade } from "./cascade.js";
import { runAgentLoop, type LoopEvent, type LoopTool } from "./loop.js";
import { rebuildHistoryFromSessionFile } from "./memory.js";

/** Mirrors the SDK's classifyAgentHarnessTerminalOutcome semantics for empty visible results. */
function classifyEmptyResult(answer: string, reasoning: string): "empty" | "reasoning-only" | undefined {
  const hasVisibleText = answer.trim().length > 0 && answer !== "(no answer)";
  if (hasVisibleText) return undefined;
  if (reasoning.trim()) return "reasoning-only";
  return "empty";
}
import {
  CURRENT_MODELS,
  HARNESS_ID,
  PROVIDER_ID,
  resolveConfig,
  resolveModelId,
  type DeepSeekHarnessConfig,
  type DeepSeekToolSchema,
} from "./types.js";

/** Minimal structural types for the prepared attempt params we consume. */
interface PreparedImageContent {
  type: "image";
  data: string;
  mimeType: string;
}

interface PreparedToolLike {
  name: string;
  label?: string;
  description?: string;
  parameters?: Record<string, unknown>;
  executionMode?: "sequential" | "parallel";
  execute: (
    toolCallId: string,
    params: unknown,
    signal?: AbortSignal,
  ) => Promise<{ content: Array<{ type: string; text?: string }>; details?: unknown }>;
}

interface PreparedAttemptLike {
  sessionId: string;
  sessionFile?: string;
  prompt: string;
  transcriptPrompt?: string;
  systemPrompt?: string;
  tools?: PreparedToolLike[];
  images?: PreparedImageContent[];
  timeoutMs: number;
  abortSignal?: AbortSignal;
  agentId?: string;
  workspaceDir: string;
  model?: { id: string; baseUrl?: string; provider?: string };
  provider?: string;
  modelId?: string;
  onPartialReply?: (payload: { text?: string; delta?: string }) => boolean | void | Promise<boolean | void>;
  onAgentEvent?: (evt: { stream: string; data: Record<string, unknown> }) => void | Promise<void>;
  thinkLevel?: string;
}

export interface HarnessDeps {
  resolveApiKey: () => Promise<string | undefined>;
  getPluginConfig: () => DeepSeekHarnessConfig | undefined;
  /** Shared AIMD concurrency governor. */
  getGovernor?: () => import("./concurrency.js").ConcurrencyGovernor;
  logger?: {
    debug: (msg: string, data?: unknown) => void;
    warn: (msg: string, data?: unknown) => void;
    error: (msg: string, data?: unknown) => void;
  };
}

export function createDeepSeekHarness(deps: HarnessDeps): AgentHarness {
  return {
    id: HARNESS_ID,
    label: "DeepSeek native agent harness",
    autoSelection: { providerIds: [PROVIDER_ID] },

    supports(ctx) {
      if (ctx.provider === PROVIDER_ID) {
        return { supported: true, priority: 100, reason: "openclaw-deepseek companion provider" };
      }
      return { supported: false, reason: "only claims openclaw-deepseek provider refs" };
    },

    async runAttempt(rawParams) {
      const params = rawParams as unknown as PreparedAttemptLike;
      const log = deps.logger;
      try {
        const apiKey = await deps.resolveApiKey();
        if (!apiKey) {
          return failureResult(params, "DEEPSEEK_API_KEY is not configured for the openclaw-deepseek plugin");
        }
        const config = resolveConfig(deps.getPluginConfig());
        const requestedModel = params.model?.id ?? params.modelId ?? config.model;
        const { modelId, retiredAlias } = resolveModelId(requestedModel);
        if (retiredAlias) {
          log?.warn("retired DeepSeek model alias routed to successor", {
            requested: retiredAlias,
            routedTo: modelId,
          });
        }
        const modelInfo = CURRENT_MODELS.find((m) => m.id === modelId);
        const baseUrl = params.model?.baseUrl ?? config.baseUrl;

        // Vision: only flash accepts images; fail readable on pro.
        const images = params.images ?? [];
        if (images.length > 0 && modelInfo && !modelInfo.vision) {
          return failureResult(
            params,
            `Model "${modelId}" does not accept image input. Use "deepseek-flash" for vision tasks.`,
          );
        }
        const imageDataUrls = images.map((image) => `data:${image.mimeType};base64,${image.data}`);

        const client = new DeepSeekClient({
          baseUrl,
          apiKey,
          timeoutMs: config.requestTimeoutMs,
          governor: deps.getGovernor?.(),
        });

        const systemPrompt = params.systemPrompt?.trim() ||
          "You are a helpful assistant running inside the OpenClaw agent harness. Answer in the user's language.";
        const prompt = params.transcriptPrompt ?? params.prompt;

        // Map OpenClaw think level onto DeepSeek reasoning effort. Reasoning
        // models think by default; "off" disables thinking entirely.
        const thinkLevel = params.thinkLevel;
        const reasoningEffort = thinkLevel
          ? normalizeThinkLevel(thinkLevel)
          : modelInfo?.reasoning
            ? config.defaultReasoningEffort
            : "none";

        // Bridge host tools into the DeepSeek loop.
        const loopTools: LoopTool[] = [];
        for (const tool of params.tools ?? []) {
          loopTools.push({
            schema: toolSchemaOf(tool),
            executionMode: tool.executionMode,
            execute: async (args, signal) => {
              const result = await tool.execute(`openclaw-deepseek-${tool.name}-${Date.now()}`, args, signal);
              return flattenContent(result?.content);
            },
          });
        }

        // Multi-turn memory: rebuild DeepSeek-visible history from the
        // OpenClaw transcript so the harness keeps context across turns
        // (stateless harness + stateful conversation).
        const history = await rebuildHistoryFromSessionFile(params.sessionFile, { maxTurns: 12 });

        // Stable per-session user_id gives DeepSeek scheduler/KV-cache
        // isolation and better cache-hit affinity for multi-turn sessions
        // (cache hits are billed at ~1/50 of cache misses).
        const sessionUserId = params.sessionId
          ? `oc-${params.sessionId}`.replace(/[^a-zA-Z0-9\-_]/g, "-").slice(0, 512)
          : undefined;
        const result =
          modelId === CASCADE_MODEL_ID
            ? await runCascadeAttempt()
            : await runAgentLoop({
                requester: client,
                model: modelId,
                systemPrompt,
                prompt,
                tools: loopTools,
                maxSteps: config.researchMaxSteps,
                maxTokensPerStep: config.researchMaxTokensPerStep,
                reasoningEffort,
                history: history.messages,
                imageDataUrls: imageDataUrls.length > 0 ? imageDataUrls : undefined,
                userId: sessionUserId,
                signal: params.abortSignal,
                onEvent: (event) => void emitLoopEvent(params, event, log),
              });

        async function runCascadeAttempt() {
          const cascade = await runCascade({
            requester: client,
            systemPrompt,
            prompt,
            maxTokens: Math.min(config.researchMaxTokensPerStep, 8192),
            signal: params.abortSignal,
          });
          return {
            answer: cascade.text,
            allText: cascade.text,
            reasoning: "",
            steps: [],
            usage: cascade.usage,
            iterations: cascade.iterations,
            stopReason: "finished" as const,
            error: undefined,
          };
        }

        if (result.stopReason === "aborted") {
          return {
            terminal: { kind: "aborted", source: "runtime" as const },
            sessionIdUsed: params.sessionId,
            messagesSnapshot: [],
            assistantTexts: result.answer ? [result.answer] : [],
            lastAssistant: undefined,
            toolMetas: [],
            replayMetadata: { hadPotentialSideEffects: false, replaySafe: true },
            itemLifecycle: { startedCount: 0, completedCount: 0, activeCount: 0 },
            didSendViaMessagingTool: false,
            messagingToolSentTexts: [],
            messagingToolSentMediaUrls: [],
            messagingToolSentTargets: [] as never[],
            cloudCodeAssistFormatError: false,
          };
        }
        if (result.stopReason === "error") {
          return failureResult(params, result.error ?? "unknown DeepSeek loop error");
        }

        const assistantText = result.answer || result.allText || "(no answer)";
        // Classify degenerate outcomes so OpenClaw's fallback policy can
        // retry on a different model when the harness produced nothing visible.
        const classification = classifyEmptyResult(assistantText, result.reasoning);
        if (classification === "empty" || classification === "reasoning-only") {
          const base = okResultBase(params, modelId, assistantText);
          const classified: typeof base & { agentHarnessResultClassification: "empty" | "reasoning-only" } = {
            ...base,
            agentHarnessResultClassification: classification,
          };
          return classified;
        }
        return {
          terminal: { kind: "ok" as const },
          sessionIdUsed: params.sessionId,
          messagesSnapshot: [],
          assistantTexts: [assistantText],
          lastAssistant: {
            role: "assistant",
            content: [{ type: "text", text: assistantText }],
            api: "openai-completions",
            provider: PROVIDER_ID,
            model: modelId,
            usage: usageOf(result.usage, modelId),
            stopReason: "stop",
            timestamp: Date.now(),
          },
          attemptUsage: usageOf(result.usage, modelId),
          modelIterations: result.iterations,
          toolMetas: [],
          replayMetadata: { hadPotentialSideEffects: false, replaySafe: true },
          itemLifecycle: { startedCount: 0, completedCount: 0, activeCount: 0 },
          didSendViaMessagingTool: false,
          messagingToolSentTexts: [],
          messagingToolSentMediaUrls: [],
          messagingToolSentTargets: [] as never[],
          cloudCodeAssistFormatError: false,
        };
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        log?.error("openclaw-deepseek runAttempt failed", { error: message });
        return failureResult(params, message);
      }
    },
  };
}

/** Maps OpenClaw /think levels onto DeepSeek reasoning_effort values. */
function normalizeThinkLevel(level: string): "none" | "low" | "high" | "max" {
  const v = level.trim().toLowerCase().replace(/[\s_-]/g, "");
  if (["off", "none", "disabled"].includes(v)) return "none";
  if (["minimal", "low", "think"].includes(v)) return "low";
  if (["medium", "high", "thinkhard", "thinkharder", "ultrathink", "xhigh", "extrahigh", "highest"].includes(v)) return "high";
  if (["max", "ultrathinkplus", "maximum", "adaptive"].includes(v)) return "max";
  return "high";
}

function toolSchemaOf(tool: PreparedToolLike): DeepSeekToolSchema {
  return {
    type: "function",
    function: {
      name: tool.name,
      description: tool.description ?? tool.label ?? tool.name,
      parameters: normalizeJsonSchema(tool.parameters),
    },
  };
}

/** Ensures the schema is a JSON-Schema object DeepSeek accepts. */
function normalizeJsonSchema(schema: Record<string, unknown> | undefined): Record<string, unknown> {
  if (!schema || typeof schema !== "object") return { type: "object", properties: {} };
  return schema;
}

function flattenContent(content: Array<{ type: string; text?: string }> | undefined): string {
  if (!content || content.length === 0) return "";
  return content
    .map((part) => (part.type === "text" && typeof part.text === "string" ? part.text : ""))
    .filter(Boolean)
    .join("\n");
}

function usageOf(usage: import("./types.js").DeepSeekUsage, modelId: string) {
  const input = usage.prompt_tokens ?? 0;
  const output = usage.completion_tokens ?? 0;
  const cacheRead = usage.prompt_cache_hit_tokens ?? 0;
  const cacheWrite = usage.prompt_cache_miss_tokens ?? 0;
  return {
    input,
    output,
    cacheRead,
    cacheWrite,
    totalTokens: usage.total_tokens ?? input + output,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    contextUsage: { state: "unavailable" as const },
    _model: modelId,
  };
}

async function emitLoopEvent(
  params: PreparedAttemptLike,
  event: LoopEvent,
  log?: HarnessDeps["logger"],
): Promise<void> {
  try {
    if (event.type === "text" && params.onPartialReply) {
      // True delta: the channel renders incrementally instead of re-sending
      // the whole accumulated text on every chunk.
      await params.onPartialReply({ delta: event.text });
    }
    if (params.onAgentEvent) {
      const data: Record<string, unknown> =
        event.type === "toolCall"
          ? { tool: event.name, args: event.args }
          : event.type === "toolResult"
            ? { tool: event.name, ok: event.ok, preview: event.preview }
            : event.type === "step"
              ? { step: event.index }
              : { text: event.text };
      await params.onAgentEvent({
        stream: event.type === "reasoning" ? "reasoning" : "assistant",
        data,
      });
    }
  } catch (error) {
    log?.warn("openclaw-deepseek event callback failed", {
      error: error instanceof Error ? error.message : String(error),
    });
  }
}

function okResultBase(params: PreparedAttemptLike, modelId: string, assistantText: string) {
  return {
    terminal: { kind: "ok" as const },
    sessionIdUsed: params.sessionId,
    messagesSnapshot: [] as never[],
    assistantTexts: [assistantText],
    lastAssistant: {
      role: "assistant" as const,
      content: [{ type: "text" as const, text: assistantText }],
      api: "openai-completions" as const,
      provider: PROVIDER_ID,
      model: modelId,
      usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 }, contextUsage: { state: "unavailable" as const } },
      stopReason: "stop" as const,
      timestamp: Date.now(),
    },
    toolMetas: [] as never[],
    replayMetadata: { hadPotentialSideEffects: false, replaySafe: true },
    itemLifecycle: { startedCount: 0, completedCount: 0, activeCount: 0 },
    didSendViaMessagingTool: false,
    messagingToolSentTexts: [] as string[],
    messagingToolSentMediaUrls: [] as string[],
    messagingToolSentTargets: [] as never[],
    cloudCodeAssistFormatError: false,
  };
}

function failureResult(params: PreparedAttemptLike, message: string) {
  return {
    terminal: {
      kind: "failed" as const,
      source: "prompt" as const,
      error: new Error(message),
    },
    sessionIdUsed: params.sessionId,
    messagesSnapshot: [] as never[],
    assistantTexts: [] as string[],
    lastAssistant: undefined,
    toolMetas: [] as never[],
    replayMetadata: { hadPotentialSideEffects: false, replaySafe: true },
    itemLifecycle: { startedCount: 0, completedCount: 0, activeCount: 0 },
    didSendViaMessagingTool: false,
    messagingToolSentTexts: [] as string[],
    messagingToolSentMediaUrls: [] as string[],
    messagingToolSentTargets: [] as never[],
    cloudCodeAssistFormatError: false,
  };
}
