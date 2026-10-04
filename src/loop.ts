/**
 * DeepSeek function-calling agent loop.
 *
 * Drives multi-turn tool use against the DeepSeek Chat Completions API:
 * send messages -> stream assistant output -> execute tool calls -> feed
 * tool results back -> repeat until a final text answer or a step budget.
 */
import { mergeDeepSeekUsage, previewOfText, stableStringify as sharedStableStringify } from "./shared.js";
import type {
  DeepSeekChatMessage,
  DeepSeekReasoningEffort,
  DeepSeekRequester,
  DeepSeekStreamEvent,
  DeepSeekToolSchema,
  DeepSeekUsage,
} from "./types.js";

export interface LoopTool {
  schema: DeepSeekToolSchema;
  execute: (args: Record<string, unknown>, signal: AbortSignal | undefined) => Promise<string>;
  /** "sequential" tools run one-at-a-time; others may run in parallel. */
  executionMode?: "sequential" | "parallel";
}

export interface LoopStepTrace {
  index: number;
  assistantText: string | null;
  reasoning: string | null;
  toolCalls: Array<{ name: string; argsPreview: string; resultPreview: string; ok: boolean }>;
}

export interface AgentLoopResult {
  /** Final visible answer text (last assistant content without tool calls). */
  answer: string;
  /** All assistant text emitted during the loop (including intermediate narration). */
  allText: string;
  /** Aggregated reasoning_content across steps, when present. */
  reasoning: string;
  steps: LoopStepTrace[];
  usage: DeepSeekUsage;
  iterations: number;
  stopReason: "finished" | "stepBudget" | "aborted" | "error" | "contextOverflow";
  error?: string;
  /** Times the loop compacted old tool results to fit the context window. */
  compactions?: number;
}

export interface AgentLoopOptions {
  requester: DeepSeekRequester;
  model: string;
  systemPrompt: string;
  /** Initial user content. */
  prompt: string;
  tools: LoopTool[];
  maxSteps: number;
  maxTokensPerStep: number;
  /** Extra messages inserted after the system prompt (e.g. prior conversation). */
  history?: DeepSeekChatMessage[];
  /** Thinking effort. "none" disables thinking (non-thinking path). */
  reasoningEffort?: DeepSeekReasoningEffort;
  onEvent?: (event: LoopEvent) => void;
  signal?: AbortSignal;
  /** Temperature for requests (ignored when thinking is enabled). */
  temperature?: number;
  /** Optional image attachments for the first user turn (data URLs). */
  imageDataUrls?: string[];
  /** DeepSeek user_id: per-caller cache/scheduling isolation (regex [a-zA-Z0-9-_]{1,512}). */
  userId?: string;
}

export type LoopEvent =
  | { type: "reasoning"; text: string; delta: true }
  | { type: "text"; text: string; delta: true }
  | { type: "assistantText"; text: string }
  | { type: "toolCall"; name: string; args: Record<string, unknown> }
  | { type: "toolResult"; name: string; ok: boolean; preview: string }
  | { type: "step"; index: number };

export async function runAgentLoop(options: AgentLoopOptions): Promise<AgentLoopResult> {
  const { requester, model, systemPrompt, prompt, tools, maxSteps, maxTokensPerStep } = options;
  const reasoning = options.reasoningEffort !== "none";
  const firstUserContent =
    options.imageDataUrls && options.imageDataUrls.length > 0
      ? ([{ type: "text", text: prompt }, ...options.imageDataUrls.map((url) => ({ type: "image_url", image_url: { url } }))] as unknown as string)
      : prompt;
  const messages: DeepSeekChatMessage[] = [
    { role: "system", content: systemPrompt },
    ...(options.history ?? []),
    { role: "user", content: firstUserContent },
  ];
  const toolMap = new Map<string, LoopTool>();
  for (const tool of tools) toolMap.set(tool.schema.function.name, tool);
  const toolSchemas = tools.map((t) => t.schema);

  // Circuit breaker: (name, args-hash) pairs repeated across consecutive
  // steps signal a stuck loop; break out with what we have.
  const repeatedCalls = new Map<string, number>();
  let compactions = 0;
  const CONTEXT_CHAR_BUDGET = 1_800_000; // ~450K tokens, below the 1M window with headroom
  // Incremental message-size tracking: O(1) per append instead of re-serializing
  // the whole conversation on every step (JSON.stringify of megabytes per turn
  // is quadratic and freezes large-context loops).
  const messageSizes = new Map<number, number>(); // message index -> approx chars
  const approxSize = (msg: DeepSeekChatMessage): number => {
    try {
      return JSON.stringify(msg).length;
    } catch {
      // Circular or exotic message shape: size estimate falls back to a
      // conservative 4KB rather than breaking the budget tracker.
      return 4096;
    }
  };
  let totalChars = messages.reduce((sum, msg, i) => {
    const size = approxSize(msg);
    messageSizes.set(i, size);
    return sum + size;
  }, 0);
  const trackMessage = (msg: DeepSeekChatMessage): void => {
    const size = approxSize(msg);
    messageSizes.set(messages.length, size);
    totalChars += size;
  };
  const replaceMessageSize = (index: number, msg: DeepSeekChatMessage): void => {
    const next = approxSize(msg);
    const prev = messageSizes.get(index) ?? 0;
    messageSizes.set(index, next);
    totalChars += next - prev;
  };

  const result: AgentLoopResult = {
    answer: "",
    allText: "",
    reasoning: "",
    steps: [],
    usage: {},
    iterations: 0,
    stopReason: "finished",
  };

  for (let step = 0; step < maxSteps; step++) {
    options.signal?.throwIfAborted();
    options.onEvent?.({ type: "step", index: step });
    result.iterations = step + 1;

    const body: Record<string, unknown> = {
      model,
      messages,
      max_tokens: maxTokensPerStep,
      stream: true,
      stream_options: { include_usage: true },
    };
    if (options.userId) {
      body.user_id = sanitizeUserId(options.userId);
    }
    // Self-healing context budget: when the conversation exceeds the char
    // budget, compact the oldest tool results (keep the latest two steps)
    // instead of failing the whole run on a provider 400. Size tracking is
    // incremental, so this check is O(messages) at worst, not O(total chars).
    const COMPACTED_MARKER = "(compacted:";
    while (totalChars > CONTEXT_CHAR_BUDGET && messages.length > 6) {
      const toolMsgIndex = messages.findIndex(
        (m, i) =>
          i > 1 &&
          m.role === "tool" &&
          i < messages.length - 4 &&
          !(typeof m.content === "string" && m.content.startsWith(COMPACTED_MARKER)),
      );
      if (toolMsgIndex < 0) break;
      const compacted: DeepSeekChatMessage = {
        ...messages[toolMsgIndex],
        content: "(compacted: earlier tool output trimmed to fit the context window)",
      };
      messages[toolMsgIndex] = compacted;
      replaceMessageSize(toolMsgIndex, compacted);
      compactions += 1;
    }
    if (reasoning) {
      // DeepSeek thinking control (official shape) + effort level.
      body.thinking = { type: "enabled" };
      body.reasoning_effort = options.reasoningEffort ?? "high";
    } else {
      body.thinking = { type: "disabled" };
    }
    if (!reasoning && options.temperature !== undefined) {
      // Sampling params are silently ignored while thinking is enabled.
      body.temperature = options.temperature;
    }
    if (toolSchemas.length > 0) {
      body.tools = toolSchemas;
      // tool_choice required/named returns 400 while thinking is enabled.
      body.tool_choice = "auto";
    }

    let stepText = "";
    let stepReasoning = "";
    const toolCalls: Array<{ id: string; name: string; args: Record<string, unknown> }> = [];
    let finishReason: string | null = null;
    let stepUsage: DeepSeekUsage | undefined;

    try {
      for await (const event of streamWithUsage(requester, body, options.signal)) {
        if (event.type === "usage") {
          stepUsage = event.usage;
          continue;
        }
        applyStreamEvent(event, {
          onReasoning: (t) => {
            stepReasoning += t;
            options.onEvent?.({ type: "reasoning", text: t, delta: true });
          },
          onText: (t) => {
            stepText += t;
            options.onEvent?.({ type: "text", text: t, delta: true });
          },
          onToolCall: (call) => {
            toolCalls.push({ id: call.id, name: call.function.name, args: safeParseArgs(call.function.arguments) });
          },
          onFinish: (reason) => {
            finishReason = reason ?? finishReason;
          },
        });
      }
    } catch (error) {
      if (isAbort(error, options.signal)) {
        result.stopReason = "aborted";
        return result;
      }
      result.stopReason = "error";
      result.error = errorMessage(error);
      return result;
    }

    if (stepUsage) mergeDeepSeekUsage(result.usage, stepUsage);
    if (stepText) {
      result.allText += (result.allText ? "\n\n" : "") + stepText;
      options.onEvent?.({ type: "assistantText", text: stepText });
    }
    if (stepReasoning) result.reasoning += (result.reasoning ? "\n\n" : "") + stepReasoning;

    const trace: LoopStepTrace = {
      index: step,
      assistantText: stepText || null,
      reasoning: stepReasoning || null,
      toolCalls: [],
    };
    result.steps.push(trace);

    if (toolCalls.length === 0) {
      result.answer = stepText || result.allText;
      result.compactions = compactions;
      return result;
    }

    // Record the assistant tool-call turn. DeepSeek requires tool_calls on the
    // replayed assistant message; when tools are present, ALL historical
    // reasoning_content must be replayed too (400 otherwise), so we keep it.
    const assistantTurn: DeepSeekChatMessage = {
      role: "assistant",
      content: stepText || null,
      tool_calls: toolCalls.map((c) => ({
        id: c.id,
        type: "function" as const,
        function: { name: c.name, arguments: JSON.stringify(c.args) },
      })),
    };
    if (stepReasoning) assistantTurn.reasoning_content = stepReasoning;
    messages.push(assistantTurn);
    trackMessage(assistantTurn);

    for (const call of toolCalls) {
      options.signal?.throwIfAborted();
      options.onEvent?.({ type: "toolCall", name: call.name, args: call.args });
      const signature = `${call.name}:${sharedStableStringify(call.args)}`;
      const count = (repeatedCalls.get(signature) ?? 0) + 1;
      repeatedCalls.set(signature, count);
      if (count >= 3) {
        // Same call three steps in a row: the loop is stuck. Stop with what we have.
        result.stopReason = "stepBudget";
        result.answer = result.allText || `(stopped: repeated tool call ${call.name})`;
        return result;
      }
    }
    // Decay non-repeated signatures so only consecutive repeats count.
    for (const [key, count] of repeatedCalls) {
      if (!toolCalls.some((c) => `${c.name}:${sharedStableStringify(c.args)}` === key)) {
        if (count <= 1) repeatedCalls.delete(key);
        else repeatedCalls.set(key, count - 1);
      }
    }

    // Execute the batch: parallel tools run concurrently (bounded), sequential
    // tools run alone. Result order matches call order so replay stays
    // deterministic regardless of completion timing.
    const results = await executeToolBatch(toolCalls, {
      toolMap,
      signal: options.signal,
      parallelCap: 4,
      onResult: (name, ok, preview) => options.onEvent?.({ type: "toolResult", name, ok, preview }),
    });

    for (let i = 0; i < toolCalls.length; i++) {
      const call = toolCalls[i];
      const outcome = results[i];
      trace.toolCalls.push({
        name: call.name,
        argsPreview: previewOf(call.args),
        resultPreview: outcome.preview,
        ok: outcome.ok,
      });
      const toolMessage: DeepSeekChatMessage = {
        role: "tool",
        tool_call_id: call.id,
        name: call.name,
        content: outcome.output,
      };
      messages.push(toolMessage);
      trackMessage(toolMessage);
    }

    if (finishReason === "length") {
      // Continue the loop so the model can wrap up with the tool results.
      continue;
    }
  }

  result.stopReason = "stepBudget";
  result.answer = result.allText;
  result.compactions = compactions;
  return result;
}


/** Runs one single-shot completion (no tools). Used by deepseek_reason. */
export async function runSingleCompletion(options: {
  requester: DeepSeekRequester;
  model: string;
  systemPrompt: string;
  prompt: string;
  maxTokens: number;
  reasoningEffort?: DeepSeekReasoningEffort;
  temperature?: number;
  signal?: AbortSignal;
  onText?: (text: string) => void;
  onReasoning?: (text: string) => void;
}): Promise<{ text: string; reasoning: string; usage: DeepSeekUsage }> {
  const reasoning = options.reasoningEffort !== "none";
  const body: Record<string, unknown> = {
    model: options.model,
    messages: [
      { role: "system", content: options.systemPrompt },
      { role: "user", content: options.prompt },
    ],
    max_tokens: options.maxTokens,
    stream: true,
    stream_options: { include_usage: true },
  };
  if (reasoning) {
    body.thinking = { type: "enabled" };
    body.reasoning_effort = options.reasoningEffort ?? "high";
  } else {
    body.thinking = { type: "disabled" };
    if (options.temperature !== undefined) body.temperature = options.temperature;
  }

  let text = "";
  let reasoningText = "";
  let usage: DeepSeekUsage = {};
  for await (const event of streamWithUsage(options.requester, body, options.signal)) {
    if (event.type === "usage") {
      usage = event.usage;
      continue;
    }
    if (event.type === "reasoning") {
      reasoningText += event.text;
      options.onReasoning?.(event.text);
    } else if (event.type === "text") {
      text += event.text;
      options.onText?.(event.text);
    }
  }
  return { text, reasoning: reasoningText, usage };

}

type ExtendedEvent = DeepSeekStreamEvent;

async function* streamWithUsage(
  requester: DeepSeekRequester,
  body: Record<string, unknown>,
  signal: AbortSignal | undefined,
): AsyncGenerator<ExtendedEvent> {
  // DeepSeek honors stream_options.include_usage: the last data frame before
  // [DONE] carries a usage object with an empty choices array. The SSE parser
  // surfaces it as a { type: "usage" } event which the loop merges.
  yield* requester.streamChat(body, signal);
}

interface StreamEventHandlers {
  onReasoning: (text: string) => void;
  onText: (text: string) => void;
  onToolCall: (call: { id: string; function: { name: string; arguments: string } }) => void;
  onFinish: (reason: string | null) => void;
}

function applyStreamEvent(event: DeepSeekStreamEvent, handlers: StreamEventHandlers): void {
  switch (event.type) {
    case "reasoning":
      handlers.onReasoning(event.text);
      break;
    case "text":
      handlers.onText(event.text);
      break;
    case "toolCall":
      handlers.onToolCall(event.call);
      break;
    case "finish":
      handlers.onFinish(event.finishReason);
      break;
  }
}

export function safeParseArgs(raw: string): Record<string, unknown> {
  if (!raw || !raw.trim()) return {};
  try {
    const parsed = JSON.parse(raw);
    if (parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
    return { _raw: raw };
  } catch {
    // Malformed tool-call arguments: preserve the raw string so the model can
    // see what went wrong instead of silently dropping the call.
    return { _raw: raw };
  }
}

interface ToolBatchOutcome {
  output: string;
  preview: string;
  ok: boolean;
}

interface ToolBatchOptions {
  toolMap: Map<string, LoopTool>;
  signal: AbortSignal | undefined;
  parallelCap: number;
  onResult: (name: string, ok: boolean, preview: string) => void;
}

/**
 * Executes one batch of tool calls. Sequential tools isolate themselves into
 * their own Await Group; parallel tools run concurrently up to parallelCap.
 * Results are returned in call order.
 */
async function executeToolBatch(
  calls: Array<{ id: string; name: string; args: Record<string, unknown> }>,
  options: ToolBatchOptions,
): Promise<ToolBatchOutcome[]> {
  const outcomes: ToolBatchOutcome[] = new Array(calls.length);
  const runOne = async (call: { name: string; args: Record<string, unknown> }): Promise<ToolBatchOutcome> => {
    const tool = options.toolMap.get(call.name);
    if (!tool) {
      const output = `Error: unknown tool "${call.name}". Available: ${[...options.toolMap.keys()].join(", ") || "(none)"}.`;
      return { output, preview: output, ok: false };
    }
    try {
      const output = await tool.execute(call.args, options.signal);
      return { output, preview: previewOfText(output), ok: true };
    } catch (error) {
      if (options.signal?.aborted) throw error;
      const output = `Error executing ${call.name}: ${errorMessage(error)}`;
      return { output, preview: output, ok: false };
    }
  };

  const hasSequential = calls.some((c) => options.toolMap.get(c.name)?.executionMode === "sequential");
  const allSequential = hasSequential && calls.every((c) => options.toolMap.get(c.name)?.executionMode === "sequential");

  // If any tool demands sequential execution, run the whole batch serially.
  // That is the conservative contract: sequential tools must never overlap
  // with anything, including parallel ones.
  if (allSequential || (hasSequential && calls.length > 1)) {
    for (let i = 0; i < calls.length; i++) {
      const outcome = await runOne(calls[i]);
      outcomes[i] = outcome;
      options.onResult(calls[i].name, outcome.ok, outcome.preview);
    }
    return outcomes;
  }

  // Pure parallel batch with a concurrency cap.
  let cursor = 0;
  const worker = async (): Promise<void> => {
    while (cursor < calls.length) {
      const index = cursor;
      cursor += 1;
      const outcome = await runOne(calls[index]);
      outcomes[index] = outcome;
      options.onResult(calls[index].name, outcome.ok, outcome.preview);
    }
  };
  const workers = Array.from({ length: Math.min(options.parallelCap, calls.length) }, () => worker());
  await Promise.all(workers);
  return outcomes;
}



function isAbort(error: unknown, signal?: AbortSignal): boolean {
  if (signal?.aborted) return true;
  return error instanceof Error && error.name === "AbortError";
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

function previewOf(args: Record<string, unknown>): string {
  try {
    const json = JSON.stringify(args);
    return json.length > 160 ? `${json.slice(0, 160)}…` : json;
  } catch {
    // Arguments contain circular structure: preview degrades to a marker.
    return "(unserializable args)";
  }
}

/** Keeps user_id within DeepSeek's accepted alphabet and length. */
export function sanitizeUserId(raw: string): string {
  const cleaned = raw.replace(/[^a-zA-Z0-9\-_]/g, "-").slice(0, 512);
  return cleaned || "openclaw-deepseek";
}
