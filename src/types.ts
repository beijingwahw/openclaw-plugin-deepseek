/**
 * Shared types, model catalog, and policy tables for the OpenClaw DeepSeek plugin.
 * Model facts follow the official 2026-09-10 catalog (api-docs.deepseek.com).
 */

/** DeepSeek Chat Completions tool definition (OpenAI-compatible function schema). */
export interface DeepSeekToolSchema {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

/** One assistant tool_call as returned by DeepSeek Chat Completions. */
export interface DeepSeekToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}

export interface DeepSeekUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  prompt_cache_hit_tokens?: number;
  prompt_cache_miss_tokens?: number;
  completion_tokens_details?: {
    reasoning_tokens?: number;
  };
  prompt_tokens_details?: {
    cached_tokens?: number;
  };
}

export interface DeepSeekResponseMessage {
  role: "assistant";
  content: string | null;
  reasoning_content?: string | null;
  tool_calls?: DeepSeekToolCall[];
}

export interface DeepSeekResponseChoice {
  index?: number;
  message?: DeepSeekResponseMessage;
  finish_reason?: string | null;
}

export interface DeepSeekChatResponse {
  id?: string;
  model?: string;
  choices?: DeepSeekResponseChoice[];
  usage?: DeepSeekUsage;
}

/** Normalized event emitted by the streaming loop. */
export type DeepSeekStreamEvent =
  | { type: "reasoning"; text: string }
  | { type: "text"; text: string }
  | { type: "toolCall"; call: DeepSeekToolCall }
  | { type: "usage"; usage: DeepSeekUsage }
  | { type: "finish"; finishReason: string | null };

export interface DeepSeekRequester {
  /** POST {baseUrl}/chat/completions with SSE streaming. */
  streamChat(body: Record<string, unknown>, signal?: AbortSignal): AsyncIterable<DeepSeekStreamEvent>;
  /** POST {baseUrl}/chat/completions, non-streaming. */
  completeChat(body: Record<string, unknown>, signal?: AbortSignal): Promise<DeepSeekChatResponse>;
  /** GET {baseUrl}/user/balance. */
  getBalance(signal?: AbortSignal): Promise<{ balance: string; currency: string | null; raw: unknown }>;
}

/** Thinking effort levels accepted by DeepSeek. */
export type DeepSeekReasoningEffort = "none" | "low" | "high" | "max";

export interface DeepSeekHarnessConfig {
  baseUrl?: string;
  /** Primary model for the harness loop and tools. Default deepseek-flash. */
  model?: string;
  researchMaxSteps?: number;
  researchMaxTokensPerStep?: number;
  requestTimeoutMs?: number;
  /** Default thinking effort. Default "high". */
  defaultReasoningEffort?: DeepSeekReasoningEffort;
  /** Model for FIM code completion. Default deepseek-flash. */
  fimModel?: string;
  /** Daily USD budget for the live cost dashboard; warnings fire when exceeded. */
  dailyBudgetUsd?: number;
  /** Cascade judge escalation threshold: escalate when grade=escalate OR confidence < threshold. Default 0 (grade-driven only). */
  judgeThreshold?: number;
}

/** Fully resolved configuration (all fields present). */
export interface ResolvedDeepSeekConfig {
  baseUrl: string;
  model: string;
  researchMaxSteps: number;
  researchMaxTokensPerStep: number;
  requestTimeoutMs: number;
  defaultReasoningEffort: DeepSeekReasoningEffort;
  fimModel: string;
  dailyBudgetUsd: number | null;
  judgeThreshold: number;
}

export const PLUGIN_ID = "openclaw-deepseek";
export const PROVIDER_ID = "openclaw-deepseek";
export const HARNESS_ID = "openclaw-deepseek";

/** Official base URL uses the bare domain; /v1 remains compatible. */
import { clampNumber } from "./shared.js";

/**
 * Branded model reference: a string that passed through resolveModelId.
 * Callers can't accidentally pass an unvalidated model id where a resolved
 * one is required — the type only arises from the resolver.
 */
declare const resolvedBrand: unique symbol;
export type ResolvedModelRef = string & { readonly [resolvedBrand]: true };

export const DEFAULT_BASE_URL = "https://api.deepseek.com";
export const DEFAULT_MODEL = "deepseek-flash";
export const DEFAULT_REASONING_EFFORT: DeepSeekReasoningEffort = "high";
export const DEFAULT_RESEARCH_MAX_STEPS = 8;
export const DEFAULT_RESEARCH_MAX_TOKENS_PER_STEP = 16384;
export const DEFAULT_REQUEST_TIMEOUT_MS = 180_000;

/**
 * Current official catalog (2026-09-10). Prices are USD per million tokens,
 * off-peak / cache-miss input / output. Peak is 2x off-peak.
 */
export interface DeepSeekModelInfo {
  id: string;
  name: string;
  reasoning: boolean;
  vision: boolean;
  contextWindow: number;
  maxTokens: number;
  /** off-peak USD per M tokens: input(cache miss), cacheRead, output */
  cost: { input: number; output: number; cacheRead: number; cacheWrite: number };
  notes: string;
}

export const CURRENT_MODELS: DeepSeekModelInfo[] = [
  {
    id: "deepseek-flash",
    name: "DeepSeek Flash (V4.1-Flash)",
    reasoning: true,
    vision: true,
    contextWindow: 1_000_000,
    maxTokens: 384_000,
    cost: { input: 0.15, output: 0.6, cacheRead: 0.003, cacheWrite: 0.15 },
    notes: "V4.1-Flash: multimodal (image input), thinking on by default",
  },
  {
    id: "deepseek-v4-pro",
    name: "DeepSeek V4 Pro",
    reasoning: true,
    vision: false,
    contextWindow: 1_000_000,
    maxTokens: 384_000,
    cost: { input: 0.66, output: 1.98, cacheRead: 0.022, cacheWrite: 0.66 },
    notes: "V4-Pro-0813 flagship, text-only, thinking on by default",
  },
];

/**
 * Retired aliases -> current model ids. DeepSeek retired deepseek-chat and
 * deepseek-reasoner on 2026-07-24 and deepseek-v4-flash on 2026-09-10; the
 * API still accepts the old names but bills/serves the successor model, so we
 * map them eagerly and warn once.
 */
export const RETIRED_ALIASES: Record<string, string> = {
  "deepseek-chat": "deepseek-flash",
  "deepseek-reasoner": "deepseek-flash",
  "deepseek-v4-flash": "deepseek-flash",
  "deepseek-v4-flash-vision-exp": "deepseek-flash",
  "deepseek-v3.2": "deepseek-flash",
};

export function resolveModelId(requested: string): { modelId: ResolvedModelRef; retiredAlias: string | null } {
  const mapped = RETIRED_ALIASES[requested];
  if (mapped && mapped !== requested) {
    return { modelId: mapped as ResolvedModelRef, retiredAlias: requested };
  }
  return { modelId: requested as ResolvedModelRef, retiredAlias: null };
}

/** Maps OpenClaw-style thinking words onto DeepSeek reasoning_effort values. */
export function normalizeReasoningEffort(input: string | undefined): DeepSeekReasoningEffort {
  if (!input) return DEFAULT_REASONING_EFFORT;
  const v = input.trim().toLowerCase().replace(/[\s_-]/g, "");
  if (["off", "none", "disabled", "disable"].includes(v)) return "none";
  if (["minimal", "low", "think"].includes(v)) return "low";
  if (["medium", "high", "thinkhard", "thinkharder", "ultrathink", "xhigh", "extrahigh", "highest"].includes(v)) return "high";
  if (["max", "ultrathinkplus", "maximum"].includes(v)) return "max";
  return DEFAULT_REASONING_EFFORT;
}

export function resolveConfig(config: DeepSeekHarnessConfig | undefined): ResolvedDeepSeekConfig {
  return {
    baseUrl: (config?.baseUrl ?? "").trim() || DEFAULT_BASE_URL,
    model: (config?.model ?? "").trim() || DEFAULT_MODEL,
    researchMaxSteps: clampNumber(config?.researchMaxSteps, 1, 32, DEFAULT_RESEARCH_MAX_STEPS),
    researchMaxTokensPerStep: clampNumber(config?.researchMaxTokensPerStep, 256, 384_000, DEFAULT_RESEARCH_MAX_TOKENS_PER_STEP),
    requestTimeoutMs: clampNumber(config?.requestTimeoutMs, 1_000, 600_000, DEFAULT_REQUEST_TIMEOUT_MS),
    defaultReasoningEffort: config?.defaultReasoningEffort ?? DEFAULT_REASONING_EFFORT,
    fimModel: (config?.fimModel ?? "").trim() || "deepseek-flash",
    dailyBudgetUsd:
      typeof config?.dailyBudgetUsd === "number" && Number.isFinite(config.dailyBudgetUsd) && config.dailyBudgetUsd > 0
        ? config.dailyBudgetUsd
        : null,
    judgeThreshold:
      typeof config?.judgeThreshold === "number" && Number.isFinite(config.judgeThreshold)
        ? Math.min(1, Math.max(0, config.judgeThreshold))
        : 0,
  };
}


/** DeepSeek OpenAI-compatible chat message shape used inside the loop. */
export interface DeepSeekChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | null;
  reasoning_content?: string | null;
  tool_calls?: DeepSeekToolCall[];
  tool_call_id?: string;
  name?: string;
}
