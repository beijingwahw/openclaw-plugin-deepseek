/**
 * DeepSeek-native cascade — thin facade over the universal engine.
 *
 * Historical note: this module once carried a full parallel implementation of
 * draft -> judge -> escalate. It is now a compatibility adapter that runs the
 * universal cascade with the native DeepSeek executor, keeping the original
 * (requester-based) API for the harness's virtual cascade model.
 */
import { computeCost, type CostBreakdown } from "./cost.js";
import {
  createDeepSeekNativeExecutor,
  nativeExecutorFromRequester,
  runUniversalCascade,
  type UniversalCascadeEvent,
} from "./universal-cascade.js";
import { JUDGE_SYSTEM_PROMPT } from "./shared.js";
import type { DeepSeekRequester, DeepSeekUsage } from "./types.js";

export const CASCADE_MODEL_ID = "deepseek-cascade";

export interface CascadeJudge {
  grade: "accept" | "escalate";
  reason: string;
  confidence: number;
}

export interface CascadeOptions {
  requester: DeepSeekRequester;
  systemPrompt: string;
  prompt: string;
  maxTokens?: number;
  /** Model tiers, default ["deepseek-flash", "deepseek-v4-pro"]. */
  tiers?: string[];
  /** Judge prompt prefix override. */
  judgeSystem?: string;
  /** Escalate when grade=escalate OR confidence < threshold. 0 disables. */
  judgeThreshold?: number;
  onEvent?: (event: CascadeEvent) => void;
  signal?: AbortSignal;
}

export type CascadeEvent =
  | { type: "tier"; model: string; index: number }
  | { type: "draft"; model: string; text: string; usage: DeepSeekUsage }
  | { type: "judge"; grade: "accept" | "escalate"; reason: string; confidence: number }
  | { type: "escalated"; from: string; to: string }
  | { type: "final"; model: string; text: string };

export interface CascadeResult {
  text: string;
  finalModel: string;
  judge: CascadeJudge;
  iterations: number;
  usage: DeepSeekUsage;
  cost: CostBreakdown;
  escalated: boolean;
}

/** Back-compat export: the defensive judge parser now lives in shared.ts. */
export { parseJudgeVerdict as parseJudgeLine } from "./shared.js";

export async function runCascade(options: CascadeOptions): Promise<CascadeResult> {
  const tiers = options.tiers ?? ["deepseek-flash", "deepseek-v4-pro"];
  const executor = nativeExecutorFromRequester(options.requester);

  const result = await runUniversalCascade({
    resolveExecutor: () => executor,
    tiers,
    judgeRef: tiers[0],
    systemPrompt: options.systemPrompt,
    prompt: options.prompt,
    maxTokens: options.maxTokens ?? 8192,
    judgeThreshold: options.judgeThreshold,
    onEvent: (event: UniversalCascadeEvent) => {
      if (!options.onEvent) return;
      if (event.type === "tier") {
        options.onEvent({ type: "tier", model: event.ref, index: event.index });
      } else if (event.type === "draft") {
        options.onEvent({ type: "draft", model: event.ref, text: event.text, usage: {} });
      } else if (event.type === "judge") {
        options.onEvent({ type: "judge", grade: event.grade, reason: event.reason, confidence: event.confidence });
      } else if (event.type === "escalated") {
        options.onEvent({ type: "escalated", from: event.from, to: event.to });
      } else if (event.type === "final") {
        options.onEvent({ type: "final", model: event.ref, text: event.text });
      }
    },
    signal: options.signal,
  });

  // Map universal usage back onto the DeepSeek shape for cost metering.
  const usage: DeepSeekUsage = {
    prompt_tokens: result.usage.inputTokens,
    completion_tokens: result.usage.outputTokens,
    total_tokens: result.usage.totalTokens,
    prompt_cache_hit_tokens: result.usage.cacheReadTokens,
  };
  const lastCall = result.calls[result.calls.length - 1];
  const cost = lastCall ? computeCost(result.finalRef, usage) : emptyCost(result.finalRef);

  return {
    text: result.text,
    finalModel: result.finalRef,
    judge: result.judge,
    iterations: result.iterations,
    usage,
    cost,
    escalated: result.escalated,
  };
}

function emptyCost(model: string): CostBreakdown {
  return { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0, peak: false, model };
}

// Re-export so existing imports of the executor keep working.
export { createDeepSeekNativeExecutor };
