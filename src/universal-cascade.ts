/**
 * Universal speculative cascade — provider-agnostic.
 *
 * The same draft -> judge -> escalate orchestration as cascade.ts, but the
 * executor is pluggable so tiers can be ANY model the gateway can run:
 * deepseek-flash, anthropic/claude-*, openai/gpt-*, google/gemini-*, zai/*, …
 *
 * DeepSeek tiers run through the native executor (thinking control + cache
 * metering); everything else runs through the host llm.complete surface.
 */
import { computeCost } from "./cost.js";
import { DeepSeekClient } from "./deepseek-api.js";
import type { AdaptiveCascadePolicy, CascadeRouteMode } from "./policy.js";
import type { BudgetGuardian } from "./guardian.js";
import type { AuditTrail } from "./audit.js";
import { JUDGE_SYSTEM_PROMPT, parseJudgeVerdict } from "./shared.js";
import type { DeepSeekRequester, DeepSeekUsage } from "./types.js";

export type CascadeReasoning = "none" | "low" | "high" | "max";

export interface UniversalUsage {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
  cacheReadTokens?: number;
  cacheWriteTokens?: number;
  costUsd?: number;
  model?: string;
}

export interface UniversalCompleteParams {
  ref: string;
  systemPrompt: string;
  prompt: string;
  maxTokens?: number;
  reasoning?: CascadeReasoning;
  temperature?: number;
  signal?: AbortSignal;
}

export interface UniversalExecutor {
  label: string;
  complete(params: UniversalCompleteParams): Promise<{ text: string; usage: UniversalUsage }>;
}

/** Host LLM executor: routes any configured provider/model ref via the gateway. */
export function createHostLlmExecutor(
  llm: {
    complete(params: {
      messages: Array<{ role: string; content: string }>;
      model?: string;
      systemPrompt?: string;
      maxTokens?: number;
      temperature?: number;
      reasoning?: string;
      signal?: AbortSignal;
      purpose?: string;
    }): Promise<{ text: string; provider: string; model: string; usage?: { inputTokens?: number; outputTokens?: number; totalTokens?: number; costUsd?: number } }>;
  },
  options: { pluginId: string },
): UniversalExecutor {
  return {
    label: "host-llm",
    async complete(params) {
      const result = await llm.complete({
        messages: [{ role: "user", content: params.prompt }],
        model: params.ref,
        systemPrompt: params.systemPrompt,
        maxTokens: params.maxTokens,
        temperature: params.temperature,
        reasoning: params.reasoning && params.reasoning !== "none" ? params.reasoning : undefined,
        signal: params.signal,
        purpose: `${options.pluginId}:cascade`,
      });
      return {
        text: result.text,
        usage: {
          inputTokens: result.usage?.inputTokens,
          outputTokens: result.usage?.outputTokens,
          totalTokens: result.usage?.totalTokens,
          costUsd: result.usage?.costUsd,
          model: `${result.provider}/${result.model}`,
        },
      };
    },
  };
}

/** Native DeepSeek executor: thinking control + cache-aware cost metering. */
export function createDeepSeekNativeExecutor(client: DeepSeekClient): UniversalExecutor {
  return {
    label: "deepseek-native",
    async complete(params) {
      const body: Record<string, unknown> = {
        model: params.ref,
        messages: [
          { role: "system", content: params.systemPrompt },
          { role: "user", content: params.prompt },
        ],
        max_tokens: params.maxTokens ?? 8192,
        stream: true,
        stream_options: { include_usage: true },
      };
      if (params.reasoning === "none") {
        body.thinking = { type: "disabled" };
        if (params.temperature !== undefined) body.temperature = params.temperature;
      } else {
        body.thinking = { type: "enabled" };
        body.reasoning_effort = params.reasoning ?? "low";
      }
      let text = "";
      let usage: DeepSeekUsage = {};
      for await (const event of client.streamChat(body, params.signal)) {
        if (event.type === "usage") usage = event.usage;
        else if (event.type === "text") text += event.text;
      }
      const cost = computeCost(params.ref, usage);
      return {
        text,
        usage: {
          inputTokens: usage.prompt_tokens,
          outputTokens: usage.completion_tokens,
          totalTokens: usage.total_tokens,
          cacheReadTokens: usage.prompt_cache_hit_tokens,
          costUsd: cost.total,
          model: params.ref,
        },
      };
    },
  };
}

export interface UniversalCascadeOptions {
  /** Executor resolution per tier ref. */
  resolveExecutor: (ref: string) => UniversalExecutor;
  tiers: string[];
  judgeRef?: string;
  systemPrompt: string;
  prompt: string;
  maxTokens?: number;
  /** Escalate when grade=escalate OR judge confidence < threshold. 0 disables. */
  judgeThreshold?: number;
  /** Adaptive policy: skip the judge on high-acceptance shapes, start strong on low ones. */
  policy?: AdaptiveCascadePolicy;
  /** Budget guardian: economic routing under budget pressure. */
  guardian?: BudgetGuardian;
  /** Audit trail: structured decision log. */
  audit?: AuditTrail;
  onEvent?: (event: UniversalCascadeEvent) => void;
  signal?: AbortSignal;
}

export type UniversalCascadeEvent =
  | { type: "tier"; ref: string; executor: string; index: number }
  | { type: "draft"; ref: string; text: string }
  | { type: "judge"; grade: "accept" | "escalate"; reason: string; confidence: number }
  | { type: "escalated"; from: string; to: string }
  | { type: "final"; ref: string; text: string }
  | { type: "routeMode"; mode: CascadeRouteMode; reason: string };

export interface UniversalCascadeResult {
  text: string;
  finalRef: string;
  judge: { grade: "accept" | "escalate"; confidence: number; reason: string };
  iterations: number;
  usage: UniversalUsage;
  escalated: boolean;
  /** Per-call trace for transparency. */
  calls: Array<{ ref: string; role: "draft" | "judge" | "escalated-draft"; usage: UniversalUsage }>;
}

export { parseJudgeVerdict as parseUniversalJudgeLine };

const skipAuditCounter = new Map<string, number>();

/** Every 4th skip-judge run for a tier gets audited to guard against drift. */
function shouldAuditSkip(tier0: string, _prompt: string): boolean {
  const n = (skipAuditCounter.get(tier0) ?? 0) + 1;
  skipAuditCounter.set(tier0, n);
  return n % 4 === 0;
}

export async function runUniversalCascade(options: UniversalCascadeOptions): Promise<UniversalCascadeResult> {
  const tiers = options.tiers.length > 0 ? options.tiers : ["deepseek-flash", "deepseek-v4-pro"];
  const judgeRef = options.judgeRef ?? tiers[0];
  const maxTokens = options.maxTokens ?? 8192;
  const calls: UniversalCascadeResult["calls"] = [];
  const total: UniversalUsage = {};
  let finalText = "";
  let finalRef = tiers[0];
  let judge: UniversalCascadeResult["judge"] = { grade: "accept", confidence: 1, reason: "no judge needed" };
  let iterations = 0;

  const judgeExecutor = options.resolveExecutor(judgeRef);
  const recordPolicyAccepted = (): void => {
    options.policy?.record({
      tier0: tiers[0],
      finalTier: finalRef,
      prompt: options.prompt,
      accepted: true,
      escalated: false,
    });
  };
  const recordPolicyEscalated = (): void => {
    options.policy?.record({
      tier0: tiers[0],
      finalTier: finalRef,
      prompt: options.prompt,
      accepted: false,
      escalated: true,
    });
  };

  // Economic routing first: the guardian can override everything under pressure.
  const advice = options.guardian?.routeAdvice(tiers);
  let effectiveTiers = tiers;
  let guardianSkipJudge = false;
  if (advice && advice.mode !== "normal") {
    effectiveTiers = advice.tiers;
    guardianSkipJudge = advice.skipJudge;
    options.audit?.record("cascade-route", advice.reason, { mode: advice.mode, tiers: advice.tiers });
  }

  // Adaptive routing: consult the learning policy before the first call.
  let routeMode: CascadeRouteMode = "judge";
  let startIndex = 0;
  const recommendation = options.policy?.recommend(tiers[0], options.prompt);
  if (recommendation) {
    routeMode = recommendation.mode;
    options.onEvent?.({ type: "routeMode", mode: recommendation.mode, reason: recommendation.reason });
    if (recommendation.mode === "direct-strong" && tiers.length > 1) {
      startIndex = 1; // skip the weak tier entirely
    }
  }
  // Every 4th skip-judge run still pays for a judge call to guard against
  // drift (exploration in explore/exploit terms). The guardian's emergency
  // mode skips the judge unconditionally.
  let skipJudgeThisRun = guardianSkipJudge;
  if (!skipJudgeThisRun && routeMode === "skip-judge") {
    skipJudgeThisRun = !shouldAuditSkip(tiers[0], options.prompt);
    if (skipJudgeThisRun) {
      options.audit?.record("cascade-skip-judge", recommendation?.reason ?? "policy fast path", {
        tier0: tiers[0],
      });
    }
  }

  for (let index = startIndex; index < effectiveTiers.length; index++) {
    options.signal?.throwIfAborted();
    const ref = tiers[index];
    const executor = options.resolveExecutor(ref);
    const isLastTier = index === effectiveTiers.length - 1;
    iterations += 1;
    options.onEvent?.({ type: "tier", ref, executor: executor.label, index });

    const draft = await executor.complete({
      ref,
      systemPrompt: options.systemPrompt,
      prompt: options.prompt,
      maxTokens,
      // First tier drafts cheap; escalated tiers think hard.
      reasoning: index === 0 ? "low" : "max",
      signal: options.signal,
    });
    mergeUniversalUsage(total, draft.usage);
    calls.push({ ref, role: index === 0 ? "draft" : "escalated-draft", usage: draft.usage });
    finalText = draft.text;
    finalRef = ref;
    options.onEvent?.({ type: "draft", ref, text: draft.text });

    if (isLastTier) {
      judge = { grade: "accept", confidence: 1, reason: "final tier" };
      options.onEvent?.({ type: "final", ref, text: draft.text });
      break;
    }

    // Judge with the configured judge model (defaults to the cheap tier).
    // Adaptive skip: trust the draft on high-acceptance shapes (audited every 4th run).
    if (skipJudgeThisRun && isLastTier === false && index === startIndex) {
      judge = { grade: "accept", confidence: 1, reason: `policy fast path (${recommendation?.reason ?? "skip-judge"})` };
      options.onEvent?.({ type: "judge", grade: "accept", reason: judge.reason, confidence: 1 });
      options.onEvent?.({ type: "final", ref, text: draft.text });
      recordPolicyAccepted();
      break;
    }
    const judgeResult = await judgeExecutor.complete({
      ref: judgeRef,
      systemPrompt: JUDGE_SYSTEM_PROMPT,
      prompt: `Task:\n${options.prompt}\n\nDraft answer:\n${draft.text}\n\nJudge the draft now.`,
      maxTokens: 200,
      reasoning: "none",
      temperature: 0,
      signal: options.signal,
    });
    mergeUniversalUsage(total, judgeResult.usage);
    calls.push({ ref: judgeRef, role: "judge", usage: judgeResult.usage });
    const verdict = parseJudgeVerdict(judgeResult.text);
    judge = verdict;
    options.onEvent?.({ type: "judge", grade: verdict.grade, reason: verdict.reason, confidence: verdict.confidence });

    const threshold = options.judgeThreshold ?? 0;
    const lowConfidence = threshold > 0 && verdict.confidence < threshold;
    const shouldEscalate = verdict.grade === "escalate" || lowConfidence;
    if (lowConfidence && verdict.grade === "accept") {
      judge = { ...verdict, reason: `confidence ${verdict.confidence.toFixed(2)} below threshold ${threshold}` };
    }

    if (!shouldEscalate) {
      options.onEvent?.({ type: "final", ref, text: draft.text });
      recordPolicyAccepted();
      break;
    }
    options.onEvent?.({ type: "escalated", from: ref, to: effectiveTiers[index + 1] });
    options.audit?.record("cascade-escalation", `${ref} -> ${effectiveTiers[index + 1]}: ${verdict.reason}`, {
      from: ref,
      to: effectiveTiers[index + 1],
      confidence: verdict.confidence,
    });
  }
  if (finalRef !== tiers[0] || routeMode === "direct-strong") {
    recordPolicyEscalated();
  }

  total.model = "cascade";
  return {
    text: finalText,
    finalRef,
    judge,
    iterations,
    usage: total,
    escalated: finalRef !== tiers[0],
    calls,
  };
}

/** Splits tier refs into DeepSeek-native vs host-routed refs. */
export function isNativeDeepSeekRef(ref: string): boolean {
  return /^deepseek-[a-z0-9.-]+$/i.test(ref) || ref === "deepseek-cascade";
}

function mergeUniversalUsage(target: UniversalUsage, source: UniversalUsage): void {
  target.inputTokens = (target.inputTokens ?? 0) + (source.inputTokens ?? 0);
  target.outputTokens = (target.outputTokens ?? 0) + (source.outputTokens ?? 0);
  target.totalTokens = (target.totalTokens ?? 0) + (source.totalTokens ?? 0);
  target.cacheReadTokens = (target.cacheReadTokens ?? 0) + (source.cacheReadTokens ?? 0);
  target.costUsd = Math.round(((target.costUsd ?? 0) + (source.costUsd ?? 0)) * 1e6) / 1e6;
}

/** Legacy-compatible requester passthrough for tests of the native executor. */
export function nativeExecutorFromRequester(requester: DeepSeekRequester): UniversalExecutor {
  return {
    label: "deepseek-native",
    async complete(params) {
      const body: Record<string, unknown> = {
        model: params.ref,
        messages: [
          { role: "system", content: params.systemPrompt },
          { role: "user", content: params.prompt },
        ],
        max_tokens: params.maxTokens ?? 8192,
        stream: true,
        stream_options: { include_usage: true },
      };
      if (params.reasoning === "none") {
        body.thinking = { type: "disabled" };
      } else {
        body.thinking = { type: "enabled" };
        body.reasoning_effort = params.reasoning ?? "low";
      }
      let text = "";
      let usage: DeepSeekUsage = {};
      for await (const event of requester.streamChat(body, params.signal)) {
        if (event.type === "usage") usage = event.usage;
        else if (event.type === "text") text += event.text;
      }
      const cost = computeCost(params.ref, usage);
      return {
        text,
        usage: {
          inputTokens: usage.prompt_tokens,
          outputTokens: usage.completion_tokens,
          totalTokens: usage.total_tokens,
          cacheReadTokens: usage.prompt_cache_hit_tokens,
          costUsd: cost.total,
          model: params.ref,
        },
      };
    },
  };
}
