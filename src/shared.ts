/**
 * Shared helpers — single source of truth for cross-module utilities.
 * Every helper here replaces a former per-module copy.
 */
import type { DeepSeekUsage } from "./types.js";

/** Judge verdict shape shared by all cascade engines. */
export interface JudgeVerdict {
  grade: "accept" | "escalate";
  confidence: number;
  reason: string;
}

/**
 * Defensive judge-output parser (single implementation).
 * Non-JSON flow fails safe: empty output escalates, unparsable accepts.
 */
export function parseJudgeVerdict(line: string): JudgeVerdict {
  const jsonMatch = line.match(/\{[^{}]*\}/);
  if (jsonMatch) {
    try {
      const parsed = JSON.parse(jsonMatch[0]) as { grade?: unknown; confidence?: unknown; reason?: unknown };
      return {
        grade: parsed.grade === "escalate" ? "escalate" : "accept",
        confidence:
          typeof parsed.confidence === "number" && Number.isFinite(parsed.confidence)
            ? Math.min(1, Math.max(0, parsed.confidence))
            : 0.5,
        reason: typeof parsed.reason === "string" ? parsed.reason.slice(0, 200) : "",
      };
    } catch {
      // heuristic below
    }
  }
  const trimmed = line.trim();
  if (trimmed.length === 0) return { grade: "escalate", reason: "empty judge output", confidence: 0 };
  return { grade: "accept", reason: "judge output unparsable; defaulting to accept", confidence: 0.5 };
}

/** The strict judge system prompt shared by every cascade engine. */
export const JUDGE_SYSTEM_PROMPT = `You are a strict quality judge. Score the draft answer for the user's task on correctness, completeness, and specificity.
Reply with exactly one line of JSON, no other text:
{"grade":"accept|escalate","confidence":0.0-1.0,"reason":"one short sentence"}
Grade "escalate" when the answer is wrong, evasive, missing key parts, or hand-waves specifics. Grade "accept" otherwise.`;

/** Merges DeepSeek usage objects in place (single implementation). */
export function mergeDeepSeekUsage(target: DeepSeekUsage, source: DeepSeekUsage): void {
  target.prompt_tokens = (target.prompt_tokens ?? 0) + (source.prompt_tokens ?? 0);
  target.completion_tokens = (target.completion_tokens ?? 0) + (source.completion_tokens ?? 0);
  target.total_tokens = (target.total_tokens ?? 0) + (source.total_tokens ?? 0);
  target.prompt_cache_hit_tokens = (target.prompt_cache_hit_tokens ?? 0) + (source.prompt_cache_hit_tokens ?? 0);
  target.prompt_cache_miss_tokens = (target.prompt_cache_miss_tokens ?? 0) + (source.prompt_cache_miss_tokens ?? 0);
}

/** Clamp helper (single implementation; replaces clamp + clampInt copies). */
export function clampNumber(value: number | undefined, min: number, max: number, fallback: number): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return fallback;
  const n = Math.round(value);
  if (n < min) return min;
  if (n > max) return max;
  return n;
}

/** Shortens text for logs/previews. */
export function truncateText(text: string, max: number): string {
  return text.length <= max ? text : `${text.slice(0, max)}…`;
}

/** Stable key-order JSON for repeat-detection hashing. */
export function stableStringify(value: unknown): string {
  try {
    return (
      JSON.stringify(value, (_k, v) => {
        if (v && typeof v === "object" && !Array.isArray(v)) {
          const sorted: Record<string, unknown> = {};
          for (const key of Object.keys(v as Record<string, unknown>).sort()) {
            sorted[key] = (v as Record<string, unknown>)[key];
          }
          return sorted;
        }
        return v;
      }) ?? String(value)
    );
  } catch {
    // Circular references: degrade to a stable string form so repeat
    // detection still works on exotic argument shapes.
    return String(value);
  }
}

/** Bounded preview of tool output (channel/UI safe length). */
export const TOOL_PREVIEW_LIMIT = 200;
export function previewOfText(output: string): string {
  return truncateText(output, TOOL_PREVIEW_LIMIT);
}
