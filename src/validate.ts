/**
 * Runtime boundary validation.
 *
 * External inputs (SSE frames, transcript lines, provider responses) cross
 * trust boundaries. These guards replace blind `as` casts: parse, verify,
 * and shape unknown data into typed values with defensive fallbacks.
 */
import type { DeepSeekUsage } from "./types.js";

/** True when the value is a plain non-null object. */
export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Reads a finite number field, returns undefined otherwise. */
export function readNumber(source: Record<string, unknown>, key: string): number | undefined {
  const value = source[key];
  if (typeof value === "number" && Number.isFinite(value)) return value;
  return undefined;
}

/** Reads a bounded string field. */
export function readString(source: Record<string, unknown>, key: string, maxLength = 1_000_000): string | undefined {
  const value = source[key];
  if (typeof value === "string" && value.length <= maxLength) return value;
  return undefined;
}

/**
 * Validates an SSE usage frame into a DeepSeekUsage, tolerating missing and
 * extra fields. Never throws: malformed numbers are dropped, not propagated.
 */
export function parseUsageFrame(raw: unknown): DeepSeekUsage | null {
  if (!isRecord(raw)) return null;
  const usage: DeepSeekUsage = {};
  const promptTokens = readNumber(raw, "prompt_tokens");
  const completionTokens = readNumber(raw, "completion_tokens");
  const totalTokens = readNumber(raw, "total_tokens");
  const cacheHit = readNumber(raw, "prompt_cache_hit_tokens");
  const cacheMiss = readNumber(raw, "prompt_cache_miss_tokens");
  if (promptTokens !== undefined) usage.prompt_tokens = promptTokens;
  if (completionTokens !== undefined) usage.completion_tokens = completionTokens;
  if (totalTokens !== undefined) usage.total_tokens = totalTokens;
  if (cacheHit !== undefined) usage.prompt_cache_hit_tokens = cacheHit;
  if (cacheMiss !== undefined) usage.prompt_cache_miss_tokens = cacheMiss;
  // Nested details (reasoning tokens, cached tokens) — best effort.
  const details = raw.completion_tokens_details;
  if (isRecord(details)) {
    const reasoning = readNumber(details, "reasoning_tokens");
    if (reasoning !== undefined) {
      usage.completion_tokens_details = { reasoning_tokens: reasoning };
    }
  }
  const promptDetails = raw.prompt_tokens_details;
  if (isRecord(promptDetails)) {
    const cached = readNumber(promptDetails, "cached_tokens");
    if (cached !== undefined) {
      usage.prompt_tokens_details = { cached_tokens: cached };
    }
  }
  return usage;
}

/**
 * Validates an SSE tool-call delta fragment. Returns null when malformed.
 */
export function parseToolCallFragment(
  raw: unknown,
): { index: number; id?: string; name?: string; argsDelta?: string } | null {
  if (!isRecord(raw)) return null;
  const index = readNumber(raw, "index") ?? 0;
  const result: { index: number; id?: string; name?: string; argsDelta?: string } = { index };
  const id = readString(raw, "id", 256);
  if (id !== undefined) result.id = id;
  const fn = raw.function;
  if (isRecord(fn)) {
    const name = readString(fn, "name", 256);
    if (name !== undefined) result.name = name;
    const args = readString(fn, "arguments", 100_000);
    if (args !== undefined) result.argsDelta = args;
  }
  return result;
}

/**
 * Safe JSON.parse returning unknown (for boundary data). Use isRecord +
 * readers to inspect the result.
 */
export function safeJsonParse(text: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    // Boundary data is untrusted: invalid JSON maps to undefined and callers
    // treat it as absent rather than crashing.
    return undefined;
  }
}
