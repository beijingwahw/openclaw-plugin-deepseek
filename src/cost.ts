/**
 * Cost metering for DeepSeek usage.
 *
 * Prices follow the official 2026-09-10 catalog (USD per million tokens,
 * off-peak). Peak windows double all prices. Cache hits are billed at ~1/50
 * of cache-miss input, so agent loops with stable prefixes benefit massively.
 */
import { CURRENT_MODELS, type DeepSeekModelInfo, type DeepSeekUsage } from "./types.js";

/** DeepSeek peak windows (UTC): weekdays 01:00-04:00 and 06:00-10:00. */
export function isPeakNow(now: Date = new Date()): boolean {
  const day = now.getUTCDay(); // 0 = Sunday
  if (day === 0 || day === 6) return false;
  const hour = now.getUTCHours();
  return (hour >= 1 && hour < 4) || (hour >= 6 && hour < 10);
}

export interface CostBreakdown {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
  total: number;
  peak: boolean;
  model: string;
}

export function computeCost(
  modelId: string,
  usage: DeepSeekUsage,
  options: { peak?: boolean; now?: Date } = {},
): CostBreakdown {
  const peak = options.peak ?? isPeakNow(options.now);
  const info = findModel(modelId);
  const hit = usage.prompt_cache_hit_tokens ?? usage.prompt_tokens_details?.cached_tokens ?? 0;
  const promptTotal = usage.prompt_tokens ?? hit;
  const miss = Math.max(0, (usage.prompt_cache_miss_tokens ?? promptTotal - hit));
  const output = usage.completion_tokens ?? 0;
  if (!info) {
    return { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0, peak, model: modelId };
  }
  const mult = peak ? 2 : 1;
  const inputCost = (miss / 1e6) * info.cost.input * mult;
  const cacheReadCost = (hit / 1e6) * info.cost.cacheRead * mult;
  const outputCost = (output / 1e6) * info.cost.output * mult;
  return {
    input: round6(inputCost),
    output: round6(outputCost),
    cacheRead: round6(cacheReadCost),
    cacheWrite: 0,
    total: round6(inputCost + cacheReadCost + outputCost),
    peak,
    model: modelId,
  };
}

/** Cascade rows price at the flash tier (flash answers; escalation is rare). */
function findModel(modelId: string): DeepSeekModelInfo | undefined {
  if (modelId === "deepseek-cascade") {
    return findDeepSeekPriceRow("deepseek-flash");
  }
  return findDeepSeekPriceRow(modelId);
}

/** Single DeepSeek price table shared with the universal PricingTable. */
export function findDeepSeekPriceRow(modelId: string): DeepSeekModelInfo | undefined {
  return CURRENT_MODELS.find((m) => m.id === modelId);
}

export function unitPriceTable(peak: boolean): string {
  const mult = peak ? 2 : 1;
  return CURRENT_MODELS.map((m) => {
    const inPrice = (m.cost.input * mult).toFixed(3);
    const hitPrice = (m.cost.cacheRead * mult).toFixed(4);
    const outPrice = (m.cost.output * mult).toFixed(3);
    return `- ${m.id}: input $${inPrice}/M (cache-hit $${hitPrice}/M), output $${outPrice}/M`;
  }).join("\n");
}

/** Human-readable peak window description for reports. */
export const PEAK_WINDOW_DESCRIPTION =
  "peak (2x pricing) Mon-Fri 01:00-04:00 & 06:00-10:00 UTC (= Beijing 09:00-12:00 & 14:00-18:00); off-peak otherwise; Chinese public holidays excluded upstream";

function round6(n: number): number {
  return Math.round(n * 1e6) / 1e6;
}
