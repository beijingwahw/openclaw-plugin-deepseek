/**
 * Universal pricing table: per-model unit prices across all providers.
 *
 * Three sources, in priority order:
 * 1. Built-in DeepSeek table (peak/off-peak + cache-aware, exact).
 * 2. LiteLLM community catalog (one fetch, TTL-cached) for every other
 *    provider's per-million-token prices.
 * 3. Unknown models fall back to unpriced (tracked as tokens, cost = null).
 */

export interface ModelPrices {
  /** USD per million input tokens (cache miss). */
  input: number;
  /** USD per million output tokens. */
  output: number;
  /** USD per million cache-read input tokens, when known. */
  cacheRead?: number;
  /** USD per million cache-write input tokens, when known. */
  cacheWrite?: number;
  source: "deepseek-native" | "litellm" | "unpriced";
}

export interface RawUsage {
  input?: number;
  output?: number;
  cacheRead?: number;
  cacheWrite?: number;
  total?: number;
}

export interface PricedUsage extends RawUsage {
  costUsd: number | null;
  priced: boolean;
  source: ModelPrices["source"];
}

interface LiteLLMEntry {
  input_cost_per_token?: number | string;
  output_cost_per_token?: number | string;
  cache_read_input_token_cost?: number | string;
  cache_creation_input_token_cost?: number | string;
}

import { findDeepSeekPriceRow } from "./cost.js";

export const LITELLM_CATALOG_URL =
  "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json";

const LITELLM_TTL_MS = 24 * 60 * 60 * 1000; // daily refresh is plenty

export class PricingTable {
  private litellm: Map<string, ModelPrices> | null = null;
  private litellmFetchedAt = 0;
  private litellmPending: Promise<void> | null = null;
  private readonly fetchFn: typeof fetch;

  constructor(options: { fetchFn?: typeof fetch } = {}) {
    this.fetchFn = options.fetchFn ?? fetch;
  }

  /** Price one usage record for a provider/model ref. */
  price(ref: string, usage: RawUsage, options: { deepSeekPeak?: boolean } = {}): PricedUsage {
    const prices = this.lookup(ref, options.deepSeekPeak);
    if (!prices) {
      return { ...usage, costUsd: null, priced: false, source: "unpriced" };
    }
    // input = total prompt tokens; cacheRead/cacheWrite are subsets of it.
    // Bill: (input - cacheRead - cacheWrite) at the miss rate, cacheRead at
    // the (much cheaper) hit rate, cacheWrite at the write rate.
    const totalInput = usage.input ?? 0;
    const cacheRead = Math.min(usage.cacheRead ?? 0, totalInput);
    const cacheWrite = Math.min(usage.cacheWrite ?? 0, Math.max(0, totalInput - cacheRead));
    const miss = Math.max(0, totalInput - cacheRead - cacheWrite);
    const cost =
      (miss / 1e6) * prices.input +
      (cacheRead / 1e6) * (prices.cacheRead ?? prices.input) +
      (cacheWrite / 1e6) * (prices.cacheWrite ?? prices.input) +
      ((usage.output ?? 0) / 1e6) * prices.output;
    return {
      ...usage,
      costUsd: Math.round(cost * 1e6) / 1e6,
      priced: true,
      source: prices.source,
    };
  }

  lookup(ref: string, deepSeekPeak?: boolean): ModelPrices | null {
    // 1. DeepSeek-native exact table (with peak multiplier).
    const deepseek = lookupDeepSeek(ref, deepSeekPeak ?? null);
    if (deepseek) return deepseek;
    // 2. LiteLLM catalog (synchronous view of the cached map).
    const key = normalizeRef(ref);
    if (this.litellm) {
      const direct = this.litellm.get(key);
      if (direct) return direct;
      // Bare model id: try stripping the provider prefix, then suffix-scan
      // for catalog keys like "anthropic/<bare>".
      const bare = key.includes("/") ? key.split("/").slice(1).join("/") : key;
      const hit = this.litellm.get(bare);
      if (hit) return hit;
      if (!key.includes("/")) {
        for (const [name, prices] of this.litellm) {
          if (name.endsWith(`/${key}`)) return prices;
        }
      }
    }
    return null;
  }

  /** True when the LiteLLM catalog is loaded and fresh. */
  get catalogReady(): boolean {
    return this.litellm !== null && Date.now() - this.litellmFetchedAt < LITELLM_TTL_MS;
  }

  get catalogSize(): number {
    return this.litellm?.size ?? 0;
  }

  /** Loads (or refreshes) the LiteLLM catalog. Safe to call repeatedly. */
  async ensureCatalog(signal?: AbortSignal): Promise<void> {
    if (this.catalogReady) return;
    if (this.litellmPending) return this.litellmPending;
    this.litellmPending = (async () => {
      try {
        const response = await this.fetchFn(LITELLM_CATALOG_URL, { signal });
        if (!response.ok) return;
        const data = (await response.json()) as Record<string, LiteLLMEntry>;
        const map = new Map<string, ModelPrices>();
        for (const [name, entry] of Object.entries(data)) {
          if (!entry || typeof entry !== "object") continue;
          const input = perMillion(entry.input_cost_per_token);
          const output = perMillion(entry.output_cost_per_token);
          if (input === null && output === null) continue;
          map.set(name.toLowerCase(), {
            input: input ?? 0,
            output: output ?? 0,
            cacheRead: perMillion(entry.cache_read_input_token_cost) ?? undefined,
            cacheWrite: perMillion(entry.cache_creation_input_token_cost) ?? undefined,
            source: "litellm",
          });
        }
        this.litellm = map;
        this.litellmFetchedAt = Date.now();
      } catch {
        // Offline or blocked: stay unpriced for non-DeepSeek models.
      } finally {
        this.litellmPending = null;
      }
    })();
    return this.litellmPending;
  }

  /** Test hook. */
  setCatalogForTest(entries: Record<string, ModelPrices>): void {
    this.litellm = new Map(Object.entries(entries).map(([k, v]) => [k.toLowerCase(), v]));
    this.litellmFetchedAt = Date.now();
  }
}

function perMillion(value: number | string | undefined): number | null {
  if (value === undefined || value === null) return null;
  const n = typeof value === "number" ? value : Number.parseFloat(value);
  if (!Number.isFinite(n) || n < 0) return null;
  return n * 1e6; // litellm prices are per-token
}

/** DeepSeek-native pricing with optional peak multiplier (null = auto). */
export function lookupDeepSeek(ref: string, peak: boolean | null): ModelPrices | null {
  const model = normalizeRef(ref).split("/").pop() ?? "";
  // The cascade virtual model prices at the flash tier.
  const info = findDeepSeekPriceRow(model === "deepseek-cascade" ? "deepseek-flash" : model);
  if (!info) return null;
  const mult = peak === null ? 1 : peak ? 2 : 1;
  return {
    input: info.cost.input * mult,
    output: info.cost.output * mult,
    cacheRead: info.cost.cacheRead * mult,
    cacheWrite: info.cost.input * mult,
    source: "deepseek-native",
  };
}


export function normalizeRef(ref: string): string {
  return ref.trim().toLowerCase();
}
