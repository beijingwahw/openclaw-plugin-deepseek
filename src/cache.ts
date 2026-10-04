/**
 * Deterministic response cache for expensive tool calls.
 *
 * research/reason/cascade calls with identical prompts can reuse the previous
 * answer while it is fresh: LRU eviction + per-entry TTL. Cache keys hash the
 * (kind, model, prompt) tuple; hits are free (no provider call), misses run
 * the underlying tool and store the result.
 */
import { createHash } from "node:crypto";

interface CacheEntry<T> {
  value: T;
  createdAt: number;
  expiresAt: number;
  /** LRU order stamp: larger = more recent. */
  lastUsed: number;
}

export interface ResponseCacheStats {
  entries: number;
  hits: number;
  misses: number;
  hitRate: number;
}

const DEFAULT_MAX_ENTRIES = 64;
const DEFAULT_TTL_MS = 15 * 60_000; // 15 minutes: fresh enough for repeat questions

export class ResponseCache<T = unknown> {
  private readonly store = new Map<string, CacheEntry<T>>();
  private readonly maxEntries: number;
  private readonly ttlMs: number;
  private readonly now: () => number;
  private hits = 0;
  private misses = 0;

  constructor(options: { maxEntries?: number; ttlMs?: number; now?: () => number } = {}) {
    this.maxEntries = options.maxEntries ?? DEFAULT_MAX_ENTRIES;
    this.ttlMs = options.ttlMs ?? DEFAULT_TTL_MS;
    this.now = options.now ?? Date.now;
  }

  private keyOf(kind: string, model: string, prompt: string): string {
    return createHash("sha256").update(`${kind}\u0000${model}\u0000${prompt}`).digest("hex");
  }

  get(kind: string, model: string, prompt: string): T | undefined {
    const key = this.keyOf(kind, model, prompt);
    const entry = this.store.get(key);
    if (!entry) {
      this.misses += 1;
      return undefined;
    }
    const now = this.now();
    if (entry.expiresAt <= now) {
      this.store.delete(key);
      this.misses += 1;
      return undefined;
    }
    entry.lastUsed = now;
    this.hits += 1;
    return entry.value;
  }

  set(kind: string, model: string, prompt: string, value: T, ttlMs?: number): void {
    const key = this.keyOf(kind, model, prompt);
    const now = this.now();
    // LRU eviction: drop the oldest entry when full.
    if (this.store.size >= this.maxEntries && !this.store.has(key)) {
      let oldestKey: string | null = null;
      let oldestUsed = Number.POSITIVE_INFINITY;
      for (const [k, e] of this.store) {
        if (e.lastUsed < oldestUsed) {
          oldestUsed = e.lastUsed;
          oldestKey = k;
        }
      }
      if (oldestKey !== null) this.store.delete(oldestKey);
    }
    this.store.set(key, {
      value,
      createdAt: now,
      expiresAt: now + (ttlMs ?? this.ttlMs),
      lastUsed: now,
    });
  }

  /** Convenience wrapper: read-through caching around any async producer. */
  async through(
    kind: string,
    model: string,
    prompt: string,
    producer: () => Promise<T>,
    ttlMs?: number,
  ): Promise<{ value: T; cached: boolean }> {
    const hit = this.get(kind, model, prompt);
    if (hit !== undefined) return { value: hit, cached: true };
    const value = await producer();
    this.set(kind, model, prompt, value, ttlMs);
    return { value, cached: false };
  }

  stats(): ResponseCacheStats {
    const total = this.hits + this.misses;
    return {
      entries: this.store.size,
      hits: this.hits,
      misses: this.misses,
      hitRate: total === 0 ? 0 : Math.round((this.hits / total) * 100) / 100,
    };
  }

  clearForTest(): void {
    this.store.clear();
    this.hits = 0;
    this.misses = 0;
  }
}
