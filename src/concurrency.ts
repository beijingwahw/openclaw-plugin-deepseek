/**
 * AIMD concurrency governor — TCP congestion control for DeepSeek requests.
 *
 * DeepSeek enforces account-level concurrency limits (flash 2500, pro 500).
 * This governor mirrors them client-side with an additive-increase /
 * multiplicative-decrease window per model:
 *
 *   success    -> limit += 1        (linear probing upward, up to max)
 *   429        -> limit = limit / 2 (multiplicative backoff, floor 1)
 *   3x 429     -> circuit-break cooldown (30s) before the next acquire
 *
 * All DeepSeekClient requests flow through acquire/release, so every module
 * (harness, cascade, tools) shares one adaptive window per model.
 */

export type GovernorOutcome = "success" | "rate-limit" | "error";

interface ModelState {
  limit: number;
  inFlight: number;
  cooldownUntil: number;
  consecutiveRateLimits: number;
  queue: Array<() => void>;
}

export interface GovernorStats {
  model: string;
  limit: number;
  inFlight: number;
  queued: number;
  cooldownActive: boolean;
  consecutiveRateLimits: number;
}

const DEFAULT_INITIAL_LIMIT = 4;
const DEFAULT_MAX_LIMIT = 16;
const CONSECUTIVE_LIMIT_FOR_COOLDOWN = 3;
const COOLDOWN_MS = 30_000;

export class ConcurrencyGovernor {
  private readonly models = new Map<string, ModelState>();
  private readonly initialLimit: number;
  private readonly maxLimit: number;
  private readonly now: () => number;

  constructor(options: { initialLimit?: number; maxLimit?: number; now?: () => number } = {}) {
    this.initialLimit = options.initialLimit ?? DEFAULT_INITIAL_LIMIT;
    this.maxLimit = options.maxLimit ?? DEFAULT_MAX_LIMIT;
    this.now = options.now ?? Date.now;
  }

  private stateOf(model: string): ModelState {
    let state = this.models.get(model);
    if (!state) {
      state = {
        limit: Math.min(this.initialLimit, this.maxLimit),
        inFlight: 0,
        cooldownUntil: 0,
        consecutiveRateLimits: 0,
        queue: [],
      };
      this.models.set(model, state);
    }
    return state;
  }

  /** Resolves when a concurrency slot is available (FIFO queue, cooldown-aware). */
  async acquire(model: string): Promise<void> {
    const state = this.stateOf(model);
    // Circuit-break cooldown: wait it out before queueing.
    const now = this.now();
    if (state.cooldownUntil > now) {
      await sleep(state.cooldownUntil - now, this.now);
    }
    if (state.inFlight < state.limit) {
      state.inFlight += 1;
      return;
    }
    await new Promise<void>((resolve) => state.queue.push(resolve));
    state.inFlight += 1;
  }

  /** Releases a slot and updates the window per AIMD. */
  release(model: string, outcome: GovernorOutcome): void {
    const state = this.stateOf(model);
    if (state.inFlight > 0) state.inFlight -= 1;

    if (outcome === "success") {
      state.limit = Math.min(state.limit + 1, this.maxLimit);
      state.consecutiveRateLimits = 0;
    } else if (outcome === "rate-limit") {
      state.limit = Math.max(1, Math.floor(state.limit / 2));
      state.consecutiveRateLimits += 1;
      if (state.consecutiveRateLimits >= CONSECUTIVE_LIMIT_FOR_COOLDOWN) {
        state.cooldownUntil = this.now() + COOLDOWN_MS;
        state.consecutiveRateLimits = 0;
      }
    }
    // "error" leaves the window unchanged.

    // Wake the next waiter, if any.
    const next = state.queue.shift();
    if (next) next();
  }

  /** Snapshot for health reports. */
  stats(model?: string): GovernorStats[] {
    const targets = model ? [model] : [...this.models.keys()];
    return targets.map((name) => {
      const state = this.stateOf(name);
      return {
        model: name,
        limit: state.limit,
        inFlight: state.inFlight,
        queued: state.queue.length,
        cooldownActive: state.cooldownUntil > this.now(),
        consecutiveRateLimits: state.consecutiveRateLimits,
      };
    });
  }

  /** Human-readable governor state. */
  reportTable(): string[] {
    const all = this.stats();
    if (all.length === 0) return ["(concurrency: no requests yet)"];
    return all.map(
      (s) =>
        `  ${s.model}: window ${s.limit}, in-flight ${s.inFlight}, queued ${s.queued}${s.cooldownActive ? " [COOLDOWN]" : ""}`,
    );
  }

  /** Test hook. */
  resetForTest(): void {
    this.models.clear();
  }
}

function sleep(ms: number, _now: () => number): Promise<void> {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, Math.max(0, ms));
    timer.unref?.(); // cooldown waits must not pin the event loop
  });
}
