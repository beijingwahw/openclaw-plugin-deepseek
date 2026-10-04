/**
 * Adaptive cascade policy — the plugin's learning nervous system.
 *
 * Continuously learns tier-0 acceptance rates per task shape (bucket x length)
 * and recommends routing modes for future cascades:
 *
 * - "skip-judge":   tier-0 is highly reliable for this shape -> trust drafts
 *                   without paying for a judge call (fast path).
 * - "direct-strong": tier-0 keeps failing for this task type -> skip it and
 *                   start directly at the stronger tier.
 * - "judge":        not enough evidence yet -> normal draft+judge flow.
 *
 * Like a CPU branch predictor: cheap history shapes expensive future routes.
 * Stats decay (14 days) and persist across restarts.
 */

export type CascadeRouteMode = "judge" | "skip-judge" | "direct-strong";

export type TaskBucket = "code" | "reasoning" | "factual" | "creative" | "general";
export type LengthBucket = "short" | "medium" | "long";

export interface PolicyRecordInput {
  tier0: string;
  finalTier: string;
  prompt: string;
  /** True when the tier-0 draft was accepted (no escalation). */
  accepted: boolean;
  escalated: boolean;
}

export interface PolicyRecommendation {
  mode: CascadeRouteMode;
  reason: string;
  samples: number;
  acceptanceRate: number;
}

interface BucketStats {
  samples: number;
  accepted: number;
  lastTs: number;
}

const POLICY_PATH_SUFFIX = "openclaw-deepseek/cascade-policy.json";
const DECAY_MS = 14 * 24 * 3_600_000;
const SKIP_JUDGE_MIN_SAMPLES = 6;
const SKIP_JUDGE_ACCEPTANCE = 0.85;
const DIRECT_STRONG_MIN_SAMPLES = 10;
const DIRECT_STRONG_ACCEPTANCE = 0.3;
const MAX_BUCKETS = 256;

/** Heuristic task-shape classifier for policy bucketing. */
export function classifyTask(prompt: string): TaskBucket {
  const p = prompt.toLowerCase();
  if (/(code|function|bug|implement|refactor|api|regex|compile|typescript|python|script)/.test(p)) return "code";
  if (/(prove|derive|calculate|solve|math|logic|analyz|why does|step by step)/.test(p)) return "reasoning";
  if (/(who |when |where |what is|latest|news|price|compare|list |capital)/.test(p)) return "factual";
  if (/(write a story|poem|creative|brainstorm|slogan|marketing|draft a)/.test(p)) return "creative";
  return "general";
}

export function lengthBucketOf(chars: number): LengthBucket {
  if (chars < 400) return "short";
  if (chars < 2000) return "medium";
  return "long";
}

export class AdaptiveCascadePolicy {
  private stats = new Map<string, BucketStats>();
  private readonly now: () => number;
  private readonly stateDir: string | undefined;
  private persistTimer: ReturnType<typeof setTimeout> | undefined;
  private loading: Promise<void> | null = null;

  constructor(options: { now?: () => number; stateDir?: string } = {}) {
    this.now = options.now ?? Date.now;
    this.stateDir = options.stateDir;
    if (this.stateDir) {
      this.loading = this.load().catch(() => {});
    }
  }

  private async load(): Promise<void> {
    const path = this.snapshotPath();
    if (!path) return;
    const { readFile } = await import("node:fs/promises");
    try {
      const raw = await readFile(path, "utf8");
      const parsed = JSON.parse(raw) as { buckets?: Record<string, BucketStats> };
      if (parsed.buckets && typeof parsed.buckets === "object") {
        const cutoff = this.now() - DECAY_MS;
        for (const [key, value] of Object.entries(parsed.buckets)) {
          if (value && typeof value.samples === "number" && (value.lastTs ?? 0) >= cutoff) {
            this.stats.set(key, value);
          }
        }
      }
    } catch {
      // No snapshot yet: fine.
    }
  }

  private snapshotPath(): string | null {
    if (!this.stateDir) return null;
    return `${this.stateDir.replace(/\/+$/, "")}/${POLICY_PATH_SUFFIX}`;
  }

  private schedulePersist(): void {
    const path = this.snapshotPath();
    if (!path || this.persistTimer) return;
    this.persistTimer = setTimeout(() => {
      this.persistTimer = undefined;
      void this.persistNow().catch(() => {});
    }, 2_000);
    this.persistTimer.unref?.(); // persistence debounce must not pin the process
  }

  private async persistNow(): Promise<void> {
    const path = this.snapshotPath();
    if (!path) return;
    const { mkdir, writeFile, rename } = await import("node:fs/promises");
    const dir = path.slice(0, path.lastIndexOf("/"));
    const tmp = `${path}.tmp`;
    await mkdir(dir, { recursive: true });
    await writeFile(tmp, JSON.stringify({ savedAt: this.now(), buckets: Object.fromEntries(this.stats) }), "utf8");
    await rename(tmp, path);
  }

  /** Resolves when any startup load has settled (test hook). */
  async ready(): Promise<void> {
    await this.loading;
  }

  /** Records one cascade outcome into the learning stats. */
  record(input: PolicyRecordInput): void {
    const task = classifyTask(input.prompt);
    const len = lengthBucketOf(input.prompt.length);
    const accepted = input.accepted && !input.escalated;
    const keys = [
      `${input.tier0}|${task}|${len}`, // fine-grained: skip-judge signal
      `${input.tier0}|${task}`, // coarse: direct-strong signal
    ];
    const ts = this.now();
    for (const key of keys) {
      const bucket = this.stats.get(key) ?? { samples: 0, accepted: 0, lastTs: ts };
      bucket.samples += 1;
      if (accepted) bucket.accepted += 1;
      bucket.lastTs = ts;
      this.stats.set(key, bucket);
    }
    // Bound the table.
    if (this.stats.size > MAX_BUCKETS) {
      const oldest = [...this.stats.entries()].sort((a, b) => a[1].lastTs - b[1].lastTs);
      for (const [key] of oldest.slice(0, this.stats.size - MAX_BUCKETS)) {
        this.stats.delete(key);
      }
    }
    this.schedulePersist();
  }

  private rate(key: string): { samples: number; acceptanceRate: number } {
    const bucket = this.stats.get(key);
    if (!bucket || bucket.samples === 0) return { samples: 0, acceptanceRate: 0 };
    return { samples: bucket.samples, acceptanceRate: bucket.accepted / bucket.samples };
  }

  /** Recommends a routing mode for an upcoming cascade. */
  recommend(tier0: string, prompt: string): PolicyRecommendation {
    const task = classifyTask(prompt);
    const len = lengthBucketOf(prompt.length);

    // Coarse signal first: has tier-0 been failing broadly for this task type?
    const coarse = this.rate(`${tier0}|${task}`);
    if (coarse.samples >= DIRECT_STRONG_MIN_SAMPLES && coarse.acceptanceRate < DIRECT_STRONG_ACCEPTANCE) {
      return {
        mode: "direct-strong",
        reason: `tier-0 acceptance ${(coarse.acceptanceRate * 100).toFixed(0)}% over ${coarse.samples} ${task} tasks`,
        samples: coarse.samples,
        acceptanceRate: coarse.acceptanceRate,
      };
    }

    // Fine-grained signal: is tier-0 highly reliable for this exact shape?
    const fine = this.rate(`${tier0}|${task}|${len}`);
    if (fine.samples >= SKIP_JUDGE_MIN_SAMPLES && fine.acceptanceRate >= SKIP_JUDGE_ACCEPTANCE) {
      return {
        mode: "skip-judge",
        reason: `tier-0 acceptance ${(fine.acceptanceRate * 100).toFixed(0)}% over ${fine.samples} ${task}/${len} tasks`,
        samples: fine.samples,
        acceptanceRate: fine.acceptanceRate,
      };
    }

    return {
      mode: "judge",
      reason: `insufficient evidence (${fine.samples}/${SKIP_JUDGE_MIN_SAMPLES} samples for ${task}/${len})`,
      samples: fine.samples,
      acceptanceRate: fine.acceptanceRate,
    };
  }

  /** Human-readable policy table for reports. */
  reportTable(): string[] {
    if (this.stats.size === 0) return ["(policy: no learning data yet — run deepseek_genesis to calibrate)"];
    const rows: string[] = [];
    const sorted = [...this.stats.entries()].sort((a, b) => b[1].lastTs - a[1].lastTs);
    for (const [key, bucket] of sorted.slice(0, 12)) {
      const [tier, task, len] = key.split("|");
      const rate = bucket.samples > 0 ? `${Math.round((bucket.accepted / bucket.samples) * 100)}%` : "-";
      rows.push(`  ${tier} | ${task}${len ? `/${len}` : ""}: ${bucket.accepted}/${bucket.samples} accepted (${rate})`);
    }
    return rows;
  }

  /** Test hooks. */
  clearForTest(): void {
    this.stats.clear();
  }

  seedForTest(key: string, stats: BucketStats): void {
    this.stats.set(key, stats);
  }
}
