/**
 * Live cost ledger: subscribes to gateway-wide model output and aggregates
 * spend per model / session / day with a bounded event ring buffer.
 *
 * Every model call in the gateway (any provider) flows through record(), fed
 * by the plugin's llm_output hook. Costs come from the universal PricingTable;
 * unpriced models are tracked in tokens with cost = null so the report stays
 * honest instead of silently undercounting.
 */
import { PricingTable, type PricedUsage, type RawUsage } from "./pricing.js";

export interface LedgerEntry {
  ts: number;
  provider: string;
  model: string;
  ref: string;
  sessionKey?: string;
  usage: RawUsage;
  costUsd: number | null;
  priced: boolean;
  priceSource: PricedUsage["source"];
}
export interface ModelAggregate {
  ref: string;
  calls: number;
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  costUsd: number;
  unpricedCalls: number;
}

export interface LedgerSummary {
  since: number;
  calls: number;
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  costUsd: number;
  unpricedCalls: number;
  byModel: ModelAggregate[];
  byDay: Array<{ date: string; costUsd: number; calls: number }>;
  /** Today's spend in USD (UTC date). */
  todayCostUsd: number;
  /** Budget warnings when dailyBudgetUsd is configured. */
  budget?: { limitUsd: number; todayUsd: number; exceeded: boolean; percent: number };
  /** Burn-rate forecast (needs >= 2 recorded calls today). */
  forecast?: {
    spentUsd: number;
    spentHours: number;
    hourlyUsd: number;
    projectedEodUsd: number;
    redAtUtc?: string;
  };
}

const MAX_ENTRIES = 1000;
const PERSIST_PATH_SUFFIX = "openclaw-deepseek/ledger-snapshot.json";

export class CostLedger {
  private entries: LedgerEntry[] = [];
  private readonly pricing: PricingTable;
  private readonly now: () => number;
  private readonly stateDir: string | undefined;
  private persistTimer: ReturnType<typeof setTimeout> | undefined;
  private loading: Promise<void> | null = null;
  private dailyBudgetUsd: number | null = null;
  private budgetWarned = false;

  constructor(options: { pricing?: PricingTable; now?: () => number; stateDir?: string } = {}) {
    this.pricing = options.pricing ?? new PricingTable();
    this.now = options.now ?? Date.now;
    this.stateDir = options.stateDir;
    if (this.stateDir) {
      this.loading = this.loadSnapshot().catch(() => {
        // Corrupt or missing snapshot: start fresh rather than crash the plugin.
      });
    }
  }

  private snapshotPath(): string | null {
    if (!this.stateDir) return null;
    const path = `${this.stateDir.replace(/\/+$/, "")}/${PERSIST_PATH_SUFFIX}`;
    return path;
  }

  private async loadSnapshot(): Promise<void> {
    const path = this.snapshotPath();
    if (!path) return;
    const { readFile } = await import("node:fs/promises");
    try {
      const raw = await readFile(path, "utf8");
      const parsed = JSON.parse(raw) as { entries?: LedgerEntry[]; savedAt?: number };
      if (Array.isArray(parsed.entries)) {
        // Keep only the newest MAX_ENTRIES; drop anything absurdly stale (30d).
        const cutoff = this.now() - 30 * 24 * 3_600_000;
        this.entries = parsed.entries
          .filter((e) => e && typeof e.ts === "number" && e.ts >= cutoff)
          .slice(-MAX_ENTRIES);
      }
    } catch {
      // No snapshot yet: fine.
    }
  }

  /** Debounced, best-effort persistence. Never throws. */
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
    const payload = JSON.stringify({ savedAt: this.now(), entries: this.entries });
    await mkdir(dir, { recursive: true });
    await writeFile(tmp, payload, "utf8");
    await rename(tmp, path); // atomic swap
  }

  /** Resolves when any startup snapshot load has settled (test hook). */
  async ready(): Promise<void> {
    await this.loading;
  }

  get pricingTable(): PricingTable {
    return this.pricing;
  }

  /** Configures the daily USD budget (null disables warnings). */
  setDailyBudget(limitUsd: number | null): void {
    this.dailyBudgetUsd = limitUsd;
    this.budgetWarned = false;
  }

  /** Records one model call from the llm_output hook (any provider). */
  record(event: {
    provider: string;
    model: string;
    ref?: string;
    sessionKey?: string;
    usage?: RawUsage;
  }): LedgerEntry {
    const ref = event.ref ?? `${event.provider}/${event.model}`;
    const usage = event.usage ?? {};
    const priced = this.pricing.price(ref, usage);
    const entry: LedgerEntry = {
      ts: this.now(),
      provider: event.provider,
      model: event.model,
      ref,
      sessionKey: event.sessionKey,
      usage,
      costUsd: priced.costUsd,
      priced: priced.priced,
      priceSource: priced.source,
    };
    this.entries.push(entry);
    if (this.entries.length > MAX_ENTRIES) {
      this.entries.splice(0, this.entries.length - MAX_ENTRIES);
    }
    this.schedulePersist();
    return entry;
  }

  /** Aggregates the ledger, optionally scoped to a session or day window. */
  summary(options: { sessionKey?: string; sinceMs?: number } = {}): LedgerSummary {
    const since = options.sinceMs ?? 0;
    const models = new Map<string, ModelAggregate>();
    const days = new Map<string, { costUsd: number; calls: number }>();
    let calls = 0;
    let inputTokens = 0;
    let outputTokens = 0;
    let cacheReadTokens = 0;
    let costUsd = 0;
    let unpricedCalls = 0;
    let earliest = Number.POSITIVE_INFINITY;

    for (const entry of this.entries) {
      if (options.sessionKey && entry.sessionKey !== options.sessionKey) continue;
      if (entry.ts < since) continue;
      earliest = Math.min(earliest, entry.ts);
      calls += 1;
      inputTokens += entry.usage.input ?? 0;
      outputTokens += entry.usage.output ?? 0;
      cacheReadTokens += entry.usage.cacheRead ?? 0;
      if (entry.costUsd !== null) costUsd += entry.costUsd;
      if (!entry.priced) unpricedCalls += 1;

      const agg = models.get(entry.ref) ?? {
        ref: entry.ref,
        calls: 0,
        inputTokens: 0,
        outputTokens: 0,
        cacheReadTokens: 0,
        costUsd: 0,
        unpricedCalls: 0,
      };
      agg.calls += 1;
      agg.inputTokens += entry.usage.input ?? 0;
      agg.outputTokens += entry.usage.output ?? 0;
      agg.cacheReadTokens += entry.usage.cacheRead ?? 0;
      if (entry.costUsd !== null) agg.costUsd += entry.costUsd;
      if (!entry.priced) agg.unpricedCalls += 1;
      models.set(entry.ref, agg);

      const date = new Date(entry.ts).toISOString().slice(0, 10);
      const day = days.get(date) ?? { costUsd: 0, calls: 0 };
      if (entry.costUsd !== null) day.costUsd += entry.costUsd;
      day.calls += 1;
      days.set(date, day);
    }

    const today = new Date(this.now()).toISOString().slice(0, 10);
    const todayCostUsd = Math.round((days.get(today)?.costUsd ?? 0) * 1e6) / 1e6;
    const summary: LedgerSummary = {
      since: Number.isFinite(earliest) ? earliest : this.now(),
      calls,
      inputTokens,
      outputTokens,
      cacheReadTokens,
      costUsd: Math.round(costUsd * 1e6) / 1e6,
      unpricedCalls,
      byModel: [...models.values()].sort((a, b) => b.costUsd - a.costUsd),
      byDay: [...days.entries()]
        .map(([date, d]) => ({ date, costUsd: Math.round(d.costUsd * 1e6) / 1e6, calls: d.calls }))
        .sort((a, b) => a.date.localeCompare(b.date)),
      todayCostUsd,
    };
    if (this.dailyBudgetUsd !== null && this.dailyBudgetUsd > 0) {
      summary.budget = {
        limitUsd: this.dailyBudgetUsd,
        todayUsd: todayCostUsd,
        exceeded: todayCostUsd > this.dailyBudgetUsd,
        percent: Math.round((todayCostUsd / this.dailyBudgetUsd) * 100),
      };
      const forecast = this.forecast(today, todayCostUsd);
      if (forecast) summary.forecast = forecast;
    }
    return summary;
  }

  /**
   * Burn-rate forecast: spend velocity from today's first recorded call,
   * linear projection to end-of-day, and the UTC time the red threshold
   * (90% of budget) would be crossed at the current pace.
   */
  private forecast(today: string, todayCostUsd: number): LedgerSummary["forecast"] {
    const todaysEntries = this.entries.filter((e) => {
      const date = new Date(e.ts).toISOString().slice(0, 10);
      return date === today && e.costUsd !== null;
    });
    if (todaysEntries.length < 2 || !this.dailyBudgetUsd) return undefined;
    const first = Math.min(...todaysEntries.map((e) => e.ts));
    const now = this.now();
    const spentHours = Math.max(0.05, (now - first) / 3_600_000);
    const hourlyUsd = todayCostUsd / spentHours;
    const nowUtc = new Date(now);
    const hoursLeft = 24 - nowUtc.getUTCHours() - nowUtc.getUTCMinutes() / 60;
    const projectedEodUsd = todayCostUsd + hourlyUsd * hoursLeft;
    const forecast: NonNullable<LedgerSummary["forecast"]> = {
      spentUsd: todayCostUsd,
      spentHours: Math.round(spentHours * 100) / 100,
      hourlyUsd: Math.round(hourlyUsd * 1e4) / 1e4,
      projectedEodUsd: Math.round(projectedEodUsd * 1e6) / 1e6,
    };
    const redThreshold = this.dailyBudgetUsd * 0.9;
    if (hourlyUsd > 0 && todayCostUsd < redThreshold) {
      const hoursToRed = (redThreshold - todayCostUsd) / hourlyUsd;
      const redAt = new Date(now + hoursToRed * 3_600_000);
      forecast.redAtUtc = redAt.toISOString().slice(11, 16);
    }
    return forecast;
  }

  /** Renders a human-readable live dashboard. */
  render(options: { sessionKey?: string; sinceMs?: number } = {}): string {
    const summary = this.summary(options);
    const lines: string[] = [];
    const scope = options.sessionKey ? ` (session ${options.sessionKey})` : "";
    const window = options.sinceMs
      ? ` since ${new Date(Date.now() - options.sinceMs).toISOString().slice(0, 16).replace("T", " ")} UTC`
      : ` since ${new Date(summary.since).toISOString().slice(0, 16).replace("T", " ")} UTC`;
    lines.push(`Live model spend${scope}${window}:`);
    lines.push(
      `  calls: ${summary.calls} | input: ${fmtTokens(summary.inputTokens)} | output: ${fmtTokens(summary.outputTokens)} | cache-hit: ${fmtTokens(summary.cacheReadTokens)}`,
    );
    if (summary.unpricedCalls > 0) {
      lines.push(`  cost: $${summary.costUsd.toFixed(4)} (+${summary.unpricedCalls} unpriced calls tracked in tokens only)`);
    } else {
      lines.push(`  cost: $${summary.costUsd.toFixed(4)}`);
    }
    if (summary.byModel.length > 0) {
      lines.push("", "By model:");
      for (const m of summary.byModel) {
        const unpriced = m.unpricedCalls > 0 ? ` (+${m.unpricedCalls} unpriced)` : "";
        lines.push(
          `  ${m.ref}: ${m.calls} calls, ${fmtTokens(m.inputTokens)} in / ${fmtTokens(m.outputTokens)} out, $${m.costUsd.toFixed(4)}${unpriced}`,
        );
      }
    }
    if (summary.byDay.length > 1) {
      lines.push("", "By day:");
      for (const d of summary.byDay) {
        lines.push(`  ${d.date}: $${d.costUsd.toFixed(4)} (${d.calls} calls)`);
      }
    }
    if (summary.budget) {
      const b = summary.budget;
      const state = b.exceeded ? "EXCEEDED" : "on track";
      lines.push("", `Budget: $${b.todayUsd.toFixed(4)} / $${b.limitUsd.toFixed(2)} today (${b.percent}%) - ${state}`);
      if (summary.forecast) {
        const f = summary.forecast;
        lines.push(
          `Burn rate: $${f.hourlyUsd.toFixed(4)}/h over ${f.spentHours}h -> projected EOD $${f.projectedEodUsd.toFixed(2)}${f.redAtUtc ? ` | red threshold ~${f.redAtUtc} UTC` : ""}`,
        );
      }
    }
    const catalog = this.pricing.catalogReady
      ? `catalog: ${this.pricing.catalogSize} models priced`
      : "catalog: not loaded (DeepSeek priced; others tracked in tokens until loaded)";
    lines.push("", `pricing: ${catalog}`);
    return lines.join("\n");
  }

  /** Test hook. */
  clearForTest(): void {
    this.entries = [];
  }
}

function fmtTokens(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}
