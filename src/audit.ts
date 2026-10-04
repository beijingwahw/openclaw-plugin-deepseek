/**
 * Decision audit trail — structured, bounded, persistent.
 *
 * Every consequential plugin decision (cascade routing, budget transitions,
 * escalations, compactions) lands here so operators can answer "why did the
 * plugin do that" after the fact. Entries are plain JSON, capped in memory,
 * and snapshotted next to the ledger.
 */
export type AuditKind =
  | "cascade-route"
  | "cascade-escalation"
  | "cascade-skip-judge"
  | "budget-transition"
  | "context-compaction"
  | "genesis-calibration"
  | "fim-request"
  | "extract-repair";

export interface AuditEntry {
  ts: number;
  kind: AuditKind;
  summary: string;
  data?: Record<string, unknown>;
}

const MAX_AUDIT_ENTRIES = 500;
const AUDIT_PATH_SUFFIX = "openclaw-deepseek/audit-trail.json";

export class AuditTrail {
  private entries: AuditEntry[] = [];
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
    if (!this.stateDir) return;
    const { readFile } = await import("node:fs/promises");
    try {
      const raw = await readFile(this.snapshotPath() as string, "utf8");
      const parsed = JSON.parse(raw) as { entries?: AuditEntry[] };
      if (Array.isArray(parsed.entries)) {
        this.entries = parsed.entries.slice(-MAX_AUDIT_ENTRIES);
      }
    } catch {
      // fresh start
    }
  }

  private snapshotPath(): string | null {
    return this.stateDir ? `${this.stateDir.replace(/\/+$/, "")}/${AUDIT_PATH_SUFFIX}` : null;
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
    await writeFile(tmp, JSON.stringify({ savedAt: this.now(), entries: this.entries }), "utf8");
    await rename(tmp, path);
  }

  async ready(): Promise<void> {
    await this.loading;
  }

  record(kind: AuditKind, summary: string, data?: Record<string, unknown>): AuditEntry {
    const entry: AuditEntry = { ts: this.now(), kind, summary: summary.slice(0, 300), data };
    this.entries.push(entry);
    if (this.entries.length > MAX_AUDIT_ENTRIES) {
      this.entries.splice(0, this.entries.length - MAX_AUDIT_ENTRIES);
    }
    this.schedulePersist();
    return entry;
  }

  /** Recent entries, newest first, optionally filtered by kind. */
  recent(options: { kind?: AuditKind; limit?: number } = {}): AuditEntry[] {
    const limit = Math.min(options.limit ?? 20, MAX_AUDIT_ENTRIES);
    const filtered = options.kind ? this.entries.filter((e) => e.kind === options.kind) : this.entries;
    return filtered.slice(-limit).reverse();
  }

  /** Renders a human-readable audit tail. */
  render(options: { kind?: AuditKind; limit?: number } = {}): string {
    const entries = this.recent(options);
    if (entries.length === 0) return "(audit trail: no matching entries)";
    const lines = ["Audit trail (newest first):"];
    for (const entry of entries) {
      const time = new Date(entry.ts).toISOString().slice(11, 19);
      lines.push(`  [${time}] ${entry.kind}: ${entry.summary}`);
    }
    return lines.join("\n");
  }

  clearForTest(): void {
    this.entries = [];
  }
}
