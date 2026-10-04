/**
 * Budget guardian — the plugin's economic self-governance.
 *
 * Watches the daily USD budget on the live ledger and exposes an economic
 * state machine that cascades consult before routing:
 *
 *   green   (< 50% used, or no budget configured): normal operation
 *   amber   (>= 50%): economize — prefer judge-free fast paths, keep tiers
 *   red     (>= 90%): emergency — single cheapest tier, no judge calls
 *
 * State transitions are surfaced once per change so channels see them.
 */
import type { CostLedger } from "./ledger.js";

export type GuardianState = "green" | "amber" | "red" | "unconfigured";

export interface GuardianStatus {
  state: GuardianState;
  usedUsd: number;
  limitUsd: number | null;
  percent: number | null;
}

export type EconomicMode = "normal" | "economize" | "emergency";

export interface RouteAdvice {
  mode: EconomicMode;
  /** Possibly truncated tier list (emergency keeps only the cheapest). */
  tiers: string[];
  /** Skip the judge call entirely for this run. */
  skipJudge: boolean;
  reason: string;
}

export interface GuardianDeps {
  ledger: CostLedger;
  onTransition?: (from: GuardianState, to: GuardianState, status: GuardianStatus) => void;
}

const AMBER_THRESHOLD = 0.5;
const RED_THRESHOLD = 0.9;

export class BudgetGuardian {
  private lastState: GuardianState | null = null;
  private readonly deps: GuardianDeps;

  constructor(deps: GuardianDeps) {
    this.deps = deps;
  }

  /** Current economic state derived from the ledger's daily budget. */
  status(): GuardianStatus {
    const budget = this.deps.ledger.summary().budget;
    if (!budget) {
      return { state: "unconfigured", usedUsd: 0, limitUsd: null, percent: null };
    }
    const percent = budget.percent / 100;
    const state: GuardianState = percent >= RED_THRESHOLD ? "red" : percent >= AMBER_THRESHOLD ? "amber" : "green";
    const status: GuardianStatus = {
      state,
      usedUsd: budget.todayUsd,
      limitUsd: budget.limitUsd,
      percent: budget.percent,
    };
    this.maybeAnnounce(state, status);
    return status;
  }

  private maybeAnnounce(state: GuardianState, status: GuardianStatus): void {
    if (this.lastState !== state) {
      const previous = this.lastState;
      this.lastState = state;
      if (previous !== null) {
        this.deps.onTransition?.(previous, state, status);
      }
    }
  }

  /** Routing advice for an upcoming cascade. */
  routeAdvice(tiers: string[]): RouteAdvice {
    const status = this.status();
    if (status.state === "unconfigured" || status.state === "green") {
      return { mode: "normal", tiers, skipJudge: false, reason: `budget ${status.state}` };
    }
    if (status.state === "amber") {
      return {
        mode: "economize",
        tiers,
        skipJudge: false, // policy may still choose skip-judge; guardian just allows leaning cheaper
        reason: `budget ${status.percent}% used (${status.usedUsd.toFixed(4)}/${status.limitUsd?.toFixed(2)} USD) — economizing`,
      };
    }
    // red: single cheapest tier, no judge calls at all.
    return {
      mode: "emergency",
      tiers: tiers.length > 0 ? [tiers[0]] : tiers,
      skipJudge: true,
      reason: `budget ${status.percent}% used (${status.usedUsd.toFixed(4)}/${status.limitUsd?.toFixed(2)} USD) — emergency single-tier routing`,
    };
  }

  /** Test hook: resets the transition memory. */
  resetForTest(): void {
    this.lastState = null;
  }
}
