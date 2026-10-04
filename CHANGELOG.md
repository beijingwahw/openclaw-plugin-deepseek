# Changelog

## 1.7.0 (2026-10-05) — Frontier quality pass

Line-by-line hardening with 2026-era standards:

- **Runtime boundary validators** (`src/validate.ts`): SSE usage frames and
  tool-call fragments from the wire are parsed and verified, never
  `as`-cast. Malformed fields drop individually (no NaN/Infinity leaks);
  corrupt frames degrade instead of crashing the stream parser.
- **Every timer is `.unref()`-aware**: request timeouts, retry backoffs,
  persistence debounces, and cooldown sleeps can no longer hold the gateway
  process open. Verified by an active-handle growth test.
- **Zero silent catches**: all 19 bare `catch {}` blocks now carry explicit
  reason comments (tested by an awk audit — zero remain).
- **Branded `ResolvedModelRef`**: model ids that flowed through alias
  resolution carry a compile-time brand; unvalidated strings can't masquerade
  as resolved refs.

## 1.6.0 (2026-10-05) — Century evolution III: systems engines

Two classic distributed-systems engines, applied to every API-bound module:

- **AIMD concurrency governor** (`src/concurrency.ts`): TCP-style adaptive
  windows per model — linear growth on success, halving on 429, circuit-break
  cooldown after three consecutive rate limits. All DeepSeekClient traffic
  (harness, cascades, tools) flows through acquire/release; DeepSeek's
  account-level limits get a client-side mirror.
- **Deterministic response cache** (`src/cache.ts`): LRU + TTL read-through
  cache keyed by (kind, model, prompt). deepseek_research and deepseek_reason
  reuse fresh answers for identical inputs — repeat questions cost nothing.
- Genesis health check now reports concurrency windows and cache hit rates;
  the live cost dashboard shows cache stats.

## 1.5.0 (2026-10-05) — Genesis quality pass: line-by-line

Full-codebase audit and refactor. Behavior-preserving (87/87 tests unchanged
in intent), structure-transforming:

- **cascade.ts rewritten as a thin facade** over the universal engine —
  eliminated ~150 lines of duplicated draft/judge/escalate logic; the
  DeepSeek-native cascade and universal cascade now share one engine.
- **shared.ts single source of truth**: judge parser (was 3 copies), judge
  prompt (2), usage merge (3), clamp (2), preview/truncate, stable stringify.
- **Single DeepSeek price table**: pricing.ts now reads the catalog from
  cost.ts instead of maintaining a parallel copy (this audit also caught and
  fixed a regression where cascade virtual-model pricing was lost during
  dedup — restored with a test).
- **Zero `any`**: memory.ts transcript blocks gained a precise
  `TranscriptBlock` type with type guards; anthropic.ts SSE frames are fully
  typed; provider catalog contexts are named interfaces; index.ts hook
  surface is a named structural cast.
- Dead code removed: unused settle-marker branch, duplicate cleanup shim,
  local helper copies across five modules.

## 1.4.0 (2026-10-05) — Deep evolution

Three deep capabilities, each grounded in the official DeepSeek API surface:

- **Anthropic-Messages transport** (`src/anthropic.ts` + second provider
  `openclaw-deepseek-anthropic`): speaks the Claude protocol against
  https://api.deepseek.com/anthropic with server-side claude-* model mapping
  (opus -> v4-pro, sonnet/haiku/unknown -> flash). x-api-key auth,
  anthropic-version headers, content-block streaming (text + thinking),
  merged usage frames with cache-read accounting. Claude-ecosystem refs now
  route to DeepSeek through a first-class provider.
- **Harness multi-turn memory** (`src/memory.ts`): the harness was stateless
  across turns; it now rebuilds DeepSeek-visible history from the OpenClaw
  session transcript (trailing 12 exchanges), honoring the
  reasoning_content replay contract on replayed tool-call turns.
- **Burn-rate forecasting** (ledger): spend velocity from today's first call,
  linear end-of-day projection, and the UTC time the red budget threshold
  will be crossed. Surfaced in the live dashboard.

## 1.3.0 (2026-10-05) — Century upgrade: autonomy

The plugin now governs its own economics and explains its own decisions.

- **BudgetGuardian** (`src/guardian.ts`): green/amber/red economic state
  machine over the daily budget. Amber economizes; red routes single-tier
  with no judge calls. Transitions announce once and land in the audit trail.
- **AuditTrail** (`src/audit.ts`): structured decision log (routes,
  escalations, budget transitions, skip-judge fast paths) — bounded at 500
  entries, persisted across restarts, filterable, human-renderable.
- **Economic cascade routing**: the cascade consults the guardian before the
  learning policy; emergency mode overrides everything.
- **KV-cache affinity**: the harness sends a stable per-session DeepSeek
  `user_id` (sanitized), improving cache-hit rates (hits bill at ~1/50).
- **Genesis = full health check**: calibration + balance + economic state +
  pricing catalog + recent decisions in one report. The cost dashboard's
  live view now appends economic state and the audit tail.

## 1.2.0 (2026-10-05) — Genesis: the adaptive nervous system

The cascade engine learned to learn.

- **AdaptiveCascadePolicy** (`src/policy.ts`): per task-shape (code/reasoning/
  factual/creative/general x short/medium/long) acceptance learning with
  three routing modes — skip-judge fast path (>=85% acceptance over >=6
  samples), direct-strong (skip the weak tier when acceptance <30% over >=10
  samples), and normal judging otherwise. Stats decay after 14 days and
  persist across restarts.
- **Anti-drift auditing**: every 4th skip-judge run still pays for a judge
  call (explore/exploit guard).
- **deepseek_genesis tool**: self-calibration — runs probe tasks per shape,
  seeds the policy, and reports the learned routing table.
- The cascade tool now consults the policy automatically; routeMode events
  surface the chosen path.

## 1.1.0 (2026-10-05) — All-module evolution

Every module gained a substantive upgrade, including two production bug fixes
found by the new regression tests:

- **deepseek-api**: fixed a timeout-timer leak (cleanup callbacks never ran);
  all requests now manage timer lifecycle explicitly. Added DeepSeekAbortError.
- **loop**: context-overflow self-healing — incremental size tracking (O(1)
  budget checks instead of quadratic re-serialization) with oldest-tool-result
  compaction; fixed an infinite-compaction loop found by stress testing;
  circuit breaker for identical repeated tool calls.
- **cascade + universal-cascade**: judgeThreshold config (escalate on low
  confidence, not just explicit escalate grades).
- **ledger**: daily USD budget with exceeded/percent reporting.
- **extract**: real JSON-Schema subset validation (required + primitive types)
  with targeted repair prompts naming the exact failed fields.
- **harness**: empty-result classification ("empty" / "reasoning-only") so
  OpenClaw's fallback policy can retry degenerate outcomes on other models.
- **fim**: configurable model.
- **types**: new config fields fimModel / dailyBudgetUsd / judgeThreshold with
  manifest schema entries.

## 1.0.0 (2026-10-05)

First stable release. Full project upgrade from the 0.x feature series.

### Runtime
- **True streaming deltas**: the harness now forwards incremental text via
  `onPartialReply({ delta })` instead of re-sending accumulated text.
- **Persistent cost ledger**: live spend history survives gateway restarts
  (atomic JSON snapshots under the gateway state dir, 2s debounce, 30-day
  retention, bounded ring buffer).
- **Universal speculative cascade**: draft -> judge -> escalate across ANY
  gateway provider (native DeepSeek executor + host llm executor).
- **Live cross-provider cost dashboard**: `llm_output` hook records every
  model call; pricing = exact DeepSeek table (peak/off-peak, cache-aware)
  + LiteLLM community catalog (TTL-cached) + honest unpriced fallback.
- **Credits/balance view**: DeepSeek account balance + provider billing
  snapshots.
- DeepSeek-native: thinking control (`reasoning_effort` none/low/high/max),
  `reasoning_content` replay contract, vision input (flash), retired-model
  alias migration, 429/`insufficient_system_resource` retries,
  parallel tool execution with sequential isolation.

### Tools (7)
`deepseek_research`, `deepseek_reason`, `deepseek_balance`,
`deepseek_cascade`, `deepseek_extract` (JSON mode + repair),
`deepseek_fim` (beta endpoint), `deepseek_cost_report` (live/pricing/estimate/credits).

### Engineering
- TypeScript strict + `noImplicitOverride` + `noFallthroughCasesInSwitch`.
- 58 tests across 9 suites (SSE, loop, thinking, cascades, pricing, ledger
  persistence, extract, FIM, e2e hook/delta coverage).
- GitHub Actions CI: build, test, validate, npm pack smoke.
- Structural validator without a gateway (`npm run plugin:validate`).

## 0.5.0 (2026-10-05)

Live cross-provider cost dashboard, universal pricing, credits view.

## 0.4.0 (2026-10-05)

Universal speculative cascade (any provider as tiers/judge).

## 0.3.0 (2026-10-05)

Speculative cascade, structured extraction, FIM completion, cost metering,
parallel tool execution.

## 0.2.0 (2026-10-05)

Renamed to `openclaw-deepseek` (avoid collision with DeepSeek's official
"harness" project). Current model catalog (V4.1-Flash + V4-Pro), retired
alias migration, thinking-mode defenses, vision, retries.

## 0.1.0 (2026-10-05)

Initial release: native agent harness, companion provider, research/reason/
balance tools.
