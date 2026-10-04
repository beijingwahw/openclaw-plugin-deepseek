# OpenClaw DeepSeek Plugin

[![tests](https://img.shields.io/badge/tests-102%2F102-brightgreen)]()
[![version](https://img.shields.io/badge/version-1.0.0-blue)]()
[![typecheck](https://img.shields.io/badge/tsc-strict-blue)]()

A world-class DeepSeek integration for [OpenClaw](https://docs.openclaw.ai): a **native agent harness** running DeepSeek's own function-calling loop with thinking control, a **universal speculative cascade** that works across **all gateway providers** (DeepSeek → GLM → Claude → GPT → Gemini …), **structured extraction**, **FIM code completion**, **real-time cost metering** with peak/off-peak awareness, a **live cross-provider cost dashboard** fed by every model call in the gateway, and **credits/balance tracking**, and seven agent tools.

Built against the official DeepSeek API surface as of **2026-10** (V4.1-Flash + V4-Pro, thinking mode, vision, context caching, peak/off-peak pricing).

> Named `openclaw-deepseek` — deliberately **not** "DeepSeek Harness", which is DeepSeek's own flagship open-source agent project ([github.com/deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness)).

## What it adds over the built-in `deepseek` provider

OpenClaw ships a solid transport-level DeepSeek provider. This plugin goes deeper:

| Capability | Built-in provider | This plugin |
| --- | --- | --- |
| Native agent loop (DeepSeek-side tool orchestration) | — | ✅ harness with step budgets |
| `reasoning_content` full-chain replay on tool turns | ✅ (transport) | ✅ (native loop, verified by test) |
| Thinking control (`thinking` + `reasoning_effort: none/low/high/max`) | partial | ✅ incl. `/think` level mapping |
| `tool_choice` required/named → auto downgrade under thinking | — | ✅ (avoids DeepSeek 400) |
| Sampling-param stripping while thinking (temp/top_p/penalties) | — | ✅ |
| Retired model aliases (`deepseek-chat`, `deepseek-reasoner`, `deepseek-v4-flash`) | — | ✅ normalize + migrate warning |
| Vision input (V4.1-Flash) | — | ✅ images bridged as `image_url`; pro fails readable |
| 429 / `insufficient_system_resource` retry with backoff | — | ✅ |
| `: keep-alive` SSE comment tolerance + `include_usage` frames | — | ✅ |
| Speculative cascade (flash drafts, judge grades, pro escalates) | — | ✅ `deepseek-cascade` model + tool |
| Structured extraction (JSON mode + repair pass) | — | ✅ `deepseek_extract` |
| FIM code completion (beta endpoint) | — | ✅ `deepseek_fim` |
| Live cost dashboard: every provider's calls, per model/session/day | — | ✅ `llm_output` hook + ledger |
| Universal pricing (LiteLLM catalog, ~all vendors) | — | ✅ TTL-cached, offline-safe |
| Credits/balance panel (DeepSeek balance + provider billing) | — | ✅ `view: credits` |
| Autonomous budget governance (green/amber/red routing modes) | — | ✅ guardian |
| Decision audit trail (persisted, filterable) | — | ✅ |
| Per-session KV-cache affinity (`user_id`) | — | ✅ |
| Claude-protocol transport (`/anthropic` endpoint) | — | ✅ |
| Harness multi-turn memory (transcript rebuild) | — | ✅ |
| Burn-rate forecast (EOD projection + red-crossing time) | — | ✅ |
| AIMD concurrency control (per-model adaptive windows) | — | ✅ |
| Deterministic response cache (repeat calls free) | — | ✅ |
| Peak/off-peak + cache-aware cost metering | — | ✅ `deepseek_cost_report` |
| Parallel tool execution (bounded, sequential-safe) | — | ✅ |
| `deepseek_research` / `deepseek_reason` / `deepseek_balance` tools | — | ✅ |

The harness only claims `openclaw-deepseek/*` model refs — your `deepseek/*` refs keep using the built-in provider. Opt in per session or per agent.

## Install

> The live dashboard records every provider's model calls in memory (bounded ring buffer). Costs for DeepSeek are exact (peak/off-peak/cache-aware); other providers use the LiteLLM community catalog (fetched once per day, offline-safe); unknown models are tracked in tokens and reported as unpriced instead of silently undercounting.

> Cross-provider cascade tiers require `plugins.entries.openclaw-deepseek.llm.allowModelOverride: true` in gateway config (operator opt-in for model overrides, per OpenClaw policy). Pure-DeepSeek cascades need nothing extra.

```bash
openclaw plugins install clawhub:autoclaw/openclaw-plugin-deepseek
# or from a local checkout:
openclaw plugins install ./openclaw-plugin-deepseek
```

Requires OpenClaw >= 2026.9.1, Node >= 22.22.3.

## Configure

One key powers everything (reuses the built-in `deepseek` auth via `providerAuthAliases`):

```bash
export DEEPSEEK_API_KEY=sk-...   # gateway process environment, ~/.openclaw/.env, or auth store
```

Optional plugin config:

```json5
{
  plugins: {
    entries: {
      "openclaw-deepseek": {
        config: {
          baseUrl: "https://api.deepseek.com",   // bare domain per current docs; /v1 also works
          model: "deepseek-flash",               // V4.1-Flash (vision); or "deepseek-v4-pro"
          defaultReasoningEffort: "high",        // none | low | high | max
          researchMaxSteps: 8,
          researchMaxTokensPerStep: 16384,
          requestTimeoutMs: 180000
        }
      }
    }
  }
}
```

## Model catalog (2026-10, official)

| Model ref | Context | Max output | Vision | Off-peak $/M (in / cache-hit / out) |
| --- | --- | --- | --- | --- |
| `openclaw-deepseek/deepseek-flash` | 1M | 384K | ✅ | 0.15 / 0.003 / 0.6 |
| `openclaw-deepseek/deepseek-v4-pro` | 1M | 384K | ❌ | 0.66 / 0.022 / 1.98 |
| `openclaw-deepseek/deepseek-cascade` | 1M | 384K | ❌ | flash-tier pricing; pro only on escalation |
| `openclaw-deepseek-anthropic/claude-*` | 1M | 384K | ❌ | Claude protocol; opus->v4-pro, sonnet/haiku->flash |

Retired names (`deepseek-chat`, `deepseek-reasoner`, `deepseek-v4-flash`, `deepseek-v4-flash-vision-exp`) still resolve — they route to `deepseek-flash` with a one-time migration warning in the logs.

## Use

**Route turns through the native harness:**

```
/model openclaw-deepseek/deepseek-flash
/model openclaw-deepseek/deepseek-v4-pro
```

Any DeepSeek model id works — unknown ids resolve dynamically (`openclaw-deepseek/<anything>`).

Thinking control follows OpenClaw's `/think` directive: `/think off` maps to `thinking: {type: "disabled"}`, `/think medium` → `reasoning_effort: "high"`, `/think max` → `reasoning_effort: "max"`.

**Universal speculative cascade** — the signature feature:

```
/model openclaw-deepseek/deepseek-cascade      # pure DeepSeek: flash -> judge -> pro
```

```
# or via the tool, with ANY gateway model refs as tiers:
Use deepseek_cascade with:
  task: "derive the closed-form solution"
  tiers: ["deepseek-flash", "zai/glm-5.3", "anthropic/claude-sonnet-4-6"]
  judge: "zai/glm-5.3"
```

Tier-1 drafts cheaply → the judge grades the draft → low-confidence answers escalate through the remaining tiers at max reasoning. **Any configured gateway provider works as a tier or judge** — mix DeepSeek, GLM, Claude, GPT, Gemini freely. DeepSeek tiers run through the native executor (thinking control + cache-aware cost metering); everything else routes through the host LLM runtime with automatic usage/cost merging across providers. Most requests finish at tier-1 prices; hard ones get flagship quality.

**Tools (available to every agent once the plugin is enabled):**

| Tool | What it does |
| --- | --- |
| `deepseek_research` | DeepSeek-driven research loop: plans queries, calls the host `web_search` tool, iterates, and synthesizes a sourced answer. Budget-controlled (`max_searches`). |
| `deepseek_reason` | Cross-model second opinion: give it a question + draft answer, get a verdict, specific flaws, and an improved answer. |
| `deepseek_balance` | Checks the DeepSeek account balance endpoint. |
| `deepseek_genesis` | Self-calibrate the adaptive cascade policy: probe tasks seed per-shape acceptance rates; reports the learned routing table. |
| `deepseek_cascade` | Universal speculative cascade: 1-4 tiers of **any** gateway model refs, judge verdict, per-call trace, cross-provider cost report. |
| `deepseek_extract` | Structured JSON extraction with schema validation and one automatic repair retry (safe JSON-mode prompting included). |
| `deepseek_fim` | Fill-in-the-middle code completion (prefix + suffix → middle) via the beta FIM endpoint. |
| `deepseek_cost_report` | **Live cost dashboard** — `view:live` (default): gateway-wide spend per model/session/day from the `llm_output` hook ledger; `view:pricing`: unit prices (DeepSeek peak/off-peak + universal catalog); `view:estimate`: price a hypothetical request on any model; `view:credits`: DeepSeek balance + provider billing/credits. |

Example prompt: *"Use deepseek_research to compare DeepSeek V4.1-Flash vs V4-Pro pricing, then have deepseek_reason review your conclusion."*

## Architecture

```
openclaw.plugin.json     manifest: provider + harness + 7 tool contracts, auth aliases
index.ts                 definePluginEntry: wires provider, harness, tools
src/types.ts             official model catalog, alias tables, effort mapping, config
src/provider.ts          companion provider `openclaw-deepseek` (catalog + dynamic models)
src/harness.ts           AgentHarness: native loops + cascade routing + vision + /think
src/loop.ts              function-calling agent loop (thinking, parallel tools, budgets)
src/deepseek-api.ts      streaming Chat Completions client: SSE parser + retries
src/cascade.ts           DeepSeek-native cascade (legacy engine used by the virtual model)
src/universal-cascade.ts  universal cascade engine: host-llm + native executors, any provider
src/policy.ts             adaptive routing policy: per-shape acceptance learning (branch-predictor style)
src/anthropic.ts          Anthropic-Messages transport (claude-* refs -> DeepSeek)
src/memory.ts             harness multi-turn memory: transcript -> DeepSeek history
src/guardian.ts           budget guardian: green/amber/red economic state machine
src/audit.ts              decision audit trail: bounded, persisted, human-readable
src/cost.ts              DeepSeek peak/off-peak + cache-aware cost metering
src/pricing.ts            universal pricing: DeepSeek exact + LiteLLM catalog + unpriced fallback
src/ledger.ts             live cost ledger: llm_output hook -> per model/session/day aggregates
src/extract.ts           JSON-mode structured extraction with repair pass
src/fim.ts               fill-in-the-middle code completion (beta endpoint)
src/tools.ts             the 7 agent tools
test/                    58 tests: SSE, loop, thinking, cascades, pricing, ledger (incl. persistence), extract, FIM, e2e
```

Design notes:

- **Fail-closed auth.** No key → tools and harness return explicit errors instead of hanging.
- **`reasoning_content` replay contract.** Assistant tool-call turns keep `reasoning_content` on follow-up requests — the #1 community pitfall when hosting DeepSeek in agent loops (DeepSeek returns 400 otherwise).
- **Thinking-aware request shaping.** `tool_choice` is always `auto` while thinking; `temperature` only sent when thinking is off; effort words (ultrathink, xhigh, …) normalize to DeepSeek's `none/low/high/max`.
- **Usage via `stream_options.include_usage`.** The SSE parser captures the final usage frame (incl. `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`); per-step usage merges into the attempt result.
- **Retry policy.** 429 and 5xx/`insufficient_system_resource` retry with 1s/3s/9s backoff on non-streaming calls; streams fail fast (cannot replay mid-body).
- **Timeout + abort everywhere.** Every request wraps an `AbortController`; the outer attempt signal propagates through tool executions.
- **No core patches.** Pure plugin SDK surfaces: `registerProvider`, `registerAgentHarness`, `registerTool`.

## v1.0 highlights

- **True streaming**: harness forwards incremental `delta` chunks (channels render progressively, no full-text re-sends)
- **Persistent ledger**: spend history survives gateway restarts (atomic snapshots, 30-day retention, bounded ring buffer)
- **Publishable**: `npm pack` clean — 19 files, 35 kB; CI runs build/test/validate/pack on every push

## Develop

```bash
npm install
npm run build            # tsc strict
npm test                 # vitest: 102 tests
npm run plugin:validate  # structural validation without a gateway
```

Publish:

```bash
clawhub package publish autoclaw/openclaw-plugin-deepseek --dry-run
clawhub package publish autoclaw/openclaw-plugin-deepseek
```

## License

MIT
