/**
 * deepseek_research / deepseek_reason / deepseek_balance agent tools.
 *
 * deepseek_research: DeepSeek function-calling driven research loop that
 *   combines the host web_search tool with DeepSeek synthesis.
 * deepseek_reason: cheap second-opinion reviewer backed by deepseek-chat.
 * deepseek_balance: DeepSeek account balance monitoring.
 */
import { Type } from "typebox";
import { DeepSeekClient } from "./deepseek-api.js";
import { runCascade } from "./cascade.js";
import {
  createHostLlmExecutor,
  createDeepSeekNativeExecutor,
  isNativeDeepSeekRef,
  runUniversalCascade,
  type UniversalExecutor,
} from "./universal-cascade.js";
import { AdaptiveCascadePolicy, classifyTask, lengthBucketOf } from "./policy.js";
import { clampNumber, parseJudgeVerdict as parseVerdictShared } from "./shared.js";
import { ResponseCache } from "./cache.js";
import { computeCost, isPeakNow, unitPriceTable, PEAK_WINDOW_DESCRIPTION } from "./cost.js";
import { runFimCompletion } from "./fim.js";
import { runJsonExtraction } from "./extract.js";
import { runAgentLoop, runSingleCompletion } from "./loop.js";
import {
  resolveConfig,
  type DeepSeekHarnessConfig,
  type DeepSeekReasoningEffort,
  type DeepSeekToolSchema,
} from "./types.js";

export interface ToolDeps {
  resolveApiKey: () => Promise<string | undefined>;
  getPluginConfig: () => DeepSeekHarnessConfig | undefined;
  /** Host web search hook (api.runtime.webSearch.search). */
  webSearch?: (args: { query: string; count?: number }) => Promise<{ content: Array<{ type: string; text?: string }> }>;
  /** Host LLM hook (api.runtime.llm.complete) for cross-provider cascade tiers. */
  hostLlm?: {
    complete(params: {
      messages: Array<{ role: string; content: string }>;
      model?: string;
      systemPrompt?: string;
      maxTokens?: number;
      temperature?: number;
      reasoning?: string;
      signal?: AbortSignal;
      purpose?: string;
    }): Promise<{ text: string; provider: string; model: string; usage?: { inputTokens?: number; outputTokens?: number; totalTokens?: number; costUsd?: number } }>;
  };
  /** Live cost ledger (gateway-wide, fed by the llm_output hook). */
  getLedger: () => import("./ledger.js").CostLedger;
  /** Adaptive cascade policy (learning). */
  getPolicy: () => AdaptiveCascadePolicy;
  /** Budget guardian (economic routing). */
  getGuardian: () => import("./guardian.js").BudgetGuardian;
  /** Decision audit trail. */
  getAudit: () => import("./audit.js").AuditTrail;
  /** AIMD concurrency governor. */
  getGovernor: () => import("./concurrency.js").ConcurrencyGovernor;
  /** Deterministic response cache for expensive tool calls. */
  getCache: () => ResponseCache<{ text: string; details: unknown }>;
  /** Optional provider usage/credits summary hook for the credits view. */
  getUsageSummary?: () => { credits: Array<{ label: string; display: string; raw?: unknown }> } | undefined;
  logger?: { warn: (msg: string, data?: unknown) => void };
}

const RESEARCH_SYSTEM_PROMPT = `You are a focused research agent powered by DeepSeek.
Goal: answer the user's research question with well-sourced, concrete findings.

Rules:
- Start by planning 2-4 search queries (different angles, languages matching the question).
- Call web_search for each query; read results carefully.
- If results are thin, refine queries and search again (you have a limited budget).
- Synthesize a structured answer: key findings, comparisons if relevant, caveats, and source URLs.
- Always answer in the same language as the question.
- Be honest about uncertainty and conflicting evidence.`;

const REASON_SYSTEM_PROMPT = `You are a rigorous second-opinion reviewer powered by DeepSeek.
The user will show you a question and a draft answer. Provide:
1. Verdict: agree / partially-agree / disagree (one line).
2. Flaws: concrete mistakes, missing considerations, or weak reasoning (bullet list, be specific).
3. Improved answer: your own concise answer if the draft needs correction.
Answer in the same language as the question. Be direct and technical; no pleasantries.`;

function makeClient(deps: ToolDeps, apiKey: string): DeepSeekClient {
  const config = resolveConfig(deps.getPluginConfig());
  return new DeepSeekClient({
    baseUrl: config.baseUrl,
    apiKey,
    timeoutMs: config.requestTimeoutMs,
    governor: deps.getGovernor(),
  });
}

function textOfContent(content: Array<{ type: string; text?: string }> | undefined): string {
  if (!content) return "";
  return content
    .map((part) => (part.type === "text" && typeof part.text === "string" ? part.text : ""))
    .filter(Boolean)
    .join("\n");
}

export function createDeepSeekTools(deps: ToolDeps) {
  const researchSchema = Type.Object({
    question: Type.String({ description: "The research question to investigate." }),
    max_searches: Type.Optional(
      Type.Number({ description: "Maximum web searches the loop may run. Default 6." }),
    ),
  });
  const researchParams = Type.Object({
    question: Type.String({ description: "The research question to investigate." }),
    max_searches: Type.Optional(
      Type.Number({ description: "Maximum web searches the loop may run. Default 6." }),
    ),
  });

  const reasonSchema = Type.Object({
    question: Type.String({ description: "The original question being answered." }),
    draft_answer: Type.String({ description: "The draft answer to review." }),
    model: Type.Optional(
      Type.String({ description: "DeepSeek model for the review. Default deepseek-chat." }),
    ),
  });

  const balanceSchema = Type.Object({});

  return {
    research: {
      name: "deepseek_research",
      label: "DeepSeek Research",
      description:
        "Run a DeepSeek-driven research loop: plans searches, calls the host web_search tool, and synthesizes a sourced answer. Best for open questions needing current web evidence.",
      parameters: researchParams,
      async execute(_id: string, params: unknown) {
        const args = params as { question: string; max_searches?: number };
        const apiKey = await deps.resolveApiKey();
        if (!apiKey) {
          return errorResult("DEEPSEEK_API_KEY is not configured. Add it to the gateway environment or auth store.");
        }
        const config = resolveConfig(deps.getPluginConfig());
        const client = makeClient(deps, apiKey);
        const maxSearches = clampNumber(args.max_searches, 1, 16, 6);
        const cache = deps.getCache();
        const cachedResearch = cache.get("research", config.model, `${args.question}|${maxSearches}`);
        if (cachedResearch) {
          return okResult(`${cachedResearch.text}\n\n(from response cache)`, cachedResearch.details);
        }

        let searchesUsed = 0;
        const searchToolSchema: DeepSeekToolSchema = {
          type: "function",
          function: {
            name: "web_search",
            description: "Search the web. Returns titled results with snippets.",
            parameters: {
              type: "object",
              properties: {
                query: { type: "string", description: "Search query." },
              },
              required: ["query"],
            },
          },
        };

        const result = await runAgentLoop({
          requester: client,
          model: config.model,
          reasoningEffort: config.defaultReasoningEffort,
          systemPrompt: RESEARCH_SYSTEM_PROMPT,
          prompt: args.question,
          tools: [
            {
              schema: searchToolSchema,
              execute: async (toolArgs) => {
                if (searchesUsed >= maxSearches) {
                  return `Search budget exhausted (${maxSearches}). Synthesize from the results you already have.`;
                }
                const query = typeof toolArgs.query === "string" ? toolArgs.query : String(toolArgs.query ?? "");
                if (!query.trim()) return "Error: empty query.";
                searchesUsed += 1;
                if (!deps.webSearch) {
                  return "Error: no web search provider is configured on this OpenClaw gateway. Synthesize from your own knowledge and say so.";
                }
                try {
                  const searchResult = await deps.webSearch({ query, count: 6 });
                  const text = textOfContent(searchResult?.content);
                  return text || "(no results)";
                } catch (error) {
                  return `Search failed: ${error instanceof Error ? error.message : String(error)}`;
                }
              },
            },
          ],
          maxSteps: config.researchMaxSteps,
          maxTokensPerStep: config.researchMaxTokensPerStep,
        });

        if (result.stopReason === "error") {
          return errorResult(`DeepSeek research loop failed: ${result.error}`);
        }
        if (result.stopReason === "stepBudget") {
          return okResult(
            `${result.answer || result.allText}\n\n(research loop hit its step budget; findings may be partial)`,
            { iterations: result.iterations, searchesUsed, stopReason: result.stopReason },
          );
        }
        const details = {
          iterations: result.iterations,
          searchesUsed,
          stopReason: result.stopReason,
        };
        cache.set("research", config.model, `${args.question}|${maxSearches}`, {
          text: result.answer || "(no answer produced)",
          details,
        });
        return okResult(result.answer || "(no answer produced)", details);
      },
    },

    reason: {
      name: "deepseek_reason",
      label: "DeepSeek Second Opinion",
      description:
        "Ask DeepSeek for an independent second opinion on a question + draft answer. Returns verdict, flaws, and an improved answer. Cheap cross-model verification.",
      parameters: reasonSchema,
      async execute(_id: string, params: unknown) {
        const args = params as { question: string; draft_answer: string; model?: string };
        const apiKey = await deps.resolveApiKey();
        if (!apiKey) {
          return errorResult("DEEPSEEK_API_KEY is not configured.");
        }
        const config = resolveConfig(deps.getPluginConfig());
        const client = makeClient(deps, apiKey);
        const model = (args.model ?? "").trim() || config.model;

        const prompt = [
          "## Question",
          args.question,
          "",
          "## Draft answer",
          args.draft_answer,
          "",
          "Review the draft answer now.",
        ].join("\n");

        const cache = deps.getCache();
        const cacheKey = `${model}|${args.question}|${args.draft_answer.slice(0, 2000)}`;
        const cachedReason = cache.get("reason", model, cacheKey);
        if (cachedReason) {
          return okResult(`${cachedReason.text}\n\n(from response cache)`, cachedReason.details);
        }
        const result = await runSingleCompletion({
          requester: client,
          model,
          systemPrompt: REASON_SYSTEM_PROMPT,
          prompt,
          maxTokens: Math.min(config.researchMaxTokensPerStep, 8192),
          reasoningEffort: "low" as DeepSeekReasoningEffort,
        });
        if (!result.text.trim()) {
          return errorResult("DeepSeek returned an empty review.");
        }
        const details = { model, usage: result.usage };
        cache.set("reason", model, cacheKey, { text: result.text, details });
        return okResult(result.text, details);
      },
    },

    balance: {
      name: "deepseek_balance",
      label: "DeepSeek Balance",
      description: "Check the DeepSeek API account balance (total remaining credits).",
      parameters: balanceSchema,
      async execute() {
        const apiKey = await deps.resolveApiKey();
        if (!apiKey) {
          return errorResult("DEEPSEEK_API_KEY is not configured.");
        }
        const client = makeClient(deps, apiKey);
        try {
          const balance = await client.getBalance();
          const currency = balance.currency === "CNY" ? "¥" : balance.currency ? `${balance.currency} ` : "";
          return okResult(`DeepSeek account balance: ${currency}${balance.balance}`, {
            currency: balance.currency,
            balance: balance.balance,
          });
        } catch (error) {
          return errorResult(`Balance check failed: ${error instanceof Error ? error.message : String(error)}`);
        }
      },
    },

    cascade: {
      name: "deepseek_cascade",
      label: "DeepSeek Cascade (universal)",
      description:
        "Speculative multi-tier execution with a judge: tier-1 drafts cheaply, a judge grades the draft, low-confidence answers escalate through the remaining tiers at max reasoning. Tiers default to [deepseek-flash, deepseek-v4-pro] but accept ANY gateway model refs (e.g. anthropic/claude-..., openai/gpt-..., zai/glm-...). Cheaper than always using the strongest model, stronger than always using one.",
      parameters: Type.Object({
        task: Type.String({ description: "The task or question to execute." }),
        system_prompt: Type.Optional(Type.String({ description: "Optional system prompt for the executor." })),
        tiers: Type.Optional(
          Type.Array(Type.String(), {
            description: "Model refs from cheap to strong. Default [deepseek-flash, deepseek-v4-pro]. Cross-provider refs allowed, e.g. [deepseek-flash, anthropic/claude-sonnet-4-6].",
          }),
        ),
        judge: Type.Optional(Type.String({ description: "Model ref for the judge. Defaults to the first tier." })),
      }),
      async execute(_id: string, params: unknown) {
        const args = params as { task: string; system_prompt?: string; tiers?: string[]; judge?: string };
        const apiKey = await deps.resolveApiKey();
        const config = resolveConfig(deps.getPluginConfig());
        const systemPrompt =
          args.system_prompt?.trim() ||
          "You are an expert assistant. Answer precisely and completely in the user's language.";
        const tiers = args.tiers && args.tiers.length > 0 ? args.tiers.slice(0, 4) : ["deepseek-flash", "deepseek-v4-pro"];
        const judgeRef = args.judge?.trim() || tiers[0];
        const needsNative = tiers.some((t) => isNativeDeepSeekRef(t)) || isNativeDeepSeekRef(judgeRef);
        const needsHost = tiers.some((t) => !isNativeDeepSeekRef(t)) || !isNativeDeepSeekRef(judgeRef);

        if (needsNative && !apiKey) {
          return errorResult("DEEPSEEK_API_KEY is not configured but DeepSeek tiers are requested.");
        }
        if (needsHost && !deps.hostLlm) {
          return errorResult(
            "Cross-provider tiers require the host LLM runtime (api.runtime.llm), which is unavailable in this gateway build.",
          );
        }

        const nativeExecutor = needsNative
          ? createDeepSeekNativeExecutor(makeClient(deps, apiKey as string))
          : undefined;
        const hostExecutor = needsHost && deps.hostLlm
          ? createHostLlmExecutor(deps.hostLlm, { pluginId: "openclaw-deepseek" })
          : undefined;
        const resolveExecutor = (ref: string): UniversalExecutor => {
          if (isNativeDeepSeekRef(ref) && nativeExecutor) return nativeExecutor;
          if (hostExecutor) return hostExecutor;
          if (nativeExecutor) return nativeExecutor;
          throw new Error(`No executor available for model ref "${ref}"`);
        };

        try {
          const result = await runUniversalCascade({
            resolveExecutor,
            tiers,
            judgeRef,
            systemPrompt,
            prompt: args.task,
            maxTokens: Math.min(config.researchMaxTokensPerStep, 8192),
            judgeThreshold: config.judgeThreshold,
            policy: deps.getPolicy(),
            guardian: deps.getGuardian(),
            audit: deps.getAudit(),
          });
          const costUsd = result.usage.costUsd;
          const summary = [
            result.text,
            "",
            "---",
            `cascade: final model ${result.finalRef}${result.escalated ? " (escalated)" : " (draft accepted)"}`,
            `judge (${judgeRef}): ${result.judge.grade} (confidence ${result.judge.confidence.toFixed(2)}) - ${result.judge.reason}`,
            `tokens: ${result.usage.inputTokens ?? 0} in / ${result.usage.outputTokens ?? 0} out${costUsd !== undefined ? ` | cost ~$${costUsd.toFixed(4)}` : ""}`,
          ].join("\n");
          return okResult(summary, {
            finalRef: result.finalRef,
            escalated: result.escalated,
            judge: result.judge,
            iterations: result.iterations,
            usage: result.usage,
            calls: result.calls,
          });
        } catch (error) {
          return errorResult(`Cascade failed: ${error instanceof Error ? error.message : String(error)}`);
        }
      },
    },

    extract: {
      name: "deepseek_extract",
      label: "DeepSeek Structured Extraction",
      description:
        "Extract structured JSON from text using DeepSeek JSON mode with schema validation and one automatic repair retry. Provide text and a JSON schema description.",
      parameters: Type.Object({
        text: Type.String({ description: "The source text to extract from." }),
        schema_description: Type.Optional(
          Type.String({ description: "Optional JSON schema describing the desired output object." }),
        ),
        fields: Type.Optional(
          Type.Array(Type.String({ description: "Field names to extract when no schema is given." })),
        ),
      }),
      async execute(_id: string, params: unknown) {
        const args = params as { text: string; schema_description?: string; fields?: string[] };
        const apiKey = await deps.resolveApiKey();
        if (!apiKey) return errorResult("DEEPSEEK_API_KEY is not configured.");
        const config = resolveConfig(deps.getPluginConfig());
        const client = makeClient(deps, apiKey);
        const schema =
          args.schema_description && args.schema_description.trim().startsWith("{")
            ? safeJsonParse(args.schema_description) ?? defaultExtractSchema(args.fields)
            : defaultExtractSchema(args.fields);
        try {
          const result = await runJsonExtraction({
            requester: client,
            prompt: `Extract the requested fields from the following text.\n\n---\n${args.text}`,
            schema,
            model: config.model,
          });
          return okResult(JSON.stringify(result.value, null, 2), {
            value: result.value,
            repaired: result.repaired,
            usage: result.usage,
          });
        } catch (error) {
          return errorResult(`Extraction failed: ${error instanceof Error ? error.message : String(error)}`);
        }
      },
    },

    fim: {
      name: "deepseek_fim",
      label: "DeepSeek Code Completion (FIM)",
      description:
        "Fill-in-the-middle code completion: given code with a gap (prefix + optional suffix), returns the code that belongs in the middle. Powered by DeepSeek's beta FIM endpoint.",
      parameters: Type.Object({
        prefix: Type.String({ description: "Code before the cursor/gap." }),
        suffix: Type.Optional(Type.String({ description: "Code after the gap." })),
        max_tokens: Type.Optional(Type.Number({ description: "Max completion tokens. Default 512." })),
      }),
      async execute(_id: string, params: unknown) {
        const args = params as { prefix: string; suffix?: string; max_tokens?: number };
        const apiKey = await deps.resolveApiKey();
        if (!apiKey) return errorResult("DEEPSEEK_API_KEY is not configured.");
        const config = resolveConfig(deps.getPluginConfig());
        const client = makeClient(deps, apiKey);
        try {
          const result = await runFimCompletion({
            requester: client,
            prompt: args.prefix,
            suffix: args.suffix,
            model: config.model,
            maxTokens: clampNumber(args.max_tokens, 16, 8192, 512),
          });
          return okResult(result.completion, { usage: result.usage });
        } catch (error) {
          return errorResult(`FIM completion failed: ${error instanceof Error ? error.message : String(error)}`);
        }
      },
    },

    genesis: {
      name: "deepseek_genesis",
      label: "Cascade Self-Calibration (Genesis)",
      description:
        "Calibrate the adaptive cascade policy: runs tiny probe tasks through the tier-0 model, learns acceptance rates per task shape, and reports the learned routing table. Also shows current policy state. Run once after install and occasionally after model changes.",
      parameters: Type.Object({
        reset: Type.Optional(Type.Boolean({ description: "Discard learned stats before calibrating. Default false." })),
        probes: Type.Optional(
          Type.Number({ description: "Probe tasks per shape category (code/reasoning/factual/creative/general). Default 1 (5 probes total)." }),
        ),
      }),
      async execute(_id: string, params: unknown) {
        const args = params as { reset?: boolean; probes?: number };
        const apiKey = await deps.resolveApiKey();
        if (!apiKey) return errorResult("DEEPSEEK_API_KEY is not configured.");
        const config = resolveConfig(deps.getPluginConfig());
        const policy = deps.getPolicy();
        if (args.reset) policy.clearForTest();

        const probeCount = clampNumber(args.probes, 1, 3, 1);
        const client = makeClient(deps, apiKey);
        const tier0 = config.model;

        const probes: Array<{ prompt: string; expectKey: string }> = [
          { prompt: "Write a TypeScript one-liner to reverse a string array.", expectKey: "code" },
          { prompt: "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies? Answer yes or no with one sentence.", expectKey: "reasoning" },
          { prompt: "What is the capital of Australia?", expectKey: "factual" },
          { prompt: "Write a two-line haiku about persistent memory.", expectKey: "creative" },
          { prompt: "Summarize the tradeoff between SQL and NoSQL databases in one sentence.", expectKey: "general" },
        ];

        const results: Array<{ shape: string; grade: string; confidence: number }> = [];
        for (let round = 0; round < probeCount; round++) {
          for (const probe of probes) {
            const judgePrompt = `Task:\n${probe.prompt}\n\nDraft answer:\n(none - grade the tier-0 model's expected reliability)\n\nJudge the tier-0 model's expected reliability for this task type now.`;
            // One judge call per probe: measures how a judge views tier-0 reliability for this shape.
            const executor = createDeepSeekNativeExecutor(client);
            const judgeResult = await executor.complete({
              ref: tier0,
              systemPrompt: `You are grading whether a fast, cheap model (tier-0) can reliably handle tasks of this shape without escalation to a stronger model. Reply with one JSON line: {"grade":"accept|escalate","confidence":0.0-1.0,"reason":"..."}. Grade "accept" when the shape is routine for a fast model.`,
              prompt: judgePrompt,
              maxTokens: 120,
              reasoning: "none",
              temperature: 0,
            });
            const verdict = parseVerdictShared(judgeResult.text);
            results.push({ shape: probe.expectKey, grade: verdict.grade, confidence: verdict.confidence });
            // Seed the policy with the judged outcome.
            policy.record({
              tier0,
              finalTier: tier0,
              prompt: probe.prompt,
              accepted: verdict.grade === "accept",
              escalated: verdict.grade === "escalate",
            });
          }
        }

        const ledger = deps.getLedger();
        const guardian = deps.getGuardian();
        const audit = deps.getAudit();
        const guardianStatus = guardian.status();
        let balanceLine = "  balance: (no DEEPSEEK_API_KEY)";
        try {
          const balance = await client.getBalance();
          balanceLine = `  balance: ${balance.currency === "CNY" ? "\u00a5" : ""}${balance.balance}`;
        } catch {
          // Balance endpoint unreachable: the health check continues with the
          // other sections; balance is informational, not critical.
          balanceLine = "  balance: unavailable";
        }
        const lines = [
          "Adaptive cascade policy calibrated.",
          `Probes: ${results.length} across ${probeCount} round(s).`,
          "",
          "Learned routing table:",
          ...policy.reportTable(),
          "",
          "Routing rules:",
          `  skip-judge (fast path): acceptance >= 85% over >= 6 samples of the same shape`,
          `  direct-strong: acceptance < 30% over >= 10 samples of the task category`,
          `  judge (default): otherwise; skip-judge runs are audited every 4th time`,
          "",
          "Economic state:",
          `  budget: ${guardianStatus.state}${guardianStatus.percent !== null ? ` (${guardianStatus.percent}% of $${guardianStatus.limitUsd?.toFixed(2)})` : " (dailyBudgetUsd not configured)"}`,
          balanceLine,
          `  pricing catalog: ${ledger.pricingTable.catalogReady ? `${ledger.pricingTable.catalogSize} models` : "not loaded"}`,
          "",
          "Concurrency windows (AIMD):",
          ...deps.getGovernor().reportTable(),
          `Response cache: ${deps.getCache().stats().entries} entries, hit rate ${Math.round(deps.getCache().stats().hitRate * 100)}%`,
          "",
          "Recent decisions:",
          audit.render({ limit: 5 }),
        ];
        return okResult(lines.join("\n"), {
          probes: results,
          tier0,
          guardian: guardianStatus,
          shapes: [...new Set(probes.map((p) => classifyTask(p.prompt)))].map(
            (s) => `${s} (${lengthBucketOf(40)})`,
          ),
        });
      },
    },

    costReport: {
      name: "deepseek_cost_report",
      label: "Live Cost Dashboard (all providers)",
      description:
        "Real-time cost dashboard for EVERY model call in this gateway (any provider). Views: 'live' (default) aggregates spend per model/session/day from the in-memory ledger; 'pricing' shows peak/off-peak unit prices; 'estimate' prices a hypothetical request; 'credits' shows DeepSeek account balance plus provider billing/credit windows when available.",
      parameters: Type.Object({
        view: Type.Optional(
          Type.String({ description: "live | pricing | estimate | credits. Default live." }),
        ),
        session: Type.Optional(Type.String({ description: "Scope the live view to one session key." })),
        window_hours: Type.Optional(
          Type.Number({ description: "Limit the live view to the last N hours. Default: all tracked history." }),
        ),
        prompt_tokens: Type.Optional(Type.Number({ description: "estimate view: hypothetical prompt tokens (cache miss)." })),
        cached_tokens: Type.Optional(Type.Number({ description: "estimate view: hypothetical cache-hit prompt tokens." })),
        completion_tokens: Type.Optional(Type.Number({ description: "estimate view: hypothetical output tokens." })),
        model: Type.Optional(
          Type.String({ description: "estimate/pricing view: any model ref, e.g. deepseek-flash, anthropic/claude-sonnet-4-6, openai/gpt-5.5. Default deepseek-flash." }),
        ),
      }),
      async execute(_id: string, params: unknown) {
        const args = params as {
          view?: string;
          session?: string;
          window_hours?: number;
          prompt_tokens?: number;
          cached_tokens?: number;
          completion_tokens?: number;
          model?: string;
        };
        const view = (args.view ?? "live").trim().toLowerCase();
        const ledger = deps.getLedger();
        const pricing = ledger.pricingTable;
        // Warm the universal catalog in the background (non-blocking best effort).
        void pricing.ensureCatalog().catch(() => {});

        if (view === "pricing") {
          const peak = isPeakNow();
          const modelRef = args.model ?? "deepseek-flash";
          const prices = pricing.lookup(modelRef, peak);
          const lines = [
            `DeepSeek window: ${peak ? "PEAK (2x prices)" : "OFF-PEAK (standard prices)"}`,
            PEAK_WINDOW_DESCRIPTION,
            "",
            `Unit prices for ${modelRef} (per million tokens):`,
            prices
              ? `  input $${prices.input.toFixed(4)} | output $${prices.output.toFixed(4)}${prices.cacheRead !== undefined ? ` | cache-hit $${prices.cacheRead.toFixed(4)}` : ""} (${prices.source})`
              : "  no price found (unpriced; tracked in tokens only)",
            "",
            "DeepSeek table:",
            unitPriceTable(peak),
          ];
          return okResult(lines.join("\n"), { peak, prices, catalogSize: pricing.catalogSize });
        }

        if (view === "estimate") {
          const peak = isPeakNow();
          const modelRef = args.model ?? "deepseek-flash";
          const usage = {
            input: (args.prompt_tokens ?? 0) + (args.cached_tokens ?? 0),
            cacheRead: args.cached_tokens ?? 0,
            output: args.completion_tokens ?? 0,
          };
          const priced = pricing.price(modelRef, usage, { deepSeekPeak: peak });
          const lines = [
            `Estimate for ${usage.input} prompt tokens (${usage.cacheRead} cache-hit) + ${usage.output} output on ${modelRef}:`,
            priced.priced
              ? `  $${(priced.costUsd ?? 0).toFixed(4)} (${priced.source} pricing${peak ? ", DeepSeek peak 2x" : ""})`
              : "  unpriced model: tracked in tokens only (no cost estimate)",
          ];
          return okResult(lines.join("\n"), { peak, priced, usage });
        }

        if (view === "credits") {
          const apiKey = await deps.resolveApiKey();
          const lines: string[] = ["Provider accounts & credits:"];
          const details: Record<string, unknown> = {};
          if (apiKey) {
            try {
              const client = makeClient(deps, apiKey);
              const balance = await client.getBalance();
              const currency = balance.currency === "CNY" ? "\u00a5" : balance.currency ? `${balance.currency} ` : "";
              lines.push(`  DeepSeek balance: ${currency}${balance.balance}`);
              details.deepseekBalance = balance.balance;
              details.deepseekCurrency = balance.currency;
            } catch (error) {
              lines.push(
                `  DeepSeek balance: unavailable (${error instanceof Error ? error.message : String(error)})`,
              );
            }
          } else {
            lines.push("  DeepSeek balance: no DEEPSEEK_API_KEY configured");
          }
          const snapshot = deps.getUsageSummary ? deps.getUsageSummary() : undefined;
          if (snapshot && snapshot.credits.length > 0) {
            for (const credit of snapshot.credits) {
              lines.push(`  ${credit.label}: ${credit.display}`);
            }
            details.credits = snapshot.credits;
          } else {
            lines.push("  (other providers: no billing/credit snapshots registered)");
          }
          return okResult(lines.join("\n"), details);
        }

        // Default: live dashboard.
        const sinceMs = args.window_hours ? clampNumber(args.window_hours, 1, 24 * 30, 1) * 3_600_000 : undefined;
        const text = ledger.render({ sessionKey: args.session, sinceMs: sinceMs ? Date.now() - sinceMs : undefined });
        const summary = ledger.summary({ sessionKey: args.session, sinceMs: sinceMs ? Date.now() - sinceMs : undefined });
        const guardian = deps.getGuardian().status();
        const auditTail = deps.getAudit().render({ limit: 5 });
        const cacheStats = deps.getCache().stats();
        return okResult(`${text}\n\nEconomic state: ${guardian.state}${guardian.percent !== null ? ` (${guardian.percent}%)` : ""}\n\nResponse cache: ${cacheStats.entries} entries, ${Math.round(cacheStats.hitRate * 100)}% hit rate\n\n${auditTail}`, {
          summary,
          guardian,
          cache: cacheStats,
          catalogSize: pricing.catalogSize,
        });
      },
    },
  };


  function okResult(text: string, details: unknown) {
    return { content: [{ type: "text", text }], details };
  }
  function errorResult(text: string) {
    return { content: [{ type: "text", text }], details: { error: true } };
  }
}


function safeJsonParse(text: string): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(text);
    if (parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
    return null;
  } catch {
    // Schema description was not valid JSON: the caller falls back to a
    // default extraction schema.
    return null;
  }
}

function defaultExtractSchema(fields?: string[]): Record<string, unknown> {
  const names = fields && fields.length > 0 ? fields : ["summary", "entities", "dates", "numbers"];
  const properties: Record<string, unknown> = {};
  for (const name of names) {
    properties[name] = { type: "string", description: `Extracted ${name}, empty string if absent.` };
  }
  return {
    type: "object",
    properties,
    required: names,
    additionalProperties: false,
  };
}
