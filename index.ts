/**
 * OpenClaw DeepSeek plugin.
 *
 * Adds:
 * - a companion provider `openclaw-deepseek` (reuses DEEPSEEK_API_KEY, current
 *   model catalog, retired-alias normalization)
 * - a native agent harness executing DeepSeek function-calling loops with
 *   thinking control, vision input, and reasoning_content replay
 * - deepseek_research / deepseek_reason / deepseek_balance / deepseek_cascade /
 *   deepseek_extract / deepseek_fim / deepseek_cost_report agent tools
 */
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { createDeepSeekHarness } from "./src/harness.js";
import { createAnthropicProvider, createDeepSeekProvider } from "./src/provider.js";
import { createDeepSeekTools } from "./src/tools.js";
import { CostLedger } from "./src/ledger.js";
import { AdaptiveCascadePolicy } from "./src/policy.js";
import { BudgetGuardian } from "./src/guardian.js";
import { AuditTrail } from "./src/audit.js";
import { ConcurrencyGovernor } from "./src/concurrency.js";
import { ResponseCache } from "./src/cache.js";
import { PLUGIN_ID, resolveConfig, type DeepSeekHarnessConfig } from "./src/types.js";

export { PLUGIN_ID };

interface PluginApiLike {
  id: string;
  pluginConfig?: Record<string, unknown>;
  runtime: {
    modelAuth: {
      resolveApiKeyForProvider(args: { provider: string; cfg?: unknown }): Promise<string | null | undefined>;
    };
    webSearch?: {
      search(args: { config?: unknown; args: { query: string; count?: number } }): Promise<{
        content: Array<{ type: string; text?: string }>;
      }>;
    };
    llm?: {
      complete(params: {
        messages: Array<{ role: string; content: string }>;
        model?: string;
        systemPrompt?: string;
        maxTokens?: number;
        temperature?: number;
        reasoning?: string;
        signal?: AbortSignal;
        purpose?: string;
      }): Promise<{
        text: string;
        provider: string;
        model: string;
        usage?: { inputTokens?: number; outputTokens?: number; totalTokens?: number; costUsd?: number };
      }>;
    };
    logging?: {
      getChildLogger(scope: { plugin: string }, opts?: { level?: string }): {
        debug: (msg: string, data?: unknown) => void;
        warn: (msg: string, data?: unknown) => void;
        error: (msg: string, data?: unknown) => void;
      };
    };
  };
  registerProvider(provider: unknown): void;
  registerAgentHarness(harness: unknown): void;
  registerTool(tool: unknown, opts?: { optional?: boolean }): void;
  logger?: {
    debug: (msg: string, data?: unknown) => void;
    warn: (msg: string, data?: unknown) => void;
    error: (msg: string, data?: unknown) => void;
  };
}

export default definePluginEntry({
  id: PLUGIN_ID,
  name: "OpenClaw DeepSeek",
  description:
    "DeepSeek native agent harness: companion provider with current model catalog, function-calling agent loop with thinking control and vision, research/reason tools, and balance monitoring",

  register(api) {
    const pluginApi = api as unknown as PluginApiLike;
    const logger =
      pluginApi.runtime.logging?.getChildLogger({ plugin: PLUGIN_ID }) ?? pluginApi.logger;

    const getPluginConfig = (): DeepSeekHarnessConfig | undefined =>
      (pluginApi.pluginConfig ?? undefined) as DeepSeekHarnessConfig | undefined;

    const resolveApiKey = async (): Promise<string | undefined> => {
      try {
        const key = await pluginApi.runtime.modelAuth.resolveApiKeyForProvider({
          provider: "deepseek",
        });
        if (key) return key;
      } catch (error) {
        logger?.warn("deepseek auth resolution failed", {
          error: error instanceof Error ? error.message : String(error),
        });
      }
      const envKey = process.env.DEEPSEEK_API_KEY;
      return envKey && envKey.trim() ? envKey.trim() : undefined;
    };

    // Provider: current model refs + retired-alias normalization.
    pluginApi.registerProvider(
      createDeepSeekProvider({
        getPluginConfig,
        resolveApiKey,
        onAliasWarning: (retired, replacement) => {
          logger?.warn("retired DeepSeek model alias", { retired, replacement });
        },
      }),
    );

    // Second provider: Anthropic-Messages transport (claude-* refs).
    pluginApi.registerProvider(
      createAnthropicProvider({
        getPluginConfig,
        resolveApiKey,
      }),
    );

    // Harness: native DeepSeek agent loop for openclaw-deepseek/*
    pluginApi.registerAgentHarness(
      createDeepSeekHarness({
        resolveApiKey,
        getPluginConfig,
        getGovernor: () => concurrencyGovernor,
        logger,
      }),
    );

    // Live cost ledger: every model call in the gateway (any provider) is
    // recorded through the llm_output hook and priced via the universal table.
    // Snapshots persist under the gateway state dir so restarts keep history.
    const stateDir = (pluginApi as unknown as {
      runtimeState?: { resolveStateDir(env?: NodeJS.ProcessEnv): string };
    }).runtimeState?.resolveStateDir(process.env);
    const ledger = new CostLedger(stateDir ? { stateDir } : {});
    const cascadePolicy = new AdaptiveCascadePolicy(stateDir ? { stateDir } : {});
    const auditTrail = new AuditTrail(stateDir ? { stateDir } : {});
    const concurrencyGovernor = new ConcurrencyGovernor();
    const responseCache = new ResponseCache<{ text: string; details: unknown }>();
    const budgetGuardian = new BudgetGuardian({
      ledger,
      onTransition: (from, to, status) => {
        auditTrail.record("budget-transition", `${from} -> ${to} (${status.percent ?? "?"}% of budget)`, { from, to, ...status });
        logger?.warn("budget state changed", { from, to, percent: status.percent });
      },
    });
    const applyBudget = () => {
      const config = (pluginApi.pluginConfig ?? undefined) as DeepSeekHarnessConfig | undefined;
      ledger.setDailyBudget(resolveConfig(config).dailyBudgetUsd);
    };
    applyBudget();
    void ledger.pricingTable.ensureCatalog().catch(() => {});
    const onHook = (pluginApi as unknown as PluginHookSurface).on;
    if (typeof onHook === "function") {
      onHook.call(pluginApi, "llm_output", (event: unknown) => {
        try {
          const evt = event as {
            provider?: string;
            model?: string;
            resolvedRef?: string;
            sessionKey?: string;
            usage?: { input?: number; output?: number; cacheRead?: number; cacheWrite?: number; total?: number };
          };
          if (!evt?.provider || !evt?.model) return;
          ledger.record({
            provider: evt.provider,
            model: evt.model,
            ref: evt.resolvedRef,
            sessionKey: evt.sessionKey,
            usage: evt.usage,
          });
        } catch {
          // Observation-only: never disturb the agent turn on ledger issues.
        }
      });
    }

    // Tools
    const tools = createDeepSeekTools({
      resolveApiKey,
      getPluginConfig,
      getLedger: () => ledger,
      getPolicy: () => cascadePolicy,
      getGuardian: () => budgetGuardian,
      getAudit: () => auditTrail,
      getGovernor: () => concurrencyGovernor,
      getCache: () => responseCache,
      hostLlm: pluginApi.runtime.llm,
      webSearch: (args) => {
        const webSearch = pluginApi.runtime.webSearch;
        if (!webSearch) return Promise.reject(new Error("no web search provider"));
        return webSearch.search({ args: { query: args.query, count: args.count } });
      },
      logger,
    });
    pluginApi.registerTool(tools.research);
    pluginApi.registerTool(tools.reason);
    pluginApi.registerTool(tools.balance);
    pluginApi.registerTool(tools.cascade);
    pluginApi.registerTool(tools.extract);
    pluginApi.registerTool(tools.fim);
    pluginApi.registerTool(tools.costReport);
    pluginApi.registerTool(tools.genesis);
  },
});

/** Minimal hook surface the plugin consumes (structural cast target). */
interface PluginHookSurface {
  on?: (events: string, handler: (event: unknown) => void) => void;
}
