/**
 * Companion provider `openclaw-deepseek`.
 *
 * Registers current DeepSeek model refs routed through the native harness.
 * Auth reuses the built-in deepseek provider via providerAuthAliases, so one
 * DEEPSEEK_API_KEY powers both the built-in transport and this plugin.
 * Retired model names (deepseek-chat, deepseek-reasoner, ...) normalize to
 * their successors with a migration warning.
 */

export interface ProviderCatalogContext {
  resolveProviderApiKey?: (id: string) => { apiKey?: string };
}
export interface ProviderCatalogResult {
  provider: {
    baseUrl: string;
    apiKey?: string;
    api: string;
    models: Array<Record<string, unknown>>;
  };
}
export interface ProviderPlugin {
  id: string;
  label: string;
  docsPath?: string;
  envVars?: string[];
  normalizeModelId?: (modelId: string) => string;
  resolveDynamicModel?: (ctx: { modelId: string }) => Record<string, unknown>;
  catalog?: {
    order?: string;
    run: (ctx: ProviderCatalogContext) => Promise<ProviderCatalogResult | null>;
  };
  staticCatalog?: {
    order?: string;
    run: (ctx: ProviderCatalogContext) => Promise<ProviderCatalogResult>;
  };
}

import { mapClaudeRefToDeepSeek } from "./anthropic.js";
import {
  CURRENT_MODELS,
  DEFAULT_BASE_URL,
  PROVIDER_ID,
  RETIRED_ALIASES,
  resolveConfig,
  resolveModelId,
  type DeepSeekHarnessConfig,
  type DeepSeekModelInfo,
} from "./types.js";

export { CURRENT_MODELS, PROVIDER_ID };

/** Second provider: Anthropic-Messages transport with claude-* auto-mapping. */
export const ANTHROPIC_PROVIDER_ID = "openclaw-deepseek-anthropic";

export const ANTHROPIC_STATIC_MODELS = [
  { id: "claude-opus-4-8", name: "DeepSeek V4 Pro (via Anthropic endpoint, opus mapping)" },
  { id: "claude-sonnet-4-6", name: "DeepSeek Flash (via Anthropic endpoint, sonnet mapping)" },
  { id: "claude-haiku-4-5", name: "DeepSeek Flash (via Anthropic endpoint, haiku mapping)" },
].map((m) => ({
  ...m,
  reasoning: true,
  input: ["text"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 1_000_000,
  maxTokens: 384_000,
}));

export function createAnthropicProvider(deps: {
  getPluginConfig: () => DeepSeekHarnessConfig | undefined;
  resolveApiKey: () => Promise<string | undefined>;
}): ProviderPlugin {
  return {
    id: ANTHROPIC_PROVIDER_ID,
    label: "DeepSeek (Anthropic Messages endpoint)",
    docsPath: "/providers/deepseek",
    envVars: ["DEEPSEEK_API_KEY"],

    normalizeModelId(modelId: string): string {
      return modelId; // claude-* refs pass through; DeepSeek maps them server-side
    },

    resolveDynamicModel: (ctx: { modelId: string }) => {
      const config = resolveConfig(deps.getPluginConfig());
      const upstream = mapClaudeRefToDeepSeek(ctx.modelId);
      return {
        id: ctx.modelId, // keep the claude-* ref; the anthropic client maps it
        name: `${ctx.modelId} -> ${upstream}`,
        provider: ANTHROPIC_PROVIDER_ID,
        api: "anthropic-messages",
        baseUrl: `${config.baseUrl}/anthropic`,
        reasoning: true,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 1_000_000,
        maxTokens: 384_000,
      };
    },

    catalog: {
      order: "simple",
      run: async (ctx: ProviderCatalogContext) => {
        const apiKey =
          ctx?.resolveProviderApiKey?.(ANTHROPIC_PROVIDER_ID)?.apiKey ??
          ctx?.resolveProviderApiKey?.("deepseek")?.apiKey ??
          (await deps.resolveApiKey());
        if (!apiKey) return null;
        const config = resolveConfig(deps.getPluginConfig());
        return {
          provider: {
            baseUrl: `${config.baseUrl}/anthropic`,
            apiKey,
            api: "anthropic-messages",
            models: ANTHROPIC_STATIC_MODELS,
          },
        };
      },
    },

    staticCatalog: {
      order: "simple",
      run: async () => ({
        provider: {
          baseUrl: "https://api.deepseek.com/anthropic",
          api: "anthropic-messages",
          models: ANTHROPIC_STATIC_MODELS,
        },
      }),
    },
  };
}

/** Virtual cascade row: flash-speed answers with pro escalation built in. */
const CASCADE_CATALOG_ROW = {
  id: "deepseek-cascade",
  name: "DeepSeek Cascade (flash -> pro)",
  reasoning: true,
  input: ["text"],
  cost: { input: 0.15, output: 0.6, cacheRead: 0.003, cacheWrite: 0.15 },
  contextWindow: 1_000_000,
  maxTokens: 384_000,
};

let warnedAliases = new Set<string>();

function toCatalogModel(m: DeepSeekModelInfo) {
  return {
    id: m.id,
    name: m.name,
    reasoning: m.reasoning,
    input: m.vision ? (["text", "image"] as const) : (["text"] as const),
    cost: m.cost,
    contextWindow: m.contextWindow,
    maxTokens: m.maxTokens,
  };
}

export function createDeepSeekProvider(deps: {
  getPluginConfig: () => DeepSeekHarnessConfig | undefined;
  resolveApiKey: () => Promise<string | undefined>;
  onAliasWarning?: (retired: string, replacement: string) => void;
}): ProviderPlugin {
  return {
    id: PROVIDER_ID,
    label: "DeepSeek (native harness)",
    docsPath: "/providers/deepseek",
    envVars: ["DEEPSEEK_API_KEY"],

    normalizeModelId(modelId: string): string {
      const { modelId: resolved, retiredAlias } = resolveModelId(modelId);
      if (retiredAlias && !warnedAliases.has(retiredAlias)) {
        warnedAliases.add(retiredAlias);
        const message = `DeepSeek model "${retiredAlias}" was retired; routing to "${resolved}" (see https://api-docs.deepseek.com/updates). Update your config to use "${resolved}".`;
        deps.onAliasWarning?.(retiredAlias, resolved);
        console.warn(`[openclaw-deepseek] ${message}`);
      }
      return resolved;
    },

    resolveDynamicModel: (ctx: { modelId: string }) => {
      const config = resolveConfig(deps.getPluginConfig());
      const { modelId } = resolveModelId(ctx.modelId);
      const known = CURRENT_MODELS.find((m) => m.id === modelId);
      const reasoning = known?.reasoning ?? true;
      return {
        id: modelId,
        name: known?.name ?? `DeepSeek ${modelId}`,
        provider: PROVIDER_ID,
        api: "openai-completions",
        baseUrl: config.baseUrl || DEFAULT_BASE_URL,
        reasoning,
        input: known?.vision ? ["text", "image"] : ["text"],
        cost: known?.cost ?? { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: known?.contextWindow ?? 131072,
        maxTokens: known?.maxTokens ?? 8192,
      };
    },

    catalog: {
      order: "simple",
      run: async (ctx: ProviderCatalogContext) => {
        const apiKey =
          ctx.resolveProviderApiKey?.(PROVIDER_ID)?.apiKey ??
          ctx.resolveProviderApiKey?.("deepseek")?.apiKey ??
          (await deps.resolveApiKey());
        if (!apiKey) return null;
        const config = resolveConfig(deps.getPluginConfig());
        return {
          provider: {
            baseUrl: config.baseUrl || DEFAULT_BASE_URL,
            apiKey,
            api: "openai-completions",
            models: [...CURRENT_MODELS.map(toCatalogModel), CASCADE_CATALOG_ROW],
          },
        };
      },
    },

    staticCatalog: {
      order: "simple",
      run: async () => ({
        provider: {
          baseUrl: DEFAULT_BASE_URL,
          api: "openai-completions",
          models: [...CURRENT_MODELS.map(toCatalogModel), CASCADE_CATALOG_ROW],
        },
      }),
    },
  };
}

/** Test hook: clears the warned-alias set between tests. */
export function resetAliasWarningsForTest(): void {
  warnedAliases = new Set();
}

export const RETIRED_MODEL_ALIASES = RETIRED_ALIASES;
