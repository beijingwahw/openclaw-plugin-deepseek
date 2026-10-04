import { describe, expect, it } from "vitest";
import { createDeepSeekHarness } from "../src/harness.js";
import {
  CURRENT_MODELS,
  createDeepSeekProvider,
  resetAliasWarningsForTest,
} from "../src/provider.js";
import {
  RETIRED_ALIASES,
  normalizeReasoningEffort,
  resolveConfig,
  resolveModelId,
} from "../src/types.js";

describe("resolveConfig", () => {
  it("applies defaults", () => {
    const config = resolveConfig(undefined);
    expect(config.baseUrl).toBe("https://api.deepseek.com");
    expect(config.model).toBe("deepseek-flash");
    expect(config.defaultReasoningEffort).toBe("high");
    expect(config.researchMaxSteps).toBe(8);
    expect(config.requestTimeoutMs).toBe(180_000);
  });

  it("clamps out-of-range values", () => {
    const config = resolveConfig({ researchMaxSteps: 1000, researchMaxTokensPerStep: 1 });
    expect(config.researchMaxSteps).toBe(32);
    expect(config.researchMaxTokensPerStep).toBe(256);
  });

  it("maps retired model aliases to successors", () => {
    expect(resolveModelId("deepseek-chat")).toEqual({ modelId: "deepseek-flash", retiredAlias: "deepseek-chat" });
    expect(resolveModelId("deepseek-reasoner")).toEqual({ modelId: "deepseek-flash", retiredAlias: "deepseek-reasoner" });
    expect(resolveModelId("deepseek-v4-flash")).toEqual({ modelId: "deepseek-flash", retiredAlias: "deepseek-v4-flash" });
    expect(resolveModelId("deepseek-flash")).toEqual({ modelId: "deepseek-flash", retiredAlias: null });
    expect(resolveModelId("deepseek-v4-pro")).toEqual({ modelId: "deepseek-v4-pro", retiredAlias: null });
    expect(Object.keys(RETIRED_ALIASES)).not.toContain("deepseek-flash");
  });

  it("normalizes reasoning effort words", () => {
    expect(normalizeReasoningEffort("off")).toBe("none");
    expect(normalizeReasoningEffort("minimal")).toBe("low");
    expect(normalizeReasoningEffort("medium")).toBe("high");
    expect(normalizeReasoningEffort("ultrathink")).toBe("high");
    expect(normalizeReasoningEffort("xhigh")).toBe("high");
    expect(normalizeReasoningEffort("max")).toBe("max");
    expect(normalizeReasoningEffort(undefined)).toBe("high");
  });
});

describe("provider", () => {
  it("resolves dynamic models with correct surfaces", () => {
    resetAliasWarningsForTest();
    const provider = createDeepSeekProvider({
      getPluginConfig: () => undefined,
      resolveApiKey: async () => "sk-test",
    });
    const flash = (provider as any).resolveDynamicModel({ modelId: "deepseek-flash" });
    expect(flash.provider).toBe("openclaw-deepseek");
    expect(flash.api).toBe("openai-completions");
    expect(flash.reasoning).toBe(true);
    expect(flash.input).toEqual(["text", "image"]);

    const pro = (provider as any).resolveDynamicModel({ modelId: "deepseek-v4-pro" });
    expect(pro.input).toEqual(["text"]);

    const custom = (provider as any).resolveDynamicModel({ modelId: "deepseek-some-new-model" });
    expect(custom.id).toBe("deepseek-some-new-model");
    expect(custom.baseUrl).toBe("https://api.deepseek.com");
  });

  it("normalizes retired model ids once with a warning", () => {
    resetAliasWarningsForTest();
    const warnings: string[] = [];
    const provider = createDeepSeekProvider({
      getPluginConfig: () => undefined,
      resolveApiKey: async () => "sk-test",
      onAliasWarning: (retired) => warnings.push(retired),
    });
    expect((provider as any).normalizeModelId("deepseek-chat")).toBe("deepseek-flash");
    expect((provider as any).normalizeModelId("deepseek-chat")).toBe("deepseek-flash");
    expect(warnings).toEqual(["deepseek-chat"]);
  });

  it("gates the live catalog on auth", async () => {
    const provider = createDeepSeekProvider({
      getPluginConfig: () => undefined,
      resolveApiKey: async () => undefined,
    });
    const noKey = await (provider as any).catalog.run({
      resolveProviderApiKey: () => ({ apiKey: undefined }),
    });
    expect(noKey).toBeNull();

    const withKey = await (provider as any).catalog.run({
      resolveProviderApiKey: (id: string) => ({ apiKey: id === "openclaw-deepseek" ? "sk-test" : undefined }),
    });
    expect(withKey.provider.models.map((m: any) => m.id)).toEqual([
      ...CURRENT_MODELS.map((m) => m.id),
      "deepseek-cascade",
    ]);
  });

  it("serves a network-free static catalog with the current official models", async () => {
    const provider = createDeepSeekProvider({
      getPluginConfig: () => undefined,
      resolveApiKey: async () => undefined,
    });
    const result = await (provider as any).staticCatalog.run();
    expect(result.provider.baseUrl).toBe("https://api.deepseek.com");
    const ids = result.provider.models.map((m: any) => m.id);
    expect(ids).toEqual(["deepseek-flash", "deepseek-v4-pro", "deepseek-cascade"]);
    expect(result.provider.models.find((m: any) => m.id === "deepseek-flash").input).toContain("image");
    expect(CURRENT_MODELS.find((m) => m.id === "deepseek-flash")?.contextWindow).toBe(1_000_000);
  });
});

describe("harness", () => {
  it("claims only openclaw-deepseek provider refs", () => {
    const harness = createDeepSeekHarness({
      resolveApiKey: async () => "sk-test",
      getPluginConfig: () => undefined,
    });
    expect((harness as any).supports({ provider: "openclaw-deepseek" })).toMatchObject({ supported: true });
    expect((harness as any).supports({ provider: "deepseek" })).toMatchObject({ supported: false });
    expect((harness as any).supports({ provider: "openai" })).toMatchObject({ supported: false });
  });

  it("fails closed without an API key", async () => {
    const harness = createDeepSeekHarness({
      resolveApiKey: async () => undefined,
      getPluginConfig: () => undefined,
    });
    const result = await (harness as any).runAttempt({
      sessionId: "s1",
      prompt: "hi",
      workspaceDir: "/tmp",
      timeoutMs: 5000,
    });
    expect(result.terminal.kind).toBe("failed");
  });
});
