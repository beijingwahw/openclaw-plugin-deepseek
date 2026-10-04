import { afterEach, describe, expect, it } from "vitest";
import type { Server } from "node:http";
import { createServer } from "node:http";
import { CostLedger } from "../src/ledger.js";
import { PricingTable, LITELLM_CATALOG_URL } from "../src/pricing.js";

let server: Server | undefined;

afterEach(async () => {
  if (server) {
    await new Promise<void>((resolve) => server!.close(() => resolve()));
    server = undefined;
  }
});

describe("PricingTable", () => {
  it("prices DeepSeek refs natively with peak multiplier", () => {
    const table = new PricingTable();
    const offPeak = table.price("deepseek-flash", { input: 1_000_000, output: 500_000 }, { deepSeekPeak: false });
    expect(offPeak.priced).toBe(true);
    expect(offPeak.source).toBe("deepseek-native");
    // 1M * 0.15 + 0.5M * 0.6
    expect(offPeak.costUsd).toBeCloseTo(0.45, 4);

    const peak = table.price("deepseek-flash", { input: 1_000_000, output: 500_000 }, { deepSeekPeak: true });
    expect(peak.costUsd).toBeCloseTo(0.9, 4);
  });

  it("prices other providers from the LiteLLM catalog", async () => {
    let fetchedUrl = "";
    const table = new PricingTable({
      fetchFn: (async (url: string) => {
        fetchedUrl = url;
        expect(url).toBe(LITELLM_CATALOG_URL);
        return new Response(
          JSON.stringify({
            "anthropic/claude-sonnet-4-6": {
              input_cost_per_token: 0.000003,
              output_cost_per_token: 0.000015,
              cache_read_input_token_cost: 0.0000003,
            },
            "openai/gpt-5.5": {
              input_cost_per_token: 0.00000125,
              output_cost_per_token: 0.00001,
            },
            "free/local-model": { max_tokens: 100 },
          }),
          { status: 200 },
        );
      }) as unknown as typeof fetch,
    });
    await table.ensureCatalog();
    expect(table.catalogReady).toBe(true);
    expect(table.catalogSize).toBe(2); // the free entry has no prices -> skipped

    const claude = table.price("anthropic/claude-sonnet-4-6", { input: 1_000_000, output: 200_000, cacheRead: 500_000 });
    expect(claude.priced).toBe(true);
    expect(claude.source).toBe("litellm");
    // 1M input with 0.5M cache-hit: 0.5M*3 + 0.5M*0.3 + 0.2M*15 = 4.65
    expect(claude.costUsd).toBeCloseTo(4.65, 3);

    // Bare model id also resolves when the catalog keys it that way.
    const bare = table.lookup("anthropic/claude-sonnet-4-6");
    expect(bare?.source).toBe("litellm");
    const prefixed = table.lookup("claude-sonnet-4-6");
    expect(prefixed?.source).toBe("litellm");
  });

  it("falls back to unpriced for unknown models", () => {
    const table = new PricingTable();
    const result = table.price("unknown/fancy-model", { input: 100, output: 100 });
    expect(result.priced).toBe(false);
    expect(result.costUsd).toBeNull();
    expect(result.source).toBe("unpriced");
  });

  it("prices the cascade virtual model at the flash tier", () => {
    const table = new PricingTable();
    const result = table.price("deepseek-cascade", { input: 100_000, output: 10_000 });
    expect(result.priced).toBe(true);
  });
});

describe("CostLedger", () => {
  it("aggregates spend per model, session, and day", () => {
    let clock = Date.parse("2026-10-05T03:00:00Z");
    const ledger = new CostLedger({ now: () => clock });
    const table = ledger.pricingTable;
    table.setCatalogForTest({
      "anthropic/claude-sonnet-4-6": { input: 3, output: 15, cacheRead: 0.3, source: "litellm" },
    });

    ledger.record({
      provider: "deepseek",
      model: "deepseek-flash",
      ref: "deepseek/deepseek-flash",
      sessionKey: "s1",
      usage: { input: 1_000_000, output: 100_000 },
    });
    clock += 60_000;
    ledger.record({
      provider: "anthropic",
      model: "claude-sonnet-4-6",
      ref: "anthropic/claude-sonnet-4-6",
      sessionKey: "s2",
      usage: { input: 200_000, output: 50_000 },
    });
    clock += 24 * 3_600_000; // next day
    ledger.record({
      provider: "unknown",
      model: "fancy-model",
      ref: "unknown/fancy-model",
      sessionKey: "s1",
      usage: { input: 500, output: 500 },
    });

    const all = ledger.summary();
    expect(all.calls).toBe(3);
    expect(all.byModel).toHaveLength(3);
    expect(all.byDay).toHaveLength(2);
    // flash: 1M*0.15 + 0.1M*0.6 = 0.21
    const flash = all.byModel.find((m) => m.ref === "deepseek/deepseek-flash");
    expect(flash?.costUsd).toBeCloseTo(0.21, 4);
    // claude: 0.2M*3 + 0.05M*15 = 1.35
    const claude = all.byModel.find((m) => m.ref === "anthropic/claude-sonnet-4-6");
    expect(claude?.costUsd).toBeCloseTo(1.35, 4);
    // unknown model: unpriced
    const unknown = all.byModel.find((m) => m.ref === "unknown/fancy-model");
    expect(unknown?.costUsd).toBe(0);
    expect(unknown?.unpricedCalls).toBe(1);
    expect(all.unpricedCalls).toBe(1);
    expect(all.costUsd).toBeCloseTo(1.56, 3);

    // Session scope.
    const s1 = ledger.summary({ sessionKey: "s1" });
    expect(s1.calls).toBe(2);
    expect(s1.byModel.every((m) => m.ref !== "anthropic/claude-sonnet-4-6")).toBe(true);

    // Time window: only today's tail.
    const recent = ledger.summary({ sinceMs: clock - 3_600_000 });
    expect(recent.calls).toBe(1);
    expect(recent.byModel[0]?.ref).toBe("unknown/fancy-model");
  });

  it("renders a readable dashboard", () => {
    const ledger = new CostLedger();
    ledger.record({
      provider: "deepseek",
      model: "deepseek-flash",
      ref: "deepseek/deepseek-flash",
      usage: { input: 2_500_000, output: 300_000, cacheRead: 1_000_000 },
    });
    const text = ledger.render();
    expect(text).toContain("Live model spend");
    expect(text).toContain("2.5M");
    expect(text).toContain("deepseek/deepseek-flash");
    expect(text).toContain("$0.40"); // 1.5M miss*0.15 + 1M hit*0.003 + 0.3M*0.6 = 0.408
  });

  it("caps the ring buffer", () => {
    const ledger = new CostLedger();
    for (let i = 0; i < 1200; i++) {
      ledger.record({ provider: "p", model: "m", ref: "p/m", usage: { input: 1 } });
    }
    expect(ledger.summary().calls).toBe(1000);
  });
});

describe("CostLedger persistence", () => {
  it("persists a snapshot and restores it across instances", async () => {
    const os = await import("node:os");
    const { mkdtemp, readFile, stat } = await import("node:fs/promises");
    const { tmpdir } = os;
    const stateDir = await mkdtemp(`${tmpdir()}/ledger-test-`);
    let clock = Date.now();

    const first = new CostLedger({ stateDir, now: () => clock });
    first.record({
      provider: "deepseek",
      model: "deepseek-flash",
      ref: "deepseek/deepseek-flash",
      sessionKey: "s1",
      usage: { input: 100_000, output: 10_000 },
    });
    await new Promise((r) => setTimeout(r, 2_600)); // debounce window
    const snapshotPath = `${stateDir}/openclaw-deepseek/ledger-snapshot.json`;
    const statResult = await stat(snapshotPath);
    expect(statResult.isFile()).toBe(true);
    const raw = JSON.parse(await readFile(snapshotPath, "utf8"));
    expect(raw.entries).toHaveLength(1);
    expect(raw.entries[0].ref).toBe("deepseek/deepseek-flash");

    // New instance (simulated restart) restores history.
    const second = new CostLedger({ stateDir, now: () => clock });
    await second.ready();
    const summary = second.summary();
    expect(summary.calls).toBe(1);
    expect(summary.byModel[0]?.ref).toBe("deepseek/deepseek-flash");
  });

  it("drops entries older than 30 days on restore", async () => {
    const os = await import("node:os");
    const { mkdtemp, mkdir, writeFile } = await import("node:fs/promises");
    const stateDir = await mkdtemp(`${os.tmpdir()}/ledger-stale-`);
    await mkdir(`${stateDir}/openclaw-deepseek`, { recursive: true });
    const now = Date.now();
    const stale = now - 40 * 24 * 3_600_000;
    await writeFile(
      `${stateDir}/openclaw-deepseek/ledger-snapshot.json`,
      JSON.stringify({
        savedAt: stale,
        entries: [
          { ts: stale, provider: "p", model: "old", ref: "p/old", usage: { input: 1 }, costUsd: 0, priced: true, priceSource: "deepseek-native" },
          { ts: now - 1000, provider: "p", model: "new", ref: "p/new", usage: { input: 2 }, costUsd: 0, priced: true, priceSource: "deepseek-native" },
        ],
      }),
      "utf8",
    );
    const ledger = new CostLedger({ stateDir });
    await ledger.ready();
    expect(ledger.summary().calls).toBe(1);
    expect(ledger.summary().byModel[0]?.ref).toBe("p/new");
  });
});
