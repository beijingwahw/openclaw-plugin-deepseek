import { afterEach, describe, expect, it } from "vitest";
import type { Server } from "node:http";
import { createServer } from "node:http";
import { ConcurrencyGovernor } from "../src/concurrency.js";
import { ResponseCache } from "../src/cache.js";
import { DeepSeekClient, DeepSeekApiError } from "../src/deepseek-api.js";
import plugin from "../dist/index.js";

let server: Server | undefined;

afterEach(async () => {
  if (server) {
    await new Promise<void>((resolve) => server!.close(() => resolve()));
    server = undefined;
  }
});

function sse(payload: unknown): string {
  return `data: ${JSON.stringify(payload)}\n\n`;
}

describe("AIMD concurrency governor", () => {
  it("grows the window linearly on success and halves it on rate-limit", () => {
    const governor = new ConcurrencyGovernor({ initialLimit: 4 });
    // 3 successes: 4 -> 5 -> 6 -> 7
    for (let i = 0; i < 3; i++) {
      governor.release("deepseek-flash", "success");
    }
    expect(governor.stats("deepseek-flash")[0].limit).toBe(7);

    // One 429: 7 -> 3
    governor.release("deepseek-flash", "rate-limit");
    expect(governor.stats("deepseek-flash")[0].limit).toBe(3);
    expect(governor.stats("deepseek-flash")[0].consecutiveRateLimits).toBe(1);
  });

  it("keeps windows per model (flash and pro are independent)", () => {
    const governor = new ConcurrencyGovernor({ initialLimit: 4 });
    governor.release("deepseek-flash", "rate-limit");
    expect(governor.stats("deepseek-flash")[0].limit).toBe(2);
    expect(governor.stats("deepseek-v4-pro")[0].limit).toBe(4); // untouched
  });

  it("enforces the in-flight cap with FIFO queuing", async () => {
    const governor = new ConcurrencyGovernor({ initialLimit: 2 });
    const order: string[] = [];
    const task = async (name: string, ms: number) => {
      await governor.acquire("m");
      order.push(`start:${name}`);
      await new Promise((r) => setTimeout(r, ms));
      governor.release("m", "success");
      order.push(`end:${name}`);
    };
    await Promise.all([task("a", 40), task("b", 40), task("c", 40)]);
    // a and b run concurrently; c waits for a slot (starts after some end event).
    const firstEnd = Math.min(order.indexOf("end:a"), order.indexOf("end:b"));
    expect(order.indexOf("start:c")).toBeGreaterThan(firstEnd);
    expect(order.filter((e) => e.startsWith("start:")).indexOf("start:c")).toBe(2); // third to start
  });

  it("enters cooldown after three consecutive rate-limits", async () => {
    let clock = 0;
    const governor = new ConcurrencyGovernor({ initialLimit: 4, now: () => clock });
    for (let i = 0; i < 3; i++) governor.release("m", "rate-limit");
    const stats = governor.stats("m")[0];
    expect(stats.cooldownActive).toBe(true);
    // Cooldown blocks new acquires for ~30s.
    clock = 29_999;
    expect(governor.stats("m")[0].cooldownActive).toBe(true);
    clock = 30_001;
    expect(governor.stats("m")[0].cooldownActive).toBe(false);
  });
});

describe("response cache", () => {
  it("caches by (kind, model, prompt) with LRU eviction", () => {
    let clock = 1000;
    const cache = new ResponseCache<string>({ maxEntries: 2, now: () => clock });
    cache.set("research", "m", "q1", "a1");
    clock += 1;
    cache.set("research", "m", "q2", "a2");
    clock += 1;
    expect(cache.get("research", "m", "q1")).toBe("a1"); // q1 becomes most recent
    clock += 1;
    cache.set("research", "m", "q3", "a3"); // evicts q2 (LRU)
    expect(cache.get("research", "m", "q2")).toBeUndefined();
    expect(cache.get("research", "m", "q3")).toBe("a3");
    // Different kind or model misses.
    expect(cache.get("reason", "m", "q1")).toBeUndefined();
    expect(cache.get("research", "other", "q1")).toBeUndefined();
  });

  it("expires entries after the TTL", () => {
    let clock = 0;
    const cache = new ResponseCache<string>({ ttlMs: 1000, now: () => clock });
    cache.set("research", "m", "q", "a");
    expect(cache.get("research", "m", "q")).toBe("a");
    clock = 1001;
    expect(cache.get("research", "m", "q")).toBeUndefined();
  });

  it("through() wraps producers with read-through semantics", async () => {
    let clock = 0;
    const cache = new ResponseCache<string>({ now: () => clock });
    let calls = 0;
    const producer = async () => {
      calls += 1;
      return "computed";
    };
    const first = await cache.through("research", "m", "q", producer);
    expect(first.cached).toBe(false);
    const second = await cache.through("research", "m", "q", producer);
    expect(second.cached).toBe(true);
    expect(calls).toBe(1); // producer ran once
    const stats = cache.stats();
    expect(stats.hits).toBe(1);
    expect(stats.hitRate).toBe(0.5);
  });
});

describe("client flows through the governor", () => {
  it("acquires, reports rate-limits, and releases per request", async () => {
    let status = 200;
    const governor = new ConcurrencyGovernor({ initialLimit: 4 });
    const client = new DeepSeekClient({
      baseUrl: "http://127.0.0.1:1",
      apiKey: "sk",
      timeoutMs: 1000,
      governor,
      fetchFn: (async () => {
        if (status === 429) {
          // Transient: only the first call is rate-limited; retries succeed.
          status = 200;
          return new Response(JSON.stringify({ error: "rate" }), { status: 429 });
        }
        return new Response(JSON.stringify({ choices: [{ message: { role: "assistant", content: "ok" } }] }), { status: 200 });
      }) as unknown as typeof fetch,
    });

    // Success path: window grows.
    await client.completeChat({ model: "m", messages: [] });
    expect(governor.stats("m")[0].limit).toBe(5);

    // One 429 followed by success: window halves then regrows.
    await client.completeChat({ model: "m", messages: [] });
    const limits = governor.stats("m")[0];
    expect(limits.limit).toBeGreaterThanOrEqual(3); // 5 -> 2 (429) -> 3 (success)
  });
});

describe("tools use the response cache (e2e)", () => {
  it("research hits the cache on identical questions", async () => {
    let calls = 0;
    const baseUrl = await new Promise<string>((resolve) => {
      server = createServer((req, res) => {
        let data = "";
        req.on("data", (c) => (data += c));
        req.on("end", () => {
          calls += 1;
          res.writeHead(200, { "Content-Type": "text/event-stream" });
          res.end(
            [
              sse({ choices: [{ delta: { content: "cached research answer" } }] }),
              sse({ choices: [{ delta: {}, finish_reason: "stop" }] }),
              "data: [DONE]\n\n",
            ].join(""),
          );
        });
      });
      server.listen(0, "127.0.0.1", () => resolve(`http://127.0.0.1:${(server!.address() as any).port}`));
    });

    const registered: any = { providers: [], harnesses: [], tools: [] };
    const api = {
      id: "openclaw-deepseek",
      pluginConfig: { baseUrl },
      registerProvider: (p: any) => registered.providers.push(p),
      registerAgentHarness: (h: any) => registered.harnesses.push(h),
      registerTool: (t: any) => registered.tools.push(t),
      runtime: {
        modelAuth: { resolveApiKeyForProvider: async () => "sk" },
        webSearch: { search: async () => ({ content: [{ type: "text", text: "result" }] }) },
        llm: { complete: async () => ({ text: "x", provider: "zai", model: "glm" }) },
      },
    };
    plugin.register(api);
    const research = registered.tools.find((t: any) => t.name === "deepseek_research");

    const first = await research.execute("t", { question: "q1", max_searches: 1 });
    const second = await research.execute("t", { question: "q1", max_searches: 1 });
    expect(first.content[0].text).toContain("cached research answer");
    expect(second.content[0].text).toContain("from response cache");
    expect(calls).toBe(1); // only the first call reached the provider
  });
});
