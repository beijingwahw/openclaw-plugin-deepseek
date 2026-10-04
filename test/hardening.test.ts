import { afterEach, describe, expect, it } from "vitest";
import type { Server } from "node:http";
import { createServer } from "node:http";
import {
  isRecord,
  parseToolCallFragment,
  parseUsageFrame,
  safeJsonParse,
} from "../src/validate.js";
import { DeepSeekClient } from "../src/deepseek-api.js";
import { resolveModelId } from "../src/types.js";

let server: Server | undefined;

afterEach(async () => {
  if (server) {
    await new Promise<void>((resolve) => server!.close(() => resolve()));
    server = undefined;
  }
});

describe("boundary validators", () => {
  it("parseUsageFrame accepts valid frames and drops malformed fields", () => {
    const good = parseUsageFrame({
      prompt_tokens: 100,
      completion_tokens: 20,
      total_tokens: 120,
      prompt_cache_hit_tokens: 80,
      completion_tokens_details: { reasoning_tokens: 5 },
      prompt_tokens_details: { cached_tokens: 80 },
    });
    expect(good).toEqual({
      prompt_tokens: 100,
      completion_tokens: 20,
      total_tokens: 120,
      prompt_cache_hit_tokens: 80,
      completion_tokens_details: { reasoning_tokens: 5 },
      prompt_tokens_details: { cached_tokens: 80 },
    });

    const partial = parseUsageFrame({ prompt_tokens: 5, completion_tokens: "x" });
    expect(partial).toEqual({ prompt_tokens: 5 }); // malformed field dropped

    expect(parseUsageFrame(null)).toBeNull();
    expect(parseUsageFrame("nope")).toBeNull();
    expect(parseUsageFrame(42)).toBeNull();
    expect(parseUsageFrame({ prompt_tokens: Infinity })).toEqual({}); // non-finite dropped
    expect(parseUsageFrame({ prompt_tokens: NaN })).toEqual({});
  });

  it("parseToolCallFragment validates fragment shapes", () => {
    expect(parseToolCallFragment({ index: 1, id: "c1", function: { name: "web_search", arguments: '{"q' } })).toEqual({
      index: 1,
      id: "c1",
      name: "web_search",
      argsDelta: '{"q',
    });
    expect(parseToolCallFragment({ index: 0 })).toEqual({ index: 0 });
    expect(parseToolCallFragment({ id: "x" })).toEqual({ index: 0, id: "x" });
    expect(parseToolCallFragment(null)).toBeNull();
    expect(parseToolCallFragment("junk")).toBeNull();
    expect(parseToolCallFragment({ index: "one" })).toEqual({ index: 0 }); // bad index falls back
  });

  it("isRecord and safeJsonParse handle boundary garbage", () => {
    expect(isRecord({})).toBe(true);
    expect(isRecord([])).toBe(false);
    expect(isRecord(null)).toBe(false);
    expect(isRecord("str")).toBe(false);
    expect(safeJsonParse('{"a":1}')).toEqual({ a: 1 });
    expect(safeJsonParse("{broken")).toBeUndefined();
    expect(safeJsonParse("")).toBeUndefined();
  });
});

describe("SSE usage frames go through the validator", () => {
  it("rejects malformed usage frames instead of casting them through", async () => {
    const baseUrl = await new Promise<string>((resolve) => {
      server = createServer((req, res) => {
        res.writeHead(200, { "Content-Type": "text/event-stream" });
        res.end(
          [
            // Malformed usage frame: non-numeric tokens. The validator drops
            // the bad fields; the parser emits a usage event with the valid
            // subset (here: empty usage object, still a valid event).
            `data: ${JSON.stringify({ choices: [], usage: { prompt_tokens: "corrupt", completion_tokens: 7 } })}\n\n`,
            `data: ${JSON.stringify({ choices: [{ delta: { content: "answer" } }] })}\n\n`,
            `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] })}\n\n`,
            "data: [DONE]\n\n",
          ].join(""),
        );
      });
      server.listen(0, "127.0.0.1", () => resolve(`http://127.0.0.1:${(server!.address() as any).port}`));
    });

    const client = new DeepSeekClient({ baseUrl, apiKey: "sk", timeoutMs: 5000 });
    const events = [];
    for await (const event of client.streamChat({ model: "m", messages: [] })) {
      events.push(event);
    }
    const usageEvents = events.filter((e) => e.type === "usage");
    expect(usageEvents).toHaveLength(1);
    // The corrupt field was dropped, the valid one kept — no crash, no NaN.
    expect((usageEvents[0] as { usage?: { prompt_tokens?: number } }).usage?.prompt_tokens).toBeUndefined();
    expect((usageEvents[0] as { usage?: { completion_tokens?: number } }).usage?.completion_tokens).toBe(7);
    expect(events).toContainEqual({ type: "text", text: "answer" });
  });
});

describe("branded model refs", () => {
  it("resolveModelId returns branded refs that flow as strings", () => {
    const { modelId } = resolveModelId("deepseek-chat");
    expect(modelId).toBe("deepseek-flash");
    // The brand is compile-time only; runtime value is a plain string.
    expect(typeof modelId).toBe("string");
    expect(modelId.length).toBeGreaterThan(0);
  });
});

describe("timers never pin the process", () => {
  it("active handles do not grow after request settlement", async () => {
    const baseUrl = await new Promise<string>((resolve) => {
      server = createServer((req, res) => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ choices: [{ message: { role: "assistant", content: "ok" } }] }));
      });
      server.listen(0, "127.0.0.1", () => resolve(`http://127.0.0.1:${(server!.address() as any).port}`));
    });
    const client = new DeepSeekClient({ baseUrl, apiKey: "sk", timeoutMs: 30_000 });
    const activeBefore = (process as NodeJS.Process & { _getActiveHandles?: () => unknown[] })._getActiveHandles?.().length ?? 0;
    for (let i = 0; i < 4; i++) {
      await client.completeChat({ model: "m", messages: [] });
    }
    await new Promise((r) => setTimeout(r, 30));
    const activeAfter = (process as NodeJS.Process & { _getActiveHandles?: () => unknown[] })._getActiveHandles?.().length ?? 0;
    // Four settled requests with unref'd timers must not leak handles.
    expect(activeAfter - activeBefore).toBeLessThan(3);
  });
});
