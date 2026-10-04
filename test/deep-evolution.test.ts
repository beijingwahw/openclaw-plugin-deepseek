import { afterEach, describe, expect, it } from "vitest";
import type { Server } from "node:http";
import { createServer } from "node:http";
import {
  DeepSeekAnthropicClient,
  ANTHROPIC_BASE_URL,
  mapClaudeRefToDeepSeek,
} from "../src/anthropic.js";
import { rebuildHistoryFromJsonl } from "../src/memory.js";
import { CostLedger } from "../src/ledger.js";
import { createAnthropicProvider } from "../src/provider.js";
import plugin from "../dist/index.js";

let server: Server | undefined;

afterEach(async () => {
  if (server) {
    await new Promise<void>((resolve) => server!.close(() => resolve()));
    server = undefined;
  }
});

describe("claude ref mapping", () => {
  it("maps opus to v4-pro and everything else to flash", () => {
    expect(mapClaudeRefToDeepSeek("claude-opus-4-8")).toBe("deepseek-v4-pro");
    expect(mapClaudeRefToDeepSeek("claude-sonnet-4-6")).toBe("deepseek-flash");
    expect(mapClaudeRefToDeepSeek("claude-haiku-4-5")).toBe("deepseek-flash");
    expect(mapClaudeRefToDeepSeek("claude-anything-else")).toBe("deepseek-flash");
    expect(ANTHROPIC_BASE_URL).toBe("https://api.deepseek.com/anthropic");
  });
});

describe("anthropic transport", () => {
  it("streams messages with text, thinking, and usage events", async () => {
    const requests: any[] = [];
    const baseUrl = await new Promise<string>((resolve) => {
      server = createServer((req, res) => {
        let data = "";
        req.on("data", (c) => (data += c));
        req.on("end", () => {
          requests.push({ url: req.url, headers: req.headers, body: JSON.parse(data || "{}") });
          res.writeHead(200, { "Content-Type": "text/event-stream" });
          res.end(
            [
              `data: ${JSON.stringify({ type: "message_start", message: { usage: { input_tokens: 100, output_tokens: 0 } } })}\n\n`,
              `data: ${JSON.stringify({ type: "content_block_delta", delta: { thinking: "pondering" } })}\n\n`,
              `data: ${JSON.stringify({ type: "content_block_delta", delta: { text: "Hello" } })}\n\n`,
              `data: ${JSON.stringify({ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 5 } })}\n\n`,
              `data: ${JSON.stringify({ type: "message_stop" })}\n\n`,
            ].join(""),
          );
        });
      });
      server.listen(0, "127.0.0.1", () => resolve(`http://127.0.0.1:${(server!.address() as any).port}`));
    });

    const client = new DeepSeekAnthropicClient({ baseUrl, apiKey: "sk", timeoutMs: 5000 });
    const events = [];
    for await (const event of client.streamMessage({
      model: "claude-sonnet-4-6",
      system: "be brief",
      messages: [{ role: "user", content: "hi" }],
    })) {
      events.push(event);
    }

    // Request shape: Anthropic protocol over the DeepSeek endpoint.
    expect(requests[0].url).toBe("/v1/messages");
    expect(requests[0].headers["x-api-key"]).toBe("sk");
    expect(requests[0].headers["anthropic-version"]).toBe("2023-06-01");
    expect(requests[0].body.model).toBe("claude-sonnet-4-6");
    expect(requests[0].body.system).toBe("be brief");
    expect(requests[0].body.messages[0].content).toBe("hi");

    // Stream events: thinking + text + usage merged + finish.
    expect(events).toContainEqual({ type: "thinking", text: "pondering" });
    expect(events).toContainEqual({ type: "text", text: "Hello" });
    const usage = events.filter((e) => e.type === "usage");
    expect(usage.length).toBeGreaterThanOrEqual(1);
    expect(usage[0]?.usage?.prompt_tokens).toBe(100); // message_start input usage
    const finish = events.find((e) => e.type === "finish");
    expect(finish?.finishReason).toBe("end_turn");
  });

  it("maps and registers the anthropic provider", async () => {
    const provider = createAnthropicProvider({
      getPluginConfig: () => undefined,
      resolveApiKey: async () => "sk",
    });
    expect(provider.id).toBe("openclaw-deepseek-anthropic");
    const model = (provider as any).resolveDynamicModel({ modelId: "claude-opus-4-8" });
    expect(model.api).toBe("anthropic-messages");
    expect(model.baseUrl).toBe("https://api.deepseek.com/anthropic");
    expect(model.name).toContain("deepseek-v4-pro");

    const catalog = await (provider as any).catalog.run({
      resolveProviderApiKey: (id: string) => ({ apiKey: id === "openclaw-deepseek-anthropic" ? "sk" : undefined }),
    });
    expect(catalog.provider.api).toBe("anthropic-messages");
    expect(catalog.provider.models.map((m: any) => m.id)).toContain("claude-sonnet-4-6");
  });
});

describe("harness multi-turn memory", () => {
  it("rebuilds DeepSeek history from an OpenClaw transcript", () => {
    const lines = [
      JSON.stringify({ role: "user", content: "What is 2+2?" }),
      JSON.stringify({ role: "assistant", content: [{ type: "thinking", thinking: "simple math" }, { type: "toolCall", id: "c1", name: "calc", arguments: { expr: "2+2" } }] }),
      JSON.stringify({ role: "toolResult", toolCallId: "c1", toolName: "calc", content: [{ type: "text", text: "4" }] }),
      JSON.stringify({ role: "assistant", content: [{ type: "text", text: "2+2 = 4." }] }),
      JSON.stringify({ role: "user", content: "And 3+3?" }),
      JSON.stringify({ role: "assistant", content: [{ type: "text", text: "6." }] }),
      JSON.stringify({ role: "systemNote", content: "noise" }),
    ];
    const history = rebuildHistoryFromJsonl(lines.join("\n"), { maxTurns: 12 });
    expect(history.linesUsed).toBe(6);
    expect(history.linesSkipped).toBe(1);

    const roles = history.messages.map((m) => m.role);
    expect(roles).toEqual(["user", "assistant", "tool", "assistant", "user", "assistant"]);

    // reasoning_content replay contract on tool-call turns.
    const assistantToolTurn = history.messages.find((m) => m.role === "assistant" && m.tool_calls);
    expect(assistantToolTurn?.reasoning_content).toBe("simple math");
    expect(assistantToolTurn?.tool_calls?.[0]?.function.name).toBe("calc");

    const toolResult = history.messages.find((m) => m.role === "tool");
    expect(toolResult?.content).toBe("4");
  });

  it("keeps only the trailing window of exchanges", () => {
    const lines: string[] = [];
    for (let i = 0; i < 20; i++) {
      lines.push(JSON.stringify({ role: "user", content: `q${i}` }));
      lines.push(JSON.stringify({ role: "assistant", content: [{ type: "text", text: `a${i}` }] }));
    }
    const history = rebuildHistoryFromJsonl(lines.join("\n"), { maxTurns: 3 });
    const userTurns = history.messages.filter((m) => m.role === "user");
    expect(userTurns).toHaveLength(3);
    expect(userTurns[0]?.content).toBe("q17"); // last 3 exchanges
  });
});

describe("burn-rate forecast", () => {
  it("projects end-of-day spend and the red-threshold crossing time", () => {
    let clock = Date.parse("2026-10-05T02:00:00Z");
    const ledger = new CostLedger({ now: () => clock });
    ledger.setDailyBudget(1.0);
    // Two calls an hour apart: $0.10 at 02:00, $0.20 at 03:00 -> $0.10/h.
    ledger.record({ provider: "p", model: "deepseek-flash", ref: "deepseek/deepseek-flash", usage: { input: 600_000, output: 50_000 } });
    clock += 3_600_000;
    ledger.record({ provider: "p", model: "deepseek-flash", ref: "deepseek/deepseek-flash", usage: { input: 1_200_000, output: 100_000 } });

    const summary = ledger.summary();
    expect(summary.forecast).toBeDefined();
    const f = summary.forecast!;
    expect(f.spentHours).toBeCloseTo(1.0, 1);
    // call 1: 0.6M*0.15 + 0.05M*0.6 = $0.12; call 2: $0.24 -> $0.36 total over 1h.
    expect(f.hourlyUsd).toBeCloseTo(0.36, 2);
    expect(f.projectedEodUsd).toBeGreaterThan(0.36); // 21h left at $0.36/h
    // Red threshold $0.90: (0.9-0.36)/0.36 = 1.5h from 03:00 -> ~04:30 UTC.
    expect(f.redAtUtc).toBe("04:30");
    expect(ledger.render()).toContain("Burn rate");
  });
});

describe("plugin registration with dual providers", () => {
  it("registers both transports", () => {
    const registered: any = { providers: [], harnesses: [], tools: [] };
    const api = {
      id: "openclaw-deepseek",
      pluginConfig: {},
      registerProvider: (p: any) => registered.providers.push(p),
      registerAgentHarness: (h: any) => registered.harnesses.push(h),
      registerTool: (t: any) => registered.tools.push(t),
      runtime: {
        modelAuth: { resolveApiKeyForProvider: async () => "sk" },
        webSearch: { search: async () => ({ content: [] }) },
        llm: { complete: async () => ({ text: "x", provider: "zai", model: "glm" }) },
      },
    };
    plugin.register(api);
    const ids = registered.providers.map((p: any) => p.id).sort();
    expect(ids).toEqual(["openclaw-deepseek", "openclaw-deepseek-anthropic"]);
  });
});
