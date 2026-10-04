import { afterEach, describe, expect, it } from "vitest";
import type { Server } from "node:http";
import { createServer } from "node:http";
import plugin from "../dist/index.js";

let server: Server | undefined;

afterEach(async () => {
  if (server) {
    await new Promise<void>((resolve) => server!.close(() => resolve()));
    server = undefined;
  }
});

function sseFrame(payload: unknown): string {
  return `data: ${JSON.stringify(payload)}\n\n`;
}

/**
 * Mock DeepSeek API. Streams real SSE for stream:true requests and branches
 * on the last message role: user -> tool call, tool -> final answer.
 */
function startMockDeepSeek(): Promise<string> {
  return new Promise((resolve) => {
    server = createServer((req, res) => {
      let data = "";
      req.on("data", (c) => (data += c));
      req.on("end", () => {
        const body = JSON.parse(data || "{}");
        const last = body.messages[body.messages.length - 1];
        let frames: string[];
        if (last.role === "user") {
          frames = [
            sseFrame({ choices: [{ delta: { reasoning_content: "user wants capital of France" } }] }),
            sseFrame({ choices: [{ delta: { tool_calls: [{ index: 0, id: "call_x", function: { name: "lookup_city", arguments: '{"country":"France"}' } }] } }] }),
            sseFrame({ choices: [{ delta: {}, finish_reason: "tool_calls" }] }),
            "data: [DONE]\n\n",
          ];
        } else {
          frames = [
            sseFrame({ choices: [{ delta: { content: "The capital is Paris." } }] }),
            sseFrame({ choices: [{ delta: {}, finish_reason: "stop" }] }),
            sseFrame({ choices: [], usage: { prompt_tokens: 42, completion_tokens: 7, total_tokens: 49 } }),
            "data: [DONE]\n\n",
          ];
        }
        if (body.stream) {
          res.writeHead(200, { "Content-Type": "text/event-stream" });
          res.end(frames.join(""));
        } else {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ choices: [{ message: { role: "assistant", content: "ok" } }] }));
        }
      });
    });
    server!.listen(0, "127.0.0.1", () => {
      resolve(`http://127.0.0.1:${(server!.address() as { port: number }).port}`);
    });
  });
}

describe("plugin registration end-to-end", () => {
  it("registers provider, harness, and all three tools", () => {
    const registered = { providers: [], harnesses: [], tools: [] };
    const api = makeApi(registered, {});
    plugin.register(api);
    expect(registered.providers.map((p: any) => p.id)).toContain("openclaw-deepseek");
    expect(registered.harnesses.map((h: any) => h.id)).toContain("openclaw-deepseek");
    expect(registered.tools.map((t: any) => t.name).sort()).toEqual([
      "deepseek_balance",
      "deepseek_cascade",
      "deepseek_cost_report",
      "deepseek_extract",
      "deepseek_fim",
      "deepseek_genesis",
      "deepseek_reason",
      "deepseek_research",
    ]);
  });

  it("registered harness runs a full turn against a mock DeepSeek API", async () => {
    const baseUrl = await startMockDeepSeek();
    const registered = { providers: [], harnesses: [], tools: [] };
    const api = makeApi(registered, { baseUrl });
    plugin.register(api);
    const harness = registered.harnesses[0];

    expect(harness.supports({ provider: "openclaw-deepseek" })).toMatchObject({ supported: true });
    expect(harness.supports({ provider: "deepseek" }).supported).toBe(false);

    const deltas: string[] = [];
    const partials: string[] = [];
    const result = await harness.runAttempt({
      sessionId: "smoke-1",
      prompt: "What is the capital of France?",
      workspaceDir: "/tmp",
      timeoutMs: 10_000,
      tools: [
        {
          name: "lookup_city",
          description: "Look up a city fact",
          parameters: { type: "object", properties: { country: { type: "string" } } },
          execute: async (_id, params) => ({
            content: [{ type: "text", text: `capital of ${params.country} is Paris` }],
          }),
        },
      ],
      onPartialReply: (payload: { text?: string; delta?: string }) => {
        if (payload.delta) deltas.push(payload.delta);
        if (payload.text) partials.push(payload.text);
      },
    });

    expect(result.terminal.kind).toBe("ok");
    expect(result.assistantTexts[0]).toBe("The capital is Paris.");
    expect(result.lastAssistant.provider).toBe("openclaw-deepseek");
    expect(result.lastAssistant.model).toBe("deepseek-flash");
    expect(result.lastAssistant.usage.input).toBe(42);
    expect(result.modelIterations).toBe(2);
    // v1.0: harness streams true deltas (no full-text re-sends).
    expect(partials).toEqual([]);
    expect(deltas).toContain("The capital is Paris.");
  });

  it("deepseek_research tool runs its loop against a mock API", async () => {
    const baseUrl = await new Promise<string>((resolve) => {
      server = createServer((req, res) => {
        let data = "";
        req.on("data", (c) => (data += c));
        req.on("end", () => {
          const body = JSON.parse(data || "{}");
          const last = body.messages[body.messages.length - 1];
          let frames: string[];
          if (last.role === "user") {
            frames = [
              sseFrame({ choices: [{ delta: { tool_calls: [{ index: 0, id: "c1", function: { name: "web_search", arguments: '{"query":"deepseek v4 release"}' } }] } }] }),
              sseFrame({ choices: [{ delta: {}, finish_reason: "tool_calls" }] }),
              "data: [DONE]\n\n",
            ];
          } else {
            frames = [
              sseFrame({ choices: [{ delta: { content: "DeepSeek V4 is released (mock findings)." } }] }),
              sseFrame({ choices: [{ delta: {}, finish_reason: "stop" }] }),
              "data: [DONE]\n\n",
            ];
          }
          if (body.stream) {
            res.writeHead(200, { "Content-Type": "text/event-stream" });
            res.end(frames.join(""));
          } else {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({}));
          }
        });
      });
      server.listen(0, "127.0.0.1", () => {
        resolve(`http://127.0.0.1:${(server!.address() as { port: number }).port}`);
      });
    });

    const registered = { providers: [], harnesses: [], tools: [] };
    const api = makeApi(registered, { baseUrl });
    plugin.register(api);
    const research = registered.tools.find((t: any) => t.name === "deepseek_research");

    const result = await research.execute("test-call", { question: "DeepSeek V4 latest news", max_searches: 2 });
    const text = result.content.map((c: any) => c.text).join("");
    expect(text).toContain("DeepSeek V4 is released");
    expect(result.details.searchesUsed).toBe(1);
    expect(result.details.stopReason).toBe("finished");
  });

  it("cascade tool routes cross-provider tiers through the host llm runtime", async () => {
    const baseUrl = await startMockDeepSeek();
    const registered = { providers: [], harnesses: [], tools: [] };
    const api = makeApi(registered, { baseUrl });
    plugin.register(api);
    const cascade = registered.tools.find((t: any) => t.name === "deepseek_cascade");

    const result = await cascade.execute("t", {
      task: "explain quantile regression",
      tiers: ["deepseek-flash", "zai/glm-5.3"],
      judge: "zai/glm-5.3",
    });
    const text = result.content.map((c: any) => c.text).join("");
    // Draft from native DeepSeek got escalated by the host judge, final answer
    // comes from the host-routed tier.
    expect(text).toContain("host-routed cross-provider answer");
    expect(result.details.escalated).toBe(true);
    expect(result.details.finalRef).toBe("zai/glm-5.3");
    expect(result.details.calls.map((c: any) => c.role)).toEqual(["draft", "judge", "escalated-draft"]);
  });

  it("llm_output hook feeds the live cost ledger (all providers)", async () => {
    const registered = { providers: [], harnesses: [], tools: [] } as any;
    const api = makeApi(registered, {});
    plugin.register(api);

    // The plugin subscribed to llm_output; simulate gateway-wide model calls.
    const handler = registered.hooks?.["llm_output"];
    expect(typeof handler).toBe("function");
    handler({
      provider: "deepseek",
      model: "deepseek-flash",
      resolvedRef: "deepseek/deepseek-flash",
      sessionKey: "agent:main:im:1",
      usage: { input: 1_000_000, output: 100_000, cacheRead: 800_000 },
    });
    handler({
      provider: "anthropic",
      model: "claude-sonnet-4-6",
      resolvedRef: "anthropic/claude-sonnet-4-6",
      sessionKey: "agent:main:im:2",
      usage: { input: 200_000, output: 50_000 },
    });

    const report = registered.tools.find((t: any) => t.name === "deepseek_cost_report");
    const result = await report.execute("t", { view: "live" });
    const text = result.content.map((c: any) => c.text).join("");
    expect(text).toContain("Live model spend");
    expect(text).toContain("deepseek/deepseek-flash");
    expect(text).toContain("anthropic/claude-sonnet-4-6");
    // flash: 0.2M miss*0.15 + 0.8M hit*0.003 + 0.1M*0.6 = 0.0924
    // claude unpriced (catalog not fetched in test env) -> unpriced calls counted
    expect(result.details.summary.calls).toBe(2);
    expect(result.details.summary.byModel).toHaveLength(2);
    expect(result.details.summary.unpricedCalls).toBe(1);
    expect(result.details.summary.byModel[0].ref).toBe("deepseek/deepseek-flash");

    // Session-scoped view filters correctly.
    const scoped = await report.execute("t", { view: "live", session: "agent:main:im:1" });
    expect(scoped.details.summary.calls).toBe(1);
  });

  it("tools fail closed without DEEPSEEK_API_KEY", async () => {
    const registered = { providers: [], harnesses: [], tools: [] };
    const api = makeApi(registered, { noKey: true, baseUrl: "http://127.0.0.1:9" });
    plugin.register(api);
    const balance = registered.tools.find((t: any) => t.name === "deepseek_balance");
    const result = await balance.execute("t", {});
    expect(result.content[0].text).toContain("DEEPSEEK_API_KEY is not configured");
  });
});

function makeApi(
  registered: { providers: any[]; harnesses: any[]; tools: any[] },
  opts: { noKey?: boolean; baseUrl?: string } = {},
) {
  const apiKey = opts.noKey ? null : "sk-test-key";
  return {
    id: "openclaw-deepseek",
    pluginConfig: opts.baseUrl ? { baseUrl: opts.baseUrl } : {},
    registerProvider: (p: any) => registered.providers.push(p),
    registerAgentHarness: (h: any) => registered.harnesses.push(h),
    registerTool: (t: any) => registered.tools.push(t),
    on: (events: string, handler: (event: unknown) => void) => {
      (registered as any).hooks = (registered as any).hooks ?? {};
      (registered as any).hooks[events] = handler;
    },
    runtime: {
      modelAuth: {
        resolveApiKeyForProvider: async () => apiKey,
      },
      webSearch: {
        search: async () => ({
          content: [{ type: "text", text: "mock search results about deepseek" }],
        }),
      },
      llm: {
        complete: async (params: any) => ({
          text: params.systemPrompt?.includes("quality judge")
            ? '{"grade":"escalate","confidence":0.2,"reason":"vague"}'
            : "host-routed cross-provider answer",
          provider: "zai",
          model: params.model ?? "glm-5.3",
          usage: { inputTokens: 42, outputTokens: 7, costUsd: 0.002 },
        }),
      },
    },
  };
}
