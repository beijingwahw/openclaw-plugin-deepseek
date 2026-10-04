import { afterEach, describe, expect, it } from "vitest";
import type { Server } from "node:http";
import { createServer } from "node:http";
import { DeepSeekClient } from "../src/deepseek-api.js";
import { runAgentLoop } from "../src/loop.js";
import { runUniversalCascade } from "../src/universal-cascade.js";
import { CostLedger } from "../src/ledger.js";
import { runJsonExtraction, validateAgainstSchema } from "../src/extract.js";
import { createDeepSeekHarness } from "../src/harness.js";
import type { DeepSeekRequester } from "../src/types.js";

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

function startSseServer(handler: (body: any) => string[]): Promise<string> {
  return new Promise((resolve) => {
    server = createServer((req, res) => {
      let data = "";
      req.on("data", (c) => (data += c));
      req.on("end", () => {
        const body = JSON.parse(data || "{}");
        res.writeHead(200, { "Content-Type": "text/event-stream" });
        res.end(handler(body).join(""));
      });
    });
    server.listen(0, "127.0.0.1", () => resolve(`http://127.0.0.1:${(server!.address() as any).port}`));
  });
}

function requesterAt(baseUrl: string, timeoutMs = 5000): DeepSeekRequester {
  const client = new DeepSeekClient({ baseUrl, apiKey: "sk", timeoutMs });
  return {
    streamChat: (body, signal) => client.streamChat(body, signal),
    completeChat: (body, signal) => client.completeChat(body, signal),
    getBalance: (signal) => client.getBalance(signal),
  };
}

describe("v1.1 module evolutions", () => {
  it("api client clears timeout timers after every request (no leaks)", async () => {
    const baseUrl = await new Promise<string>((resolve) => {
      server = createServer((req, res) => {
        let data = "";
        req.on("data", (c) => (data += c));
        req.on("end", () => {
          const body = JSON.parse(data || "{}");
          if (body.stream) {
            res.writeHead(200, { "Content-Type": "text/event-stream" });
            res.end([sseFrame({ choices: [{ delta: { content: "ok" } }] }), "data: [DONE]\n\n"].join(""));
          } else {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ choices: [{ message: { role: "assistant", content: "ok" } }] }));
          }
        });
      });
      server!.listen(0, "127.0.0.1", () => resolve(`http://127.0.0.1:${(server!.address() as any).port}`));
    });
    const activeBefore = (process as any)._getActiveHandles?.()?.length ?? 0;
    for (let i = 0; i < 5; i++) {
      await requesterAt(baseUrl).completeChat({ model: "m", messages: [{ role: "user", content: "hi" }] });
    }
    await new Promise((r) => setTimeout(r, 50));
    const activeAfter = (process as any)._getActiveHandles?.()?.length ?? 0;
    // 5 requests must not leave 5 dangling timers (server socket excluded).
    expect(activeAfter - activeBefore).toBeLessThan(3);
  });

  it("loop compacts old tool results when the context budget overflows", async () => {
    const big = "x".repeat(200_000);
    const requests: any[] = [];
    let toolRounds = 0;
    const baseUrl = await startSseServer((body) => {
      requests.push(body);
      const last = body.messages[body.messages.length - 1];
      const toolMessages = body.messages.filter((m: any) => m.role === "tool").length;
      if (last.role === "user" || (last.role === "tool" && toolMessages < 12)) {
        toolRounds += 1;
        return [
          sseFrame({ choices: [{ delta: { tool_calls: [{ index: 0, id: `c${toolRounds}`, function: { name: "fetch", arguments: JSON.stringify({ n: toolRounds }) } }] } }] }),
          sseFrame({ choices: [{ delta: {}, finish_reason: "tool_calls" }] }),
          "data: [DONE]\n\n",
        ];
      }
      return [
        sseFrame({ choices: [{ delta: { content: "final" } }] }),
        sseFrame({ choices: [{ delta: {}, finish_reason: "stop" }] }),
        "data: [DONE]\n\n",
      ];
    });
    const result = await runAgentLoop({
      requester: requesterAt(baseUrl),
      model: "m",
      systemPrompt: "s",
      prompt: "go",
      tools: [
        {
          schema: {
            type: "function",
            function: { name: "fetch", parameters: { type: "object", properties: { n: { type: "number" } }, required: ["n"] } },
          },
          execute: async (args) => `${big}#${args.n ?? 0}`,
        },
      ],
      maxSteps: 30,
      maxTokensPerStep: 100,
    });
    expect(result.answer).toBe("final");
    expect(result.stopReason).toBe("finished");
    expect((result.compactions ?? 0)).toBeGreaterThan(0);
    // Compacted messages present in later requests.
    const later = requests.find((r) => r.messages.some((m: any) => typeof m.content === "string" && m.content.includes("compacted")));
    expect(later).toBeTruthy();
  });

  it("loop breaks on identical repeated tool calls", async () => {
    const baseUrl = await startSseServer((body) => {
      const last = body.messages[body.messages.length - 1];
      if (last.role === "user" || last.role === "tool") {
        return [
          sseFrame({ choices: [{ delta: { tool_calls: [{ index: 0, id: `c${Date.now()}`, function: { name: "spin", arguments: "{}" } }] } }] }),
          sseFrame({ choices: [{ delta: {}, finish_reason: "tool_calls" }] }),
          "data: [DONE]\n\n",
        ];
      }
      return [sseFrame({ choices: [{ delta: { content: "never" } }] }), "data: [DONE]\n\n"];
    });
    const result = await runAgentLoop({
      requester: requesterAt(baseUrl),
      model: "m",
      systemPrompt: "s",
      prompt: "go",
      tools: [
        {
          schema: { type: "function", function: { name: "spin", parameters: { type: "object", properties: {} } } },
          execute: async () => "again",
        },
      ],
      maxSteps: 10,
      maxTokensPerStep: 50,
    });
    expect(result.stopReason).toBe("stepBudget");
    expect(result.iterations).toBeLessThan(6); // broke early, not at the cap
    expect(result.answer).toContain("repeated tool call spin");
  });

  it("cascade escalates on low judge confidence when a threshold is set", async () => {
    const executor = {
      label: "fake",
      async complete(params: any) {
        if (params.systemPrompt.includes("quality judge")) {
          return {
            // Grade says accept, but confidence is below the 0.8 threshold.
            text: '{"grade":"accept","confidence":0.55,"reason":"looks fine"}',
            usage: {},
          };
        }
        return { text: `answer-from-${params.ref}`, usage: {} };
      },
    };
    const result = await runUniversalCascade({
      resolveExecutor: () => executor as any,
      tiers: ["a/cheap", "b/strong"],
      judgeThreshold: 0.8,
      systemPrompt: "s",
      prompt: "p",
    });
    expect(result.escalated).toBe(true);
    expect(result.finalRef).toBe("b/strong");
    // The final tier short-circuits the judge; the escalation itself is the proof.
    expect(result.calls.map((c) => c.role)).toContain("escalated-draft");
  });

  it("ledger reports daily budget state", () => {
    let clock = Date.now();
    const ledger = new CostLedger({ now: () => clock });
    ledger.setDailyBudget(1.0);
    ledger.record({ provider: "deepseek", model: "deepseek-flash", ref: "deepseek/deepseek-flash", usage: { input: 1_000_000, output: 100_000 } });
    // ~$0.21 today, under budget
    let summary = ledger.summary();
    expect(summary.budget?.exceeded).toBe(false);
    expect(summary.budget?.percent).toBe(21);
    // Push past the budget.
    ledger.record({ provider: "deepseek", model: "deepseek-flash", ref: "deepseek/deepseek-flash", usage: { input: 10_000_000, output: 500_000 } });
    summary = ledger.summary();
    expect(summary.budget?.exceeded).toBe(true);
    expect(ledger.render()).toContain("EXCEEDED");
  });

  it("extract validates against the schema and repairs missing fields", async () => {
    let calls = 0;
    const baseUrl = await startSseServer(() => {
      calls += 1;
      if (calls === 1) {
        return [sseFrame({ choices: [{ delta: { content: '{"vendor":"Acme"}' } }] }), "data: [DONE]\n\n"];
      }
      return [sseFrame({ choices: [{ delta: { content: '{"vendor":"Acme","total":42}' } }] }), "data: [DONE]\n\n"];
    });
    const result = await runJsonExtraction({
      requester: requesterAt(baseUrl),
      prompt: "extract",
      schema: {
        type: "object",
        properties: { vendor: { type: "string" }, total: { type: "number" } },
        required: ["vendor", "total"],
      },
    });
    expect(result.repaired).toBe(true);
    expect(result.value).toEqual({ vendor: "Acme", total: 42 });
  });

  it("validateAgainstSchema reports type mismatches", () => {
    const issues = validateAgainstSchema(
      { vendor: 123, total: "42" },
      { type: "object", properties: { vendor: { type: "string" }, total: { type: "number" } }, required: ["vendor", "total"] },
    );
    expect(issues).toContain('field "vendor" should be string');
    expect(issues).toContain('field "total" should be number');
    expect(validateAgainstSchema({ vendor: "a", total: 1 }, { type: "object", properties: { vendor: { type: "string" }, total: { type: "number" } }, required: ["vendor", "total"] })).toEqual([]);
  });

  it("harness classifies empty visible results for fallback policy", async () => {
    const baseUrl = await startSseServer(() => [
      sseFrame({ choices: [{ delta: { reasoning_content: "deep thought but no text" } }] }),
      sseFrame({ choices: [{ delta: {}, finish_reason: "stop" }] }),
      "data: [DONE]\n\n",
    ]);
    const harness = createDeepSeekHarness({
      resolveApiKey: async () => "sk",
      getPluginConfig: () => ({ baseUrl }),
    });
    const result = await (harness as any).runAttempt({
      sessionId: "s",
      prompt: "q",
      workspaceDir: "/tmp",
      timeoutMs: 10_000,
    });
    expect(result.terminal.kind).toBe("ok");
    expect(result.agentHarnessResultClassification).toBe("reasoning-only");
  });
});
