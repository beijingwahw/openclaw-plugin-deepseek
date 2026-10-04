import { afterEach, describe, expect, it } from "vitest";
import type { Server } from "node:http";
import { createServer } from "node:http";
import { runCascade, parseJudgeLine, CASCADE_MODEL_ID } from "../src/cascade.js";
import { computeCost, isPeakNow, unitPriceTable } from "../src/cost.js";
import { runJsonExtraction } from "../src/extract.js";
import { runFimCompletion } from "../src/fim.js";
import { runAgentLoop } from "../src/loop.js";
import { DeepSeekClient } from "../src/deepseek-api.js";
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

function startSseServer(handler: (body: any, messages: any[]) => string[]): Promise<string> {
  return new Promise((resolve) => {
    server = createServer((req, res) => {
      let data = "";
      req.on("data", (c) => (data += c));
      req.on("end", () => {
        const body = JSON.parse(data || "{}");
        res.writeHead(200, { "Content-Type": "text/event-stream" });
        res.end(handler(body, body.messages ?? []).join(""));
      });
    });
    server.listen(0, "127.0.0.1", () => resolve(`http://127.0.0.1:${(server!.address() as any).port}`));
  });
}

function requesterAt(baseUrl: string): DeepSeekRequester {
  const client = new DeepSeekClient({ baseUrl, apiKey: "sk", timeoutMs: 5000 });
  return {
    streamChat: (body, signal) => client.streamChat(body, signal),
    completeChat: (body, signal) => client.completeChat(body, signal),
    getBalance: (signal) => client.getBalance(signal),
  };
}

describe("cascade", () => {
  it("accepts a good flash draft without escalating", async () => {
    const calls: any[] = [];
    const baseUrl = await startSseServer((body) => {
      calls.push(body);
      const system = body.messages?.[0]?.content ?? "";
      if (system.includes("quality judge")) {
        return [
          sseFrame({ choices: [{ delta: { content: '{"grade":"accept","confidence":0.93,"reason":"solid"}' } }] }),
          sseFrame({ choices: [{ delta: {}, finish_reason: "stop" }] }),
          "data: [DONE]\n\n",
        ];
      }
      return [
        sseFrame({ choices: [{ delta: { content: "Paris is the capital of France." } }] }),
        sseFrame({ choices: [{ delta: {}, finish_reason: "stop" }] }),
        "data: [DONE]\n\n",
      ];
    });

    const events: string[] = [];
    const result = await runCascade({
      requester: requesterAt(baseUrl),
      systemPrompt: "assistant",
      prompt: "capital of France?",
      onEvent: (e) => events.push(e.type),
    });

    expect(result.escalated).toBe(false);
    expect(result.finalModel).toBe("deepseek-flash");
    expect(result.text).toContain("Paris");
    expect(result.judge.grade).toBe("accept");
    // Draft at low effort, judge at none.
    const draft = calls.find((c) => c.reasoning_effort === "low");
    expect(draft?.model).toBe("deepseek-flash");
    expect(calls.some((c) => c.thinking?.type === "disabled")).toBe(true);
    expect(events).toContain("judge");
    expect(events).toContain("final");
  });

  it("escalates to v4-pro at max effort when the judge rejects", async () => {
    const calls: any[] = [];
    const baseUrl = await startSseServer((body) => {
      calls.push(body);
      const system = body.messages?.[0]?.content ?? "";
      if (system.includes("quality judge")) {
        return [
          sseFrame({ choices: [{ delta: { content: '{"grade":"escalate","confidence":0.2,"reason":"vague"}' } }] }),
          sseFrame({ choices: [{ delta: {}, finish_reason: "stop" }] }),
          "data: [DONE]\n\n",
        ];
      }
      if (body.model === "deepseek-v4-pro") {
        return [
          sseFrame({ choices: [{ delta: { reasoning_content: "deep thought" } }] }),
          sseFrame({ choices: [{ delta: { content: "The precise answer is 42 with full derivation." } }] }),
          sseFrame({ choices: [{ delta: {}, finish_reason: "stop" }] }),
          "data: [DONE]\n\n",
        ];
      }
      return [
        sseFrame({ choices: [{ delta: { content: "uhhh maybe 42?" } }] }),
        sseFrame({ choices: [{ delta: {}, finish_reason: "stop" }] }),
        "data: [DONE]\n\n",
      ];
    });

    const result = await runCascade({
      requester: requesterAt(baseUrl),
      systemPrompt: "assistant",
      prompt: "derive the answer",
    });

    expect(result.escalated).toBe(true);
    expect(result.finalModel).toBe("deepseek-v4-pro");
    expect(result.text).toContain("precise answer");
    const escalated = calls.find((c) => c.model === "deepseek-v4-pro" && c.reasoning_effort === "max");
    expect(escalated).toBeTruthy();
    expect(CASCADE_MODEL_ID).toBe("deepseek-cascade");
  });

  it("parses judge output defensively", () => {
    expect(parseJudgeLine('{"grade":"escalate","confidence":0.1,"reason":"bad"}').grade).toBe("escalate");
    expect(parseJudgeLine("no json at all, but long enough to look real").grade).toBe("accept");
    expect(parseJudgeLine("").grade).toBe("escalate");
    expect(parseJudgeLine('garbage {"grade":"accept","confidence":"x"}').grade).toBe("accept");
  });
});

describe("cost metering", () => {
  it("computes cache-aware costs with peak doubling", () => {
    const usage = {
      prompt_tokens: 1_000_000,
      prompt_cache_hit_tokens: 900_000,
      prompt_cache_miss_tokens: 100_000,
      completion_tokens: 50_000,
    };
    const offPeak = computeCost("deepseek-flash", usage, { peak: false });
    // 0.1M miss * $0.15 + 0.9M hit * $0.003 + 0.05M out * $0.6
    expect(offPeak.input).toBeCloseTo(0.015, 4);
    expect(offPeak.cacheRead).toBeCloseTo(0.0027, 4);
    expect(offPeak.output).toBeCloseTo(0.03, 4);
    expect(offPeak.total).toBeCloseTo(0.0477, 3);

    const peak = computeCost("deepseek-flash", usage, { peak: true });
    expect(peak.total).toBeCloseTo(offPeak.total * 2, 3);
  });

  it("prices cascade at the flash tier", () => {
    const cost = computeCost("deepseek-cascade", {
      prompt_tokens: 1000,
      prompt_cache_miss_tokens: 1000,
      completion_tokens: 1000,
    });
    expect(cost.total).toBeGreaterThan(0);
  });

  it("classifies peak windows by UTC weekday/hour", () => {
    // Beijing 09:00-12:00 & 14:00-18:00 == UTC 01:00-04:00 & 06:00-10:00 weekdays.
    expect(isPeakNow(new Date("2026-10-05T02:00:00Z"))).toBe(true); // Monday UTC 02:00
    expect(isPeakNow(new Date("2026-10-05T05:00:00Z"))).toBe(false); // Monday UTC 05:00 gap
    expect(isPeakNow(new Date("2026-10-05T12:00:00Z"))).toBe(false); // Monday UTC noon
    expect(isPeakNow(new Date("2026-10-04T02:00:00Z"))).toBe(false); // Sunday
  });

  it("renders a unit price table for both windows", () => {
    expect(unitPriceTable(false)).toContain("deepseek-flash");
    expect(unitPriceTable(true)).toContain("deepseek-v4-pro");
  });
});

describe("structured extraction", () => {
  it("returns parsed JSON and repairs one invalid pass", async () => {
    let calls = 0;
    const baseUrl = await startSseServer((body) => {
      calls += 1;
      if (calls === 1) {
        return [
          sseFrame({ choices: [{ delta: { content: "```json\n{\"vendor\": \"Acme\", \"total\":" } }] }),
          sseFrame({ choices: [{ delta: {}, finish_reason: "length" }] }),
          "data: [DONE]\n\n",
        ];
      }
      return [
        sseFrame({ choices: [{ delta: { content: '{"vendor":"Acme","total":99.5}' } }] }),
        sseFrame({ choices: [{ delta: {}, finish_reason: "stop" }] }),
        "data: [DONE]\n\n",
      ];
    });

    const result = await runJsonExtraction({
      requester: requesterAt(baseUrl),
      prompt: "extract from this invoice",
      schema: { type: "object", properties: { vendor: { type: "string" }, total: { type: "number" } } },
    });
    expect(result.repaired).toBe(true);
    expect(result.value).toEqual({ vendor: "Acme", total: 99.5 });
  });

  it("sends response_format json_object with explicit JSON instruction", async () => {
    const bodies: any[] = [];
    const baseUrl = await startSseServer((body) => {
      bodies.push(body);
      return [
        sseFrame({ choices: [{ delta: { content: '{"ok":true}' } }] }),
        sseFrame({ choices: [{ delta: {}, finish_reason: "stop" }] }),
        "data: [DONE]\n\n",
      ];
    });
    await runJsonExtraction({
      requester: requesterAt(baseUrl),
      prompt: "extract",
      schema: { type: "object", properties: { ok: { type: "boolean" } } },
    });
    const body = bodies[0];
    expect(body.response_format).toEqual({ type: "json_object" });
    expect(body.thinking).toEqual({ type: "disabled" });
    expect(body.messages[1].content).toContain("Return ONLY a JSON object");
  });
});

describe("FIM completion", () => {
  it("sends prompt/suffix on the beta surface", async () => {
    const bodies: any[] = [];
    const baseUrl = await startSseServer((body) => {
      bodies.push(body);
      return [
        sseFrame({ choices: [{ delta: { content: "n = n + 1" } }] }),
        sseFrame({ choices: [{ delta: {}, finish_reason: "stop" }] }),
        "data: [DONE]\n\n",
      ];
    });
    const result = await runFimCompletion({
      requester: requesterAt(baseUrl),
      prompt: "def inc(n):\n    ",
      suffix: "\n    return n",
    });
    expect(result.completion).toBe("n = n + 1");
    expect(bodies[0].prompt).toContain("def inc(n):");
    expect(bodies[0].suffix).toBe("\n    return n");
    expect(bodies[0].model).toBe("deepseek-flash");
  });
});

describe("parallel tool execution", () => {
  it("runs parallel tools concurrently and keeps result order", async () => {
    const active: number[] = [];
    let maxConcurrent = 0;
    const baseUrl = await startSseServer((body) => {
      const last = body.messages[body.messages.length - 1];
      if (last.role === "user") {
        return [
          sseFrame({
            choices: [
              {
                delta: {
                  tool_calls: [
                    { index: 0, id: "c1", function: { name: "slow_a", arguments: "{}" } },
                    { index: 1, id: "c2", function: { name: "slow_b", arguments: "{}" } },
                    { index: 2, id: "c3", function: { name: "slow_c", arguments: "{}" } },
                  ],
                },
              },
            ],
          }),
          sseFrame({ choices: [{ delta: {}, finish_reason: "tool_calls" }] }),
          "data: [DONE]\n\n",
        ];
      }
      return [
        sseFrame({ choices: [{ delta: { content: "all done" } }] }),
        sseFrame({ choices: [{ delta: {}, finish_reason: "stop" }] }),
        "data: [DONE]\n\n",
      ];
    });

    const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));
    const mk = (name: string, ms: number, mode?: "sequential" | "parallel") => ({
      schema: { type: "function" as const, function: { name, parameters: { type: "object", properties: {} } } },
      executionMode: mode,
      execute: async () => {
        active.push(1);
        maxConcurrent = Math.max(maxConcurrent, active.length);
        await delay(ms);
        active.pop();
        return `${name}-result`;
      },
    });

    const result = await runAgentLoop({
      requester: requesterAt(baseUrl),
      model: "deepseek-flash",
      systemPrompt: "s",
      prompt: "go",
      tools: [mk("slow_a", 60), mk("slow_b", 60), mk("slow_c", 60)],
      maxSteps: 2,
      maxTokensPerStep: 100,
    });

    expect(result.answer).toBe("all done");
    expect(maxConcurrent).toBeGreaterThan(1); // genuinely parallel
  });

  it("isolates sequential tools from overlapping", async () => {
    const baseUrl = await startSseServer((body) => {
      const last = body.messages[body.messages.length - 1];
      if (last.role === "user") {
        return [
          sseFrame({
            choices: [
              {
                delta: {
                  tool_calls: [
                    { index: 0, id: "c1", function: { name: "seq_a", arguments: "{}" } },
                    { index: 1, id: "c2", function: { name: "seq_b", arguments: "{}" } },
                  ],
                },
              },
            ],
          }),
          sseFrame({ choices: [{ delta: {}, finish_reason: "tool_calls" }] }),
          "data: [DONE]\n\n",
        ];
      }
      return [
        sseFrame({ choices: [{ delta: { content: "done" } }] }),
        sseFrame({ choices: [{ delta: {}, finish_reason: "stop" }] }),
        "data: [DONE]\n\n",
      ];
    });

    const active: number[] = [];
    let maxConcurrent = 0;
    const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));
    const result = await runAgentLoop({
      requester: requesterAt(baseUrl),
      model: "deepseek-flash",
      systemPrompt: "s",
      prompt: "go",
      tools: [
        {
          schema: { type: "function", function: { name: "seq_a", parameters: { type: "object", properties: {} } } },
          executionMode: "sequential" as const,
          execute: async () => {
            active.push(1);
            maxConcurrent = Math.max(maxConcurrent, active.length);
            await delay(50);
            active.pop();
            return "a";
          },
        },
        {
          schema: { type: "function", function: { name: "seq_b", parameters: { type: "object", properties: {} } } },
          executionMode: "sequential" as const,
          execute: async () => {
            active.push(1);
            maxConcurrent = Math.max(maxConcurrent, active.length);
            await delay(50);
            active.pop();
            return "b";
          },
        },
      ],
      maxSteps: 2,
      maxTokensPerStep: 100,
    });
    expect(result.answer).toBe("done");
    expect(maxConcurrent).toBe(1); // never overlapped
  });
});
