import { afterEach, describe, expect, it } from "vitest";
import type { Server } from "node:http";
import { createServer } from "node:http";
import { runAgentLoop } from "../src/loop.js";
import type { DeepSeekRequester, DeepSeekStreamEvent } from "../src/types.js";

let server: Server | undefined;

afterEach(async () => {
  if (server) {
    await new Promise<void>((resolve) => server!.close(() => resolve()));
    server = undefined;
  }
});

function startMockServer(handler: (url: string, body: any) => any): Promise<string> {
  return new Promise((resolve) => {
    server = createServer((req, res) => {
      let data = "";
      req.on("data", (chunk) => (data += chunk));
      req.on("end", () => {
        const url = req.url ?? "";
        let parsedBody: any;
        try {
          parsedBody = JSON.parse(data || "{}");
        } catch {
          parsedBody = {};
        }
        const result = handler(url, parsedBody);
        if (result === null) {
          res.writeHead(404, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "not found" }));
          return;
        }
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(result));
      });
    });
    server.listen(0, "127.0.0.1", () => {
      const address = server!.address() as { port: number };
      resolve(`http://127.0.0.1:${address.port}`);
    });
  });
}

function sseFrame(payload: unknown): string {
  return `data: ${JSON.stringify(payload)}\n\n`;
}

describe("runAgentLoop", () => {
  it("runs a full tool-call round trip and returns the final answer", async () => {
    const baseUrl = await startMockServer((url, body) => {
      if (!url.includes("/chat/completions")) return null;
      const messages = body.messages as Array<Record<string, any>>;
      const last = messages[messages.length - 1];
      // First call: assistant asks for a tool. After tool result: final answer.
      if (last.role === "user") {
        return {
          __sse: [
            sseFrame({ choices: [{ delta: { reasoning_content: "planning" } }] }),
            sseFrame({ choices: [{ delta: { tool_calls: [{ index: 0, id: "c1", function: { name: "web_search", arguments: '{"query":"x"}' } }] } }] }),
            sseFrame({ choices: [{ delta: {}, finish_reason: "tool_calls" }] }),
            "data: [DONE]\n\n",
          ],
        };
      }
      if (last.role === "tool") {
        return {
          __sse: [
            sseFrame({ choices: [{ delta: { content: "The answer is 42." } }] }),
            sseFrame({ choices: [{ delta: {}, finish_reason: "stop" }] }),
            sseFrame({ choices: [], usage: { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120 } }),
            "data: [DONE]\n\n",
          ],
        };
      }
      return { __sse: ["data: [DONE]\n\n"] };
    });

    const requester: DeepSeekRequester = {
      async *streamChat(body) {
        const response = await fetch(`${baseUrl}/chat/completions`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const payload = (await response.json()) as { __sse: string[] };
        const encoder = new TextEncoder();
        const stream = new ReadableStream<Uint8Array>({
          start(controller) {
            for (const chunk of payload.__sse) controller.enqueue(encoder.encode(chunk));
            controller.close();
          },
        });
        // Reuse the production SSE parser for fidelity.
        const { parseSseStream } = await import("../src/deepseek-api.js");
        yield* parseSseStream(stream);
      },
      async completeChat() {
        throw new Error("not used");
      },
      async getBalance() {
        throw new Error("not used");
      },
    };

    const events: string[] = [];
    const result = await runAgentLoop({
      requester,
      model: "deepseek-chat",
      systemPrompt: "test",
      prompt: "meaning of life",
      tools: [
        {
          schema: { type: "function", function: { name: "web_search", parameters: { type: "object", properties: { query: { type: "string" } }, required: ["query"] } } },
          execute: async (args) => `results for ${String(args.query)}`,
        },
      ],
      maxSteps: 4,
      maxTokensPerStep: 512,
      onEvent: (e) => events.push(e.type),
    });

    expect(result.stopReason).toBe("finished");
    expect(result.answer).toBe("The answer is 42.");
    expect(result.iterations).toBe(2);
    expect(result.usage.prompt_tokens).toBe(100);
    expect(result.reasoning).toBe("planning");
    expect(result.steps[0]?.toolCalls?.[0]?.name).toBe("web_search");
    expect(events).toContain("toolCall");
    expect(events).toContain("toolResult");
  });

  it("stops at the step budget without tool calls completing", async () => {
    const baseUrl = await startMockServer((url) => {
      if (!url.includes("/chat/completions")) return null;
      return {
        __sse: [
          sseFrame({ choices: [{ delta: { tool_calls: [{ index: 0, id: "c1", function: { name: "loop_forever", arguments: "{}" } }] } }] }),
          sseFrame({ choices: [{ delta: {}, finish_reason: "tool_calls" }] }),
          "data: [DONE]\n\n",
        ],
      };
    });
    const requester: DeepSeekRequester = {
      async *streamChat(body) {
        const response = await fetch(`${baseUrl}/chat/completions`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const payload = (await response.json()) as { __sse: string[] };
        const encoder = new TextEncoder();
        const stream = new ReadableStream<Uint8Array>({
          start(controller) {
            for (const chunk of payload.__sse) controller.enqueue(encoder.encode(chunk));
            controller.close();
          },
        });
        const { parseSseStream } = await import("../src/deepseek-api.js");
        yield* parseSseStream(stream);
      },
      async completeChat() {
        throw new Error("not used");
      },
      async getBalance() {
        throw new Error("not used");
      },
    };

    const result = await runAgentLoop({
      requester,
      model: "deepseek-chat",
      systemPrompt: "test",
      prompt: "go",
      tools: [
        {
          schema: { type: "function", function: { name: "loop_forever", parameters: { type: "object", properties: {} } } },
          execute: async () => "again",
        },
      ],
      maxSteps: 2,
      maxTokensPerStep: 64,
    });
    expect(result.stopReason).toBe("stepBudget");
    expect(result.iterations).toBe(2);
  });
});
