import { afterEach, describe, expect, it } from "vitest";
import type { Server } from "node:http";
import { createServer } from "node:http";
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

function makeRequester(baseUrl: string): DeepSeekRequester {
  const client = new DeepSeekClient({ baseUrl, apiKey: "sk-test", timeoutMs: 5000 });
  return {
    streamChat: (body, signal) => client.streamChat(body, signal),
    completeChat: (body, signal) => client.completeChat(body, signal),
    getBalance: (signal) => client.getBalance(signal),
  };
}

describe("thinking-mode request shaping", () => {
  it("sends thinking enabled + reasoning_effort and downgrades tool_choice", async () => {
    const bodies: any[] = [];
    const baseUrl = await new Promise<string>((resolve) => {
      server = createServer((req, res) => {
        let data = "";
        req.on("data", (c) => (data += c));
        req.on("end", () => {
          const body = JSON.parse(data || "{}");
          bodies.push(body);
          res.writeHead(200, { "Content-Type": "text/event-stream" });
          res.end(
            [
              sseFrame({ choices: [{ delta: { content: "ok" } }] }),
              sseFrame({ choices: [{ delta: {}, finish_reason: "stop" }] }),
              "data: [DONE]\n\n",
            ].join(""),
          );
        });
      });
      server.listen(0, "127.0.0.1", () => resolve(`http://127.0.0.1:${(server!.address() as any).port}`));
    });

    await runAgentLoop({
      requester: makeRequester(baseUrl),
      model: "deepseek-flash",
      systemPrompt: "s",
      prompt: "p",
      tools: [
        {
          schema: { type: "function", function: { name: "t", parameters: { type: "object", properties: {} } } },
          execute: async () => "x",
        },
      ],
      maxSteps: 2,
      maxTokensPerStep: 100,
      reasoningEffort: "max",
    });

    const body = bodies[0];
    expect(body.thinking).toEqual({ type: "enabled" });
    expect(body.reasoning_effort).toBe("max");
    expect(body.tool_choice).toBe("auto"); // never required/named while thinking
    expect(body.stream_options).toEqual({ include_usage: true });
  });

  it("disables thinking and strips sampling params when off", async () => {
    const bodies: any[] = [];
    const baseUrl = await new Promise<string>((resolve) => {
      server = createServer((req, res) => {
        let data = "";
        req.on("data", (c) => (data += c));
        req.on("end", () => {
          bodies.push(JSON.parse(data || "{}"));
          res.writeHead(200, { "Content-Type": "text/event-stream" });
          res.end(
            [
              sseFrame({ choices: [{ delta: { content: "ok" } }] }),
              sseFrame({ choices: [{ delta: {}, finish_reason: "stop" }] }),
              "data: [DONE]\n\n",
            ].join(""),
          );
        });
      });
      server.listen(0, "127.0.0.1", () => resolve(`http://127.0.0.1:${(server!.address() as any).port}`));
    });

    await runAgentLoop({
      requester: makeRequester(baseUrl),
      model: "deepseek-flash",
      systemPrompt: "s",
      prompt: "p",
      tools: [],
      maxSteps: 1,
      maxTokensPerStep: 100,
      reasoningEffort: "none",
      temperature: 0.7,
    });

    const body = bodies[0];
    expect(body.thinking).toEqual({ type: "disabled" });
    expect(body.reasoning_effort).toBeUndefined();
    expect(body.temperature).toBe(0.7); // only sent when thinking is off
  });

  it("replays reasoning_content on assistant tool-call turns", async () => {
    const bodies: any[] = [];
    const baseUrl = await new Promise<string>((resolve) => {
      server = createServer((req, res) => {
        let data = "";
        req.on("data", (c) => (data += c));
        req.on("end", () => {
          const body = JSON.parse(data || "{}");
          bodies.push(body);
          const last = body.messages[body.messages.length - 1];
          if (last.role === "user") {
            res.writeHead(200, { "Content-Type": "text/event-stream" });
            res.end(
              [
                sseFrame({ choices: [{ delta: { reasoning_content: "pondering" } }] }),
                sseFrame({ choices: [{ delta: { tool_calls: [{ index: 0, id: "c1", function: { name: "t", arguments: "{}" } }] } }] }),
                sseFrame({ choices: [{ delta: {}, finish_reason: "tool_calls" }] }),
                "data: [DONE]\n\n",
              ].join(""),
            );
          } else {
            res.writeHead(200, { "Content-Type": "text/event-stream" });
            res.end(
              [
                sseFrame({ choices: [{ delta: { content: "done" } }] }),
                sseFrame({ choices: [{ delta: {}, finish_reason: "stop" }] }),
                "data: [DONE]\n\n",
              ].join(""),
            );
          }
        });
      });
      server.listen(0, "127.0.0.1", () => resolve(`http://127.0.0.1:${(server!.address() as any).port}`));
    });

    const result = await runAgentLoop({
      requester: makeRequester(baseUrl),
      model: "deepseek-flash",
      systemPrompt: "s",
      prompt: "p",
      tools: [
        {
          schema: { type: "function", function: { name: "t", parameters: { type: "object", properties: {} } } },
          execute: async () => "x",
        },
      ],
      maxSteps: 3,
      maxTokensPerStep: 100,
      reasoningEffort: "high",
    });

    expect(result.answer).toBe("done");
    const secondRequest = bodies[1];
    const assistantTurn = secondRequest.messages.find((m: any) => m.role === "assistant" && m.tool_calls);
    // DeepSeek requires reasoning_content replay on tool-call turns (400 otherwise).
    expect(assistantTurn?.reasoning_content).toBe("pondering");
  });

  it("retries 429 responses with backoff before failing", async () => {
    let calls = 0;
    const baseUrl = await new Promise<string>((resolve) => {
      server = createServer((req, res) => {
        let data = "";
        req.on("data", (c) => (data += c));
        req.on("end", () => {
          calls += 1;
          if (calls === 1) {
            res.writeHead(429, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: { message: "rate limited" } }));
            return;
          }
          // completeChat is non-streaming: respond with plain JSON.
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ choices: [{ message: { role: "assistant", content: "ok" } }] }));
        });
      });
      server.listen(0, "127.0.0.1", () => resolve(`http://127.0.0.1:${(server!.address() as any).port}`));
    });

    // completeChat (non-streaming) is the retryable path.
    const client = new DeepSeekClient({ baseUrl, apiKey: "sk", timeoutMs: 5000 });
    const response = await client.completeChat({ model: "m", messages: [{ role: "user", content: "hi" }] });
    expect(calls).toBe(2);
    expect(response.choices?.[0]?.message?.content).toBe("ok");
  });
});
