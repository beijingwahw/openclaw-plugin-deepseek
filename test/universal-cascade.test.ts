import { afterEach, describe, expect, it } from "vitest";
import type { Server } from "node:http";
import { createServer } from "node:http";
import {
  createHostLlmExecutor,
  isNativeDeepSeekRef,
  nativeExecutorFromRequester,
  runUniversalCascade,
  type UniversalExecutor,
} from "../src/universal-cascade.js";
import { DeepSeekClient } from "../src/deepseek-api.js";

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

describe("isNativeDeepSeekRef", () => {
  it("classifies refs", () => {
    expect(isNativeDeepSeekRef("deepseek-flash")).toBe(true);
    expect(isNativeDeepSeekRef("deepseek-v4-pro")).toBe(true);
    expect(isNativeDeepSeekRef("deepseek-cascade")).toBe(true);
    expect(isNativeDeepSeekRef("anthropic/claude-sonnet-4-6")).toBe(false);
    expect(isNativeDeepSeekRef("openai/gpt-5.5")).toBe(false);
    expect(isNativeDeepSeekRef("zai/glm-5.3")).toBe(false);
  });
});

describe("universal cascade", () => {
  it("runs pure-DeepSeek tiers through the native executor", async () => {
    const requests: any[] = [];
    const baseUrl = await startSseServer((body) => {
      requests.push(body);
      const system = body.messages?.[0]?.content ?? "";
      if (system.includes("quality judge")) {
        return [
          sseFrame({ choices: [{ delta: { content: '{"grade":"accept","confidence":0.95,"reason":"good"}' } }] }),
          sseFrame({ choices: [{ delta: {}, finish_reason: "stop" }] }),
          "data: [DONE]\n\n",
        ];
      }
      return [
        sseFrame({ choices: [{ delta: { content: "native answer" } }] }),
        sseFrame({ choices: [{ delta: {}, finish_reason: "stop" }] }),
        sseFrame({ choices: [], usage: { prompt_tokens: 10, completion_tokens: 5, prompt_cache_hit_tokens: 4 } }),
        "data: [DONE]\n\n",
      ];
    });

    const client = new DeepSeekClient({ baseUrl, apiKey: "sk", timeoutMs: 5000 });
    const native = nativeExecutorFromRequester(client);
    const result = await runUniversalCascade({
      resolveExecutor: () => native,
      tiers: ["deepseek-flash", "deepseek-v4-pro"],
      systemPrompt: "assistant",
      prompt: "q",
    });

    expect(result.finalRef).toBe("deepseek-flash");
    expect(result.escalated).toBe(false);
    expect(result.text).toBe("native answer");
    expect(result.usage.inputTokens).toBe(10); // draft reported usage; judge frame carried none
    expect(result.usage.cacheReadTokens).toBe(4);
    expect(result.usage.costUsd).toBeGreaterThan(0); // metered natively
    expect(result.calls.map((c) => c.role)).toEqual(["draft", "judge"]);
  });

  it("runs cross-provider tiers through mixed executors", async () => {
    const deepSeekCalls: any[] = [];
    const baseUrl = await startSseServer((body) => {
      deepSeekCalls.push(body);
      return [
        sseFrame({ choices: [{ delta: { content: "draft from deepseek" } }] }),
        sseFrame({ choices: [{ delta: {}, finish_reason: "stop" }] }),
        "data: [DONE]\n\n",
      ];
    });

    const hostCalls: string[] = [];
    const hostLlm = {
      complete: async (params: any) => {
        hostCalls.push(params.model ?? "(default)");
        if (params.systemPrompt?.includes("quality judge")) {
          return {
            text: '{"grade":"escalate","confidence":0.2,"reason":"too shallow"}',
            provider: "anthropic",
            model: "claude-sonnet-4-6",
            usage: { inputTokens: 100, outputTokens: 20, costUsd: 0.003 },
          };
        }
        return {
          text: "claude's careful answer",
          provider: "anthropic",
          model: "claude-sonnet-4-6",
          usage: { inputTokens: 500, outputTokens: 200, costUsd: 0.012 },
        };
      },
    };
    const host = createHostLlmExecutor(hostLlm, { pluginId: "openclaw-deepseek" });
    const client = new DeepSeekClient({ baseUrl, apiKey: "sk", timeoutMs: 5000 });
    const native = nativeExecutorFromRequester(client);

    const result = await runUniversalCascade({
      resolveExecutor: (ref) => (isNativeDeepSeekRef(ref) ? native : host),
      tiers: ["deepseek-flash", "anthropic/claude-sonnet-4-6"],
      judgeRef: "anthropic/claude-sonnet-4-6",
      systemPrompt: "assistant",
      prompt: "hard question",
    });

    // Draft came from DeepSeek native path.
    expect(deepSeekCalls.length).toBe(1);
    expect(deepSeekCalls[0].model).toBe("deepseek-flash");
    expect(deepSeekCalls[0].reasoning_effort).toBe("low");
    // Judge + escalated draft went through the host.
    expect(hostCalls).toEqual(["anthropic/claude-sonnet-4-6", "anthropic/claude-sonnet-4-6"]);
    expect(result.escalated).toBe(true);
    expect(result.finalRef).toBe("anthropic/claude-sonnet-4-6");
    expect(result.text).toBe("claude's careful answer");
    // Usage merged across providers.
    expect(result.usage.inputTokens).toBe(600);
    expect(result.usage.costUsd).toBeCloseTo(0.015, 4);
    expect(result.calls.map((c) => c.role)).toEqual(["draft", "judge", "escalated-draft"]);
  });

  it("accepts a good cross-provider draft without escalating", async () => {
    const hostLlm = {
      complete: async (params: any) => ({
        text: params.systemPrompt?.includes("quality judge")
          ? '{"grade":"accept","confidence":0.9,"reason":"complete"}'
          : "glm draft answer",
        provider: "zai",
        model: "glm-5.3",
        usage: { inputTokens: 50, outputTokens: 30, costUsd: 0.001 },
      }),
    };
    const host = createHostLlmExecutor(hostLlm, { pluginId: "openclaw-deepseek" });
    const result = await runUniversalCascade({
      resolveExecutor: () => host,
      tiers: ["zai/glm-5.3", "anthropic/claude-sonnet-4-6"],
      systemPrompt: "assistant",
      prompt: "q",
    });
    expect(result.escalated).toBe(false);
    expect(result.finalRef).toBe("zai/glm-5.3");
    expect(result.judge.grade).toBe("accept");
  });

  it("host executor forwards reasoning levels and skips reasoning when off", async () => {
    const seen: any[] = [];
    const hostLlm = {
      complete: async (params: any) => {
        seen.push(params);
        return { text: "ok", provider: "zai", model: "glm-5.3", usage: {} };
      },
    };
    const host = createHostLlmExecutor(hostLlm, { pluginId: "openclaw-deepseek" });
    await host.complete({ ref: "zai/glm-5.3", systemPrompt: "s", prompt: "p", reasoning: "max" });
    await host.complete({ ref: "zai/glm-5.3", systemPrompt: "s", prompt: "p", reasoning: "none" });
    expect(seen[0].reasoning).toBe("max");
    expect(seen[0].model).toBe("zai/glm-5.3");
    expect(seen[0].purpose).toBe("openclaw-deepseek:cascade");
    expect(seen[1].reasoning).toBeUndefined(); // off -> omitted for hosts
  });

  it("supports more than two tiers", async () => {
    let calls = 0;
    const executor: UniversalExecutor = {
      label: "fake",
      async complete(params) {
        calls += 1;
        if (params.systemPrompt.includes("quality judge")) {
          return { text: '{"grade":"escalate","confidence":0.1,"reason":"no"}', usage: {} };
        }
        return { text: `answer-from-${params.ref}`, usage: {} };
      },
    };
    const result = await runUniversalCascade({
      resolveExecutor: () => executor,
      tiers: ["a/cheap", "b/mid", "c/strong"],
      systemPrompt: "s",
      prompt: "p",
    });
    expect(result.finalRef).toBe("c/strong");
    expect(result.escalated).toBe(true);
    expect(result.text).toBe("answer-from-c/strong");
    expect(result.iterations).toBe(3);
  });
});
