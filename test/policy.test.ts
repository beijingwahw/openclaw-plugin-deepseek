import { afterEach, describe, expect, it } from "vitest";
import type { Server } from "node:http";
import { createServer } from "node:http";
import { AdaptiveCascadePolicy, classifyTask, lengthBucketOf } from "../src/policy.js";
import { runUniversalCascade } from "../src/universal-cascade.js";
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

describe("task classification", () => {
  it("buckets prompts by shape and length", () => {
    expect(classifyTask("Refactor this function to use async")).toBe("code");
    expect(classifyTask("Prove that the sum of two evens is even")).toBe("reasoning");
    expect(classifyTask("What is the latest price of GPU X?")).toBe("factual");
    expect(classifyTask("Write a poem about the sea")).toBe("creative");
    expect(classifyTask("Help me with this thing")).toBe("general");
    expect(lengthBucketOf(100)).toBe("short");
    expect(lengthBucketOf(1000)).toBe("medium");
    expect(lengthBucketOf(5000)).toBe("long");
  });
});

describe("AdaptiveCascadePolicy", () => {
  it("recommends skip-judge after consistently accepted drafts", () => {
    const policy = new AdaptiveCascadePolicy();
    const prompt = "Write a TypeScript function to reverse a string.";
    for (let i = 0; i < 8; i++) {
      policy.record({ tier0: "deepseek-flash", finalTier: "deepseek-flash", prompt, accepted: true, escalated: false });
    }
    const rec = policy.recommend("deepseek-flash", prompt);
    expect(rec.mode).toBe("skip-judge");
    expect(rec.samples).toBeGreaterThanOrEqual(6);
    expect(rec.acceptanceRate).toBeGreaterThanOrEqual(0.85);
  });

  it("recommends direct-strong when tier-0 keeps failing", () => {
    const policy = new AdaptiveCascadePolicy();
    const prompt = "Prove the Riemann hypothesis in full generality, reasoning step by step.";
    for (let i = 0; i < 12; i++) {
      policy.record({ tier0: "deepseek-flash", finalTier: "deepseek-v4-pro", prompt, accepted: false, escalated: true });
    }
    const rec = policy.recommend("deepseek-flash", prompt);
    expect(rec.mode).toBe("direct-strong");
  });

  it("defaults to judge with insufficient evidence", () => {
    const policy = new AdaptiveCascadePolicy();
    const rec = policy.recommend("deepseek-flash", "novel prompt shape");
    expect(rec.mode).toBe("judge");
    expect(rec.samples).toBe(0);
  });

  it("persists learning across instances", async () => {
    const os = await import("node:os");
    const { mkdtemp, stat } = await import("node:fs/promises");
    const stateDir = await mkdtemp(`${os.tmpdir()}/policy-test-`);
    const first = new AdaptiveCascadePolicy({ stateDir });
    for (let i = 0; i < 8; i++) {
      first.record({ tier0: "t0", finalTier: "t0", prompt: "write code for x", accepted: true, escalated: false });
    }
    await new Promise((r) => setTimeout(r, 2_600));
    const snap = `${stateDir}/openclaw-deepseek/cascade-policy.json`;
    expect((await stat(snap)).isFile()).toBe(true);

    const second = new AdaptiveCascadePolicy({ stateDir });
    await second.ready();
    const rec = second.recommend("t0", "write code for y");
    expect(rec.mode).toBe("skip-judge"); // learning survived the restart
  });
});

describe("adaptive universal cascade", () => {
  function makeExecutor(behavior: { draftText: string; grade: "accept" | "escalate" }) {
    return {
      label: "fake",
      async complete(params: any) {
        if (params.systemPrompt.includes("quality judge")) {
          return {
            text: `{"grade":"${behavior.grade}","confidence":0.9,"reason":"ok"}`,
            usage: {},
          };
        }
        return { text: behavior.draftText, usage: {} };
      },
    };
  }

  it("uses the policy fast path (skip-judge) on high-acceptance shapes", async () => {
    const policy = new AdaptiveCascadePolicy();
    // Seed: tier-0 is very reliable for code/short prompts.
    const prompt = "Write a TypeScript function to reverse a string.";
    for (let i = 0; i < 8; i++) {
      policy.record({ tier0: "a/cheap", finalTier: "a/cheap", prompt, accepted: true, escalated: false });
    }
    let judgeCalls = 0;
    const executor = makeExecutor({ draftText: "const rev = (s) => [...s].reverse().join('');", grade: "accept" });
    const wrapped = {
      label: executor.label,
      async complete(params: any) {
        if (params.systemPrompt.includes("quality judge")) judgeCalls += 1;
        return executor.complete(params);
      },
    };
    const events: string[] = [];
    const result = await runUniversalCascade({
      resolveExecutor: () => wrapped as any,
      tiers: ["a/cheap", "b/strong"],
      systemPrompt: "s",
      prompt,
      policy,
      onEvent: (e) => events.push(e.type),
    });
    // Fast path: no judge call for this run (first skip after seeding).
    expect(judgeCalls).toBe(0);
    expect(result.escalated).toBe(false);
    expect(result.judge.reason).toContain("policy fast path");
    expect(events).toContain("routeMode");
    expect(events).toContain("final");
  });

  it("starts directly at the strong tier on direct-strong recommendation", async () => {
    const policy = new AdaptiveCascadePolicy();
    const prompt = "Prove the Riemann hypothesis, reasoning step by step in full generality.";
    for (let i = 0; i < 12; i++) {
      policy.record({ tier0: "a/cheap", finalTier: "b/strong", prompt, accepted: false, escalated: true });
    }
    let cheapCalls = 0;
    const executor = {
      label: "fake",
      async complete(params: any) {
        if (params.ref === "a/cheap") cheapCalls += 1;
        if (params.systemPrompt.includes("quality judge")) {
          return { text: '{"grade":"accept","confidence":0.9,"reason":"ok"}', usage: {} };
        }
        return { text: `answer from ${params.ref}`, usage: {} };
      },
    };
    const result = await runUniversalCascade({
      resolveExecutor: () => executor as any,
      tiers: ["a/cheap", "b/strong"],
      systemPrompt: "s",
      prompt,
      policy,
    });
    expect(cheapCalls).toBe(0); // weak tier skipped entirely
    expect(result.finalRef).toBe("b/strong");
    expect(result.calls.map((c) => c.ref)).not.toContain("a/cheap");
  });
});

describe("genesis tool e2e", () => {
  it("calibrates the policy through the registered plugin", async () => {
    const baseUrl = await new Promise<string>((resolve) => {
      server = createServer((req, res) => {
        let data = "";
        req.on("data", (c) => (data += c));
        req.on("end", () => {
          res.writeHead(200, { "Content-Type": "text/event-stream" });
          res.end(
            [
              sseFrame({ choices: [{ delta: { content: '{"grade":"accept","confidence":0.9,"reason":"routine"}' } }] }),
              sseFrame({ choices: [{ delta: {}, finish_reason: "stop" }] }),
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
        webSearch: { search: async () => ({ content: [] }) },
        llm: { complete: async () => ({ text: "x", provider: "zai", model: "glm" }) },
      },
    };
    plugin.register(api);

    const genesis = registered.tools.find((t: any) => t.name === "deepseek_genesis");
    expect(genesis).toBeTruthy();
    const result = await genesis.execute("t", {});
    const text = result.content.map((c: any) => c.text).join("");
    expect(text).toContain("calibrated");
    expect(text).toContain("Learned routing table");
    expect(result.details.probes.length).toBe(5); // one per shape
  });
});
