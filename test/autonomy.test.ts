import { describe, expect, it } from "vitest";
import { BudgetGuardian } from "../src/guardian.js";
import { AuditTrail } from "../src/audit.js";
import { CostLedger } from "../src/ledger.js";
import { runUniversalCascade } from "../src/universal-cascade.js";
import { sanitizeUserId } from "../src/loop.js";

describe("BudgetGuardian", () => {
  it("stays green under half budget and unconfigured without one", () => {
    const ledger = new CostLedger();
    const guardian = new BudgetGuardian({ ledger });
    expect(guardian.status().state).toBe("unconfigured");

    ledger.setDailyBudget(1.0);
    ledger.record({ provider: "deepseek", model: "deepseek-flash", ref: "deepseek/deepseek-flash", usage: { input: 100_000, output: 10_000 } });
    // ~$0.02 = 2% -> green
    expect(guardian.status().state).toBe("green");
    const advice = guardian.routeAdvice(["a/cheap", "b/strong"]);
    expect(advice.mode).toBe("normal");
    expect(advice.tiers).toHaveLength(2);
  });

  it("turns amber at 50% and red at 90% with emergency routing", () => {
    const ledger = new CostLedger();
    const transitions: string[] = [];
    const guardian = new BudgetGuardian({
      ledger,
      onTransition: (from, to) => transitions.push(`${from}->${to}`),
    });
    ledger.setDailyBudget(1.0);
    // ~$0.55 -> amber
    ledger.record({ provider: "deepseek", model: "deepseek-flash", ref: "deepseek/deepseek-flash", usage: { input: 3_000_000, output: 200_000 } });
    let status = guardian.status();
    expect(status.state).toBe("amber");
    let advice = guardian.routeAdvice(["a/cheap", "b/strong"]);
    expect(advice.mode).toBe("economize");
    expect(advice.tiers).toHaveLength(2); // economize keeps tiers

    // Push to red: flash ~$0.55 + pro ~$0.42 -> ~$0.97
    ledger.record({ provider: "deepseek", model: "deepseek-v4-pro", ref: "deepseek/deepseek-v4-pro", usage: { input: 400_000, output: 100_000 } });
    status = guardian.status();
    expect(status.state).toBe("red");
    advice = guardian.routeAdvice(["a/cheap", "b/strong", "c/strongest"]);
    expect(advice.mode).toBe("emergency");
    expect(advice.tiers).toEqual(["a/cheap"]); // single cheapest tier
    expect(advice.skipJudge).toBe(true);
    expect(transitions).toContain("amber->red");
  });
});

describe("AuditTrail", () => {
  it("records, filters, renders, and bounds entries", () => {
    let clock = 1_000_000;
    const audit = new AuditTrail({ now: () => clock });
    audit.record("cascade-route", "economy mode engaged", { percent: 92 });
    clock += 5;
    audit.record("cascade-escalation", "a/cheap -> b/strong", { confidence: 0.2 });
    clock += 5;
    audit.record("budget-transition", "amber -> red", {});

    const all = audit.recent({ limit: 10 });
    expect(all).toHaveLength(3);
    expect(all[0].kind).toBe("budget-transition"); // newest first

    const escalations = audit.recent({ kind: "cascade-escalation" });
    expect(escalations).toHaveLength(1);

    const text = audit.render({ limit: 2 });
    expect(text).toContain("Audit trail");
    expect(text).toContain("cascade-escalation");

    for (let i = 0; i < 600; i++) audit.record("fim-request", `fill ${i}`);
    expect(audit.recent({ limit: 1000 })).toHaveLength(500); // bounded
  });
});

describe("economic cascade routing", () => {
  it("emergency mode routes single-tier without judge calls", async () => {
    const ledger = new CostLedger();
    ledger.setDailyBudget(1.0);
    // ~$0.95 spent -> red.
    ledger.record({ provider: "deepseek", model: "deepseek-flash", ref: "deepseek/deepseek-flash", usage: { input: 5_000_000, output: 400_000 } });
    const guardian = new BudgetGuardian({ ledger });
    const audit = new AuditTrail();
    let judgeCalls = 0;
    const executor = {
      label: "fake",
      async complete(params: any) {
        if (params.systemPrompt.includes("quality judge")) {
          judgeCalls += 1;
          return { text: '{"grade":"escalate","confidence":0.1,"reason":"weak"}', usage: {} };
        }
        return { text: `from ${params.ref}`, usage: {} };
      },
    };
    const result = await runUniversalCascade({
      resolveExecutor: () => executor as any,
      tiers: ["a/cheap", "b/strong"],
      systemPrompt: "s",
      prompt: "some task",
      guardian,
      audit,
    });
    expect(judgeCalls).toBe(0); // emergency: no judge at all
    expect(result.finalRef).toBe("a/cheap"); // single cheapest tier
    expect(audit.recent({ kind: "cascade-route" })[0]?.summary).toContain("emergency");
  });
});

describe("user_id cache affinity", () => {
  it("sanitizes session ids into DeepSeek's accepted alphabet", () => {
    expect(sanitizeUserId("agent:main:im:466500fb")).toBe("agent-main-im-466500fb");
    expect(sanitizeUserId("session/with spaces&symbols!")).toBe("session-with-spaces-symbols-");
    expect(sanitizeUserId("")).toBe("openclaw-deepseek");
    expect(sanitizeUserId("x".repeat(600))).toHaveLength(512);
  });

  it("loop sends user_id on requests", async () => {
    const { runAgentLoop } = await import("../src/loop.js");
    const bodies: any[] = [];
    const { createServer } = await import("node:http");
    const server = await new Promise<any>((resolve) => {
      const s = createServer((req, res) => {
        let data = "";
        req.on("data", (c) => (data += c));
        req.on("end", () => {
          bodies.push(JSON.parse(data || "{}"));
          res.writeHead(200, { "Content-Type": "text/event-stream" });
          res.end(`data: ${JSON.stringify({ choices: [{ delta: { content: "ok" } }] })}\n\ndata: [DONE]\n\n`);
        });
      });
      s.listen(0, "127.0.0.1", () => resolve(s));
    });
    const port = (server.address() as any).port;
    const { DeepSeekClient } = await import("../src/deepseek-api.js");
    const client = new DeepSeekClient({ baseUrl: `http://127.0.0.1:${port}`, apiKey: "sk", timeoutMs: 5000 });
    await runAgentLoop({
      requester: {
        streamChat: (body, signal) => client.streamChat(body, signal),
        completeChat: (b, s) => client.completeChat(b, s),
        getBalance: (s) => client.getBalance(s),
      },
      model: "m",
      systemPrompt: "s",
      prompt: "p",
      tools: [],
      maxSteps: 1,
      maxTokensPerStep: 50,
      userId: "oc-agent-main-im-123",
    });
    expect(bodies[0].user_id).toBe("oc-agent-main-im-123");
    await new Promise<void>((r) => server.close(() => r()));
  });
});
