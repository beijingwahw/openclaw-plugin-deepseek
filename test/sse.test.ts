import { describe, expect, it } from "vitest";
import { parseSseStream } from "../src/deepseek-api.js";
import type { DeepSeekStreamEvent } from "../src/types.js";

function sse(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  let index = 0;
  return new ReadableStream<Uint8Array>({
    pull(controller) {
      if (index < chunks.length) {
        controller.enqueue(encoder.encode(chunks[index]));
        index += 1;
      } else {
        controller.close();
      }
    },
  });
}

async function collect(stream: ReadableStream<Uint8Array>): Promise<DeepSeekStreamEvent[]> {
  const events: DeepSeekStreamEvent[] = [];
  for await (const event of parseSseStream(stream)) {
    events.push(event);
  }
  return events;
}

function frame(delta: Record<string, unknown>, extra: Record<string, unknown> = {}): string {
  return `data: ${JSON.stringify({ choices: [{ delta }], ...extra })}\n\n`;
}

describe("parseSseStream", () => {
  it("emits text deltas", async () => {
    const events = await collect(sse([frame({ content: "Hello" }), frame({ content: " world" }), "data: [DONE]\n\n"]));
    expect(events).toEqual([
      { type: "text", text: "Hello" },
      { type: "text", text: " world" },
      { type: "finish", finishReason: "done" },
    ]);
  });

  it("emits reasoning deltas", async () => {
    const events = await collect(sse([frame({ reasoning_content: "thinking..." }), frame({ content: "answer" })]));
    expect(events).toContainEqual({ type: "reasoning", text: "thinking..." });
    expect(events).toContainEqual({ type: "text", text: "answer" });
  });

  it("accumulates streamed tool calls", async () => {
    const events = await collect(
      sse([
        frame({ tool_calls: [{ index: 0, id: "call_1", function: { name: "web_search", arguments: '{"que' } }] }),
        frame({ tool_calls: [{ index: 0, function: { arguments: 'ry":"deepseek"}' } }] }),
        frame({ tool_calls: [{ index: 1, id: "call_2", function: { name: "read_file", arguments: '{"path":"/x"}' } }] }),
        "data: [DONE]\n\n",
      ]),
    );
    const toolCalls = events.filter((e) => e.type === "toolCall");
    expect(toolCalls).toHaveLength(2);
    expect(toolCalls[0]).toEqual({
      type: "toolCall",
      call: {
        id: "call_1",
        type: "function",
        function: { name: "web_search", arguments: '{"query":"deepseek"}' },
      },
    });
    expect(toolCalls[1]).toEqual({
      type: "toolCall",
      call: {
        id: "call_2",
        type: "function",
        function: { name: "read_file", arguments: '{"path":"/x"}' },
      },
    });
  });

  it("captures usage frames with empty choices", async () => {
    const usage = { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 };
    const events = await collect(
      sse([frame({ content: "hi" }), `data: ${JSON.stringify({ choices: [], usage })}\n\n`, "data: [DONE]\n\n"]),
    );
    const usageEvents = events.filter((e) => e.type === "usage");
    expect(usageEvents).toHaveLength(1);
    expect(usageEvents[0]).toEqual({ type: "usage", usage });
  });

  it("handles finish_reason", async () => {
    const events = await collect(
      sse([
        `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: "tool_calls" }] })}\n\n`,
      ]),
    );
    expect(events).toContainEqual({ type: "finish", finishReason: "tool_calls" });
  });

  it("ignores comments and non-data lines, tolerates split chunks", async () => {
    const events = await collect(
      sse([": keep-alive\n\n", "data: ", JSON.stringify({ choices: [{ delta: { content: "split" } }] }) + "\n\n"]),
    );
    expect(events).toContainEqual({ type: "text", text: "split" });
  });

  it("emits synthetic finish when stream ends without [DONE]", async () => {
    const events = await collect(sse([frame({ content: "x" })]));
    const last = events[events.length - 1];
    expect(last?.type).toBe("finish");
    if (last?.type === "finish") expect(last.finishReason).toBeNull();
  });
});
