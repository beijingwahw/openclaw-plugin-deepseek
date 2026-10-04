/**
 * DeepSeek Anthropic-Messages transport.
 *
 * DeepSeek exposes an Anthropic-compatible endpoint at
 * https://api.deepseek.com/anthropic with automatic model mapping:
 *   claude-opus*   -> deepseek-v4-pro
 *   claude-sonnet* / claude-haiku* / unknown -> deepseek-flash
 *
 * This lets Claude-ecosystem tooling (array system prompts, content blocks,
 * cache_control markers, x-api-key auth) talk to DeepSeek natively. The
 * plugin exposes it as a second provider so OpenClaw can route
 * `openclaw-deepseek-anthropic/claude-*` refs through this surface.
 */
import type { DeepSeekUsage } from "./types.js";

export const ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic";

/** Maps Claude model names onto DeepSeek surfaces per DeepSeek's docs. */
export function mapClaudeRefToDeepSeek(claudeRef: string): string {
  const model = claudeRef.toLowerCase();
  if (model.includes("opus")) return "deepseek-v4-pro";
  return "deepseek-flash"; // sonnet, haiku, and unknown refs
}

export interface AnthropicMessage {
  role: "user" | "assistant";
  content: string | Array<AnthropicContentBlock>;
}

export type AnthropicContentBlock =
  | { type: "text"; text: string }
  | { type: "thinking"; thinking: string }
  | { type: "tool_use"; id: string; name: string; input: Record<string, unknown> }
  | { type: "tool_result"; tool_use_id: string; content: string };

export interface AnthropicStreamEvent {
  type: "text" | "thinking" | "usage" | "finish";
  text?: string;
  finishReason?: string | null;
  usage?: DeepSeekUsage;
}

export interface AnthropicClientOptions {
  baseUrl?: string;
  apiKey: string;
  timeoutMs: number;
  fetchFn?: typeof fetch;
}

export class DeepSeekAnthropicClient {
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly timeoutMs: number;
  private readonly fetchFn: typeof fetch;

  constructor(options: AnthropicClientOptions) {
    this.baseUrl = (options.baseUrl ?? ANTHROPIC_BASE_URL).replace(/\/+$/, "");
    this.apiKey = options.apiKey;
    this.timeoutMs = options.timeoutMs;
    this.fetchFn = options.fetchFn ?? fetch;
  }

  /**
   * Streams one Anthropic Messages completion. SSE events:
   * message_start / content_block_start / content_block_delta /
   * content_block_stop / message_delta / message_stop.
   */
  async *streamMessage(
    params: {
      model: string;
      system?: string;
      messages: AnthropicMessage[];
      maxTokens?: number;
      temperature?: number;
      thinking?: boolean;
    },
    signal?: AbortSignal,
  ): AsyncGenerator<AnthropicStreamEvent> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    timer.unref?.(); // never hold the gateway process open
    const onOuterAbort = () => controller.abort();
    signal?.addEventListener("abort", onOuterAbort, { once: true });
    let usage: DeepSeekUsage | undefined;
    let finishReason: string | null = null;
    try {
      const body: Record<string, unknown> = {
        model: params.model,
        max_tokens: params.maxTokens ?? 8192,
        messages: params.messages,
        stream: true,
      };
      if (params.system !== undefined) body.system = params.system;
      if (params.temperature !== undefined) body.temperature = params.temperature;
      if (params.thinking) {
        body.thinking = { type: "enabled", budget_tokens: 4096 };
      }
      const response = await this.fetchFn(`${this.baseUrl}/v1/messages`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-api-key": this.apiKey,
          "anthropic-version": "2023-06-01",
          Accept: "text/event-stream",
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!response.ok || !response.body) {
        let text = "";
        try {
          text = await response.text();
        } catch {
          // Body already consumed or connection dropped; the status code is
          // the actionable signal, the body text is best-effort detail.
          text = "<unreadable body>";
        }
        throw new Error(`DeepSeek Anthropic endpoint error ${response.status}: ${text.slice(0, 300)}`);
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let sep: number;
        while ((sep = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, sep).replace(/\r$/, "");
          buffer = buffer.slice(sep + 1);
          const event = parseAnthropicSseLine(line);
          if (!event) continue;
          if (event.type === "usage") {
            usage = event.usage ?? usage;
            // message_delta frames can carry both usage and the stop reason.
            const deltaReason = (event as { finishReason?: string | null }).finishReason;
            if (deltaReason) finishReason = deltaReason;
            yield event; // transparent: consumers see every usage frame
            continue;
          }
          if (event.type === "finish") {
            // message_delta carries the real stop reason; message_stop only
            // confirms completion, so never overwrite a concrete reason.
            finishReason = finishReason && finishReason !== "message_stop" ? finishReason : event.finishReason ?? finishReason;
            continue;
          }
          yield event;
        }
      }
      if (usage) yield { type: "usage", usage };
      yield { type: "finish", finishReason };
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onOuterAbort);
    }
  }
}

function parseAnthropicSseLine(line: string): AnthropicStreamEvent | null {
  const trimmed = line.trim();
  if (!trimmed || trimmed.startsWith(":") || trimmed.startsWith("event:")) return null;
  if (!trimmed.startsWith("data:")) return null;
  const payload = trimmed.slice(5).trim();
  if (payload === "[DONE]") return { type: "finish", finishReason: "done" };
  let parsed: unknown;
  try {
    parsed = JSON.parse(payload);
  } catch {
    // Non-JSON SSE payload: the parser only cares about JSON frames.
    return null;
  }
  if (!parsed || typeof parsed !== "object") return null;
  const frame = parsed as {
    type?: string;
    message?: { usage?: { input_tokens?: number; output_tokens?: number; cache_read_input_tokens?: number; cache_creation_input_tokens?: number } };
    delta?: { thinking?: string; text?: string; stop_reason?: string };
    usage?: { input_tokens?: number; output_tokens?: number; cache_read_input_tokens?: number; cache_creation_input_tokens?: number };
  };

  if (frame.type === "content_block_delta") {
    const delta = frame.delta ?? {};
    if (typeof delta.thinking === "string" && delta.thinking) {
      return { type: "thinking", text: delta.thinking };
    }
    if (typeof delta.text === "string" && delta.text) {
      return { type: "text", text: delta.text };
    }
    return null;
  }
  if (frame.type === "message_start" && frame.message?.usage) {
    const u = frame.message.usage;
    return {
      type: "usage",
      usage: {
        prompt_tokens: u.input_tokens,
        completion_tokens: u.output_tokens,
        prompt_cache_hit_tokens: u.cache_read_input_tokens,
        prompt_cache_miss_tokens: u.cache_creation_input_tokens,
      },
    };
  }
  if (frame.type === "message_delta") {
    const usageOut = frame.usage;
    const stopReason = frame.delta?.stop_reason;
    return {
      type: "usage",
      usage: usageOut
        ? {
            prompt_tokens: usageOut.input_tokens,
            completion_tokens: usageOut.output_tokens,
            prompt_cache_hit_tokens: usageOut.cache_read_input_tokens,
            prompt_cache_miss_tokens: usageOut.cache_creation_input_tokens,
          }
        : undefined,
      finishReason: typeof stopReason === "string" ? stopReason : null,
    } as AnthropicStreamEvent;
  }
  if (frame.type === "message_stop") {
    return { type: "finish", finishReason: "message_stop" };
  }
  return null;
}
