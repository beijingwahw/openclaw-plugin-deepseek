/**
 * FIM (Fill-in-the-Middle) code completion via the DeepSeek beta endpoint.
 *
 * FIM only works on the non-thinking path and takes `prompt` + `suffix`
 * instead of messages. Used for IDE-style code completion inside OpenClaw
 * tools.
 */
import type { DeepSeekRequester, DeepSeekUsage } from "./types.js";

export interface FimOptions {
  requester: DeepSeekRequester;
  prompt: string;
  suffix?: string;
  /** Model for FIM completion. FIM requires a non-thinking-capable surface. */
  model?: string;
  maxTokens?: number;
  signal?: AbortSignal;
  onText?: (text: string) => void;
}

export interface FimResult {
  completion: string;
  usage: DeepSeekUsage;
}

export async function runFimCompletion(options: FimOptions): Promise<FimResult> {
  const { requester } = options;
  // The beta FIM surface is a plain /completions endpoint. We reuse the
  // requester's streaming path with a shaped body; DeepSeek accepts stream
  // on the beta endpoint and the SSE parser is shared.
  const body: Record<string, unknown> = {
    model: options.model ?? "deepseek-flash",
    prompt: options.prompt,
    max_tokens: options.maxTokens ?? 512,
    stream: true,
    stream_options: { include_usage: true },
  };
  if (options.suffix !== undefined && options.suffix.length > 0) {
    body.suffix = options.suffix;
  }
  let completion = "";
  let usage: DeepSeekUsage = {};
  for await (const event of requester.streamChat(body, options.signal)) {
    if (event.type === "usage") {
      usage = event.usage;
    } else if (event.type === "text") {
      completion += event.text;
      options.onText?.(event.text);
    }
  }
  return { completion, usage };
}
