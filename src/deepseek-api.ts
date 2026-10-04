/**
 * Minimal DeepSeek OpenAI-compatible API client for the harness plugin.
 *
 * Implements streaming chat completions (SSE), plain completions, and the
 * balance endpoint. No SDK dependencies: plain fetch + ReadableStream.
 */
import type { ConcurrencyGovernor, GovernorOutcome } from "./concurrency.js";
import { isRecord, parseToolCallFragment, parseUsageFrame, readString } from "./validate.js";
import type {
  DeepSeekChatResponse,
  DeepSeekRequester,
  DeepSeekStreamEvent,
} from "./types.js";

export interface DeepSeekClientOptions {
  baseUrl: string;
  apiKey: string;
  timeoutMs: number;
  fetchFn?: typeof fetch;
  /** Optional AIMD concurrency governor shared across all requests. */
  governor?: ConcurrencyGovernor;
}

export class DeepSeekApiError extends Error {
  constructor(
    readonly status: number,
    readonly bodyText: string,
    readonly url: string,
  ) {
    super(`DeepSeek API error ${status} at ${url}: ${truncate(bodyText, 500)}`);
    this.name = "DeepSeekApiError";
  }
}

export class DeepSeekAbortError extends Error {
  constructor(message = "DeepSeek request aborted") {
    super(message);
    this.name = "DeepSeekAbortError";
  }
}

/** Retryable conditions: HTTP 429 plus DeepSeek's non-standard finish reason. */
function isRetryableError(error: unknown): boolean {
  if (error instanceof DeepSeekApiError) {
    if (error.status === 429 || error.status >= 500) return true;
    return error.bodyText.includes("insufficient_system_resource");
  }
  return false;
}

const RETRY_DELAYS_MS = [1_000, 3_000, 9_000];

async function withRetries<T>(
  operation: () => Promise<T>,
  signal: AbortSignal | undefined,
  onRetry?: (attempt: number, delayMs: number, error: unknown) => void,
): Promise<T> {
  let lastError: unknown;
  for (let attempt = 0; attempt <= RETRY_DELAYS_MS.length; attempt++) {
    try {
      return await operation();
    } catch (error) {
      lastError = error;
      if (attempt === RETRY_DELAYS_MS.length || !isRetryableError(error)) throw error;
      const delayMs = RETRY_DELAYS_MS[attempt];
      onRetry?.(attempt + 1, delayMs, error);
      await abortableDelay(delayMs, signal);
    }
  }
  throw lastError;
}

function abortableDelay(ms: number, signal: AbortSignal | undefined): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    timer.unref?.(); // retry backoff must not pin the event loop
    const onAbort = () => {
      clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

export class DeepSeekClient implements DeepSeekRequester {
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly timeoutMs: number;
  private readonly fetchFn: typeof fetch;
  private readonly governor: ConcurrencyGovernor | undefined;

  constructor(options: DeepSeekClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.apiKey = options.apiKey;
    this.timeoutMs = options.timeoutMs;
    this.fetchFn = options.fetchFn ?? fetch;
    this.governor = options.governor;
  }

  private url(path: string): string {
    return `${this.baseUrl}${path}`;
  }

  private headers(): HeadersInit {
    return {
      "Content-Type": "application/json",
      Accept: "application/json",
      Authorization: `Bearer ${this.apiKey}`,
    };
  }

  private async readError(response: Response, url: string): Promise<never> {
    let bodyText = "";
    try {
      bodyText = await response.text();
    } catch {
      // Body already consumed or connection dropped; the status code is the
      // actionable signal, body text is best-effort detail.
      bodyText = "<unreadable body>";
    }
    throw new DeepSeekApiError(response.status, bodyText, url);
  }

  async completeChat(
    body: Record<string, unknown>,
    signal?: AbortSignal,
  ): Promise<DeepSeekChatResponse> {
    const model = typeof body.model === "string" ? body.model : "unknown";
    await this.governor?.acquire(model);
    try {
      return await withRetries(
        async () => {
          const url = this.url("/chat/completions");
          const response = await this.fetchWithTimeout(
            url,
            {
              method: "POST",
              headers: this.headers(),
              body: JSON.stringify({ ...body, stream: false }),
            },
            signal,
          );
          if (!response.ok) await this.readError(response, url);
          return (await response.json()) as DeepSeekChatResponse;
        },
        signal,
      );
    } finally {
      this.governor?.release(model, "success");
    }
  }

  async getBalance(
    signal?: AbortSignal,
  ): Promise<{ balance: string; currency: string | null; raw: unknown }> {
    const url = this.url("/user/balance");
    const response = await this.fetchWithTimeout(
      url,
      {
        method: "GET",
        headers: this.headers(),
      },
      signal,
    );
    if (!response.ok) await this.readError(response, url);
    const data = (await response.json()) as {
      is_available?: boolean;
      balance_infos?: Array<{ currency?: string; total_balance?: string }>;
    };
    const info = data.balance_infos?.find((i) => i.total_balance !== undefined);
    return {
      balance: info?.total_balance ?? "unknown",
      currency: info?.currency ?? null,
      raw: data,
    };
  }

  async *streamChat(
    body: Record<string, unknown>,
    signal?: AbortSignal,
  ): AsyncIterable<DeepSeekStreamEvent> {
    // Streaming responses cannot be replayed mid-body; retry only the initial
    // request establishment (429/5xx before the first byte arrives inside
    // withRetries' first attempt is covered by the initial fetch throw).
    const url = this.url("/chat/completions");
    const response = await this.fetchWithTimeout(
      url,
      {
        method: "POST",
 headers: { ...this.headers(), Accept: "text/event-stream" },
        body: JSON.stringify({ ...body, stream: true }),
      },
      signal,
    );
    if (!response.ok) await this.readError(response, url);
    if (!response.body) {
      throw new Error(`DeepSeek API returned no stream body at ${url}`);
    }
    yield* parseSseStream(response.body);
  }

  private withTimeout(signal?: AbortSignal): { signal: AbortSignal; cleanup: () => void } {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    timer.unref?.(); // never hold the gateway process open for a timeout
    const onOuterAbort = () => controller.abort();
    signal?.addEventListener("abort", onOuterAbort, { once: true });
    const cleanup = () => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onOuterAbort);
    };
    // Auto-cleanup when the request settles either way; holders may also call
    // cleanup() explicitly to stop the timeout early (e.g. stream drained).
    controller.signal.addEventListener("abort", () => {
      if (signal?.aborted) cleanup();
    }, { once: true });
    const settled = controller.signal as AbortSignal & { _deepseekCleanup?: () => void };
    settled._deepseekCleanup = cleanup;
    return { signal: controller.signal, cleanup };
  }

  /** Runs one fetch with timeout lifecycle management (no timer leaks). */
  private async fetchWithTimeout(
    url: string,
    init: RequestInit,
    signal?: AbortSignal,
  ): Promise<Response> {
    const { signal: timeoutSignal, cleanup } = this.withTimeout(signal);
    try {
      return await this.fetchFn(url, { ...init, signal: timeoutSignal });
    } finally {
      cleanup();
    }
  }
}

/**
 * Parses an OpenAI-compatible SSE stream into normalized DeepSeek events.
 * Tolerates comment lines, missing fields, and non-JSON error payloads.
 */
export async function* parseSseStream(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<DeepSeekStreamEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const toolCallAccumulators = new Map<number, { id: string; name: string; args: string }>();
  let finishReason: string | null = null;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let sep: number;
      while ((sep = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, sep).replace(/\r$/, "");
        buffer = buffer.slice(sep + 1);
        const event = parseSseLine(line, toolCallAccumulators);
        if (event) {
          if (event.type === "finish") finishReason = event.finishReason;
          yield event;
        }
      }
    }
    const tail = buffer.trim();
    if (tail) {
      const event = parseSseLine(tail, toolCallAccumulators);
      if (event) {
        if (event.type === "finish") finishReason = event.finishReason;
        yield event;
      }
    }
    // Emit accumulated tool calls once the stream is drained.
    for (const call of toolCallAccumulators.values()) {
      if (call.id && call.name) {
        yield { type: "toolCall", call: { id: call.id, type: "function", function: { name: call.name, arguments: call.args || "{}" } } };
      }
    }
    if (finishReason === null) {
      yield { type: "finish", finishReason: null };
    }
  } finally {
    reader.releaseLock();
  }
}

function parseSseLine(
  line: string,
  toolCallAccumulators: Map<number, { id: string; name: string; args: string }>,
): DeepSeekStreamEvent | null {
  const trimmed = line.trim();
  if (!trimmed || trimmed.startsWith(":")) return null;
  if (trimmed === "data: [DONE]") return { type: "finish", finishReason: "done" };
  if (!trimmed.startsWith("data:")) return null;
  let payload: string;
  if (trimmed.startsWith("data: ") || trimmed.startsWith("data:\t")) {
    payload = trimmed.slice(5).trimStart();
  } else {
    payload = trimmed.slice(5);
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(payload);
  } catch {
    // Non-JSON data frame: surface the raw payload as text (some proxies emit
    // plain-string keep-alives).
    return { type: "text", text: payload };
  }
  if (parsed === null || typeof parsed !== "object") return null;
  if (!isRecord(parsed)) return null;
  const frame = parsed as unknown as {
    usage?: Record<string, unknown>;
    choices?: Array<{ delta?: Record<string, unknown>; finish_reason?: string | null }>;
  };
  // stream_options.include_usage frames carry usage with an empty choices array.
  if (isRecord(frame.usage) && (!frame.choices || frame.choices.length === 0)) {
    const usage = parseUsageFrame(frame.usage);
    if (usage) return { type: "usage", usage };
    return null;
  }
  const delta = frame.choices?.[0];
  if (!delta) return null;
  if (typeof delta.finish_reason === "string" && delta.finish_reason) {
    return { type: "finish", finishReason: delta.finish_reason };
  }
  const d = delta.delta ?? {};
  let emitted: DeepSeekStreamEvent | null = null;
  if (typeof d.reasoning_content === "string" && d.reasoning_content) {
    emitted = { type: "reasoning", text: d.reasoning_content };
  }
  if (typeof d.content === "string" && d.content) {
    emitted = emitted
      ? null // Prefer text over reasoning when both arrive in one chunk (rare).
      : { type: "text", text: d.content };
    if (emitted === null) {
      // Both present: emit reasoning first is lossy; emit combined as text.
      emitted = { type: "text", text: `${d.reasoning_content ?? ""}${d.content}` };
    }
  }
  if (Array.isArray(d.tool_calls)) {
    for (const rawFragment of d.tool_calls) {
      const fragment = parseToolCallFragment(rawFragment);
      if (!fragment) continue;
      const acc = toolCallAccumulators.get(fragment.index) ?? { id: "", name: "", args: "" };
      if (fragment.id) acc.id = fragment.id;
      if (fragment.name) acc.name = fragment.name;
      if (fragment.argsDelta) acc.args += fragment.argsDelta;
      toolCallAccumulators.set(fragment.index, acc);
    }
  }
  return emitted;
}

function truncate(text: string, max: number): string {
  return text.length <= max ? text : `${text.slice(0, max)}…`;
}
