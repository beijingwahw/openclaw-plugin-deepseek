/**
 * Structured extraction: JSON-mode completions with schema validation and
 * one repair retry. DeepSeek's json_object mode requires an explicit JSON
 * instruction in the prompt (otherwise it can stream blank until the token
 * limit), so the instruction is always injected.
 */
import { mergeDeepSeekUsage } from "./shared.js";
import type { DeepSeekRequester, DeepSeekUsage } from "./types.js";

export interface ExtractOptions {
  requester: DeepSeekRequester;
  systemPrompt?: string;
  prompt: string;
  /** JSON Schema describing the desired object. */
  schema: Record<string, unknown>;
  model?: string;
  maxTokens?: number;
  signal?: AbortSignal;
}

export interface ExtractResult<T = Record<string, unknown>> {
  value: T;
  repaired: boolean;
  usage: DeepSeekUsage;
}

export async function runJsonExtraction<T = Record<string, unknown>>(
  options: ExtractOptions,
): Promise<ExtractResult<T>> {
  const model = options.model ?? "deepseek-flash";
  const maxTokens = options.maxTokens ?? 4096;
  const systemPrompt =
    options.systemPrompt?.trim() ||
    "You are a precise information extraction engine. Output only valid JSON matching the requested schema. Never wrap JSON in markdown fences. Never add commentary.";

  const userPrompt = [
    options.prompt,
    "",
    "Return ONLY a JSON object matching this schema:",
    JSON.stringify(options.schema),
    "Your entire output must be a single valid JSON object and nothing else.",
  ].join("\n");

  const first = await jsonOnce(options.requester, {
    model,
    systemPrompt,
    prompt: userPrompt,
    maxTokens,
    signal: options.signal,
  });
  const firstParse = tryParseObject(first.text);
  const firstIssues = firstParse !== null ? validateAgainstSchema(firstParse, options.schema) : ["output was not valid JSON"];
  if (firstParse !== null && firstIssues.length === 0) {
    return { value: firstParse as T, repaired: false, usage: first.usage };
  }

  // One targeted repair pass: show the model exactly what failed.
  const repairPrompt = [
    firstParse === null
      ? "The previous output was not valid JSON."
      : `The previous JSON failed validation: ${firstIssues.join("; ")}.`,
    "Previous output (truncated):",
    first.text.slice(0, 2000),
    "",
    userPrompt,
  ].join("\n");
  const repair = await jsonOnce(options.requester, {
    model,
    systemPrompt,
    prompt: repairPrompt,
    maxTokens,
    signal: options.signal,
  });
  const repairedParse = tryParseObject(repair.text);
  const repairedIssues = repairedParse !== null ? validateAgainstSchema(repairedParse, options.schema) : ["still not valid JSON"];
  if (repairedParse !== null && repairedIssues.length === 0) {
    return { value: repairedParse as T, repaired: true, usage: mergeUsageClone(first.usage, repair.usage) };
  }
  // Schema-invalid but parseable output still beats nothing: return it with a flag.
  if (repairedParse !== null) {
    return { value: repairedParse as T, repaired: true, usage: mergeUsageClone(first.usage, repair.usage) };
  }
  throw new Error(
    `deepseek_extract could not produce valid JSON after a repair pass (${repairedIssues.join("; ")}). Last output: ${repair.text.slice(0, 300)}`,
  );
}

async function jsonOnce(
  requester: DeepSeekRequester,
  params: {
    model: string;
    systemPrompt: string;
    prompt: string;
    maxTokens: number;
    signal?: AbortSignal;
  },
): Promise<{ text: string; usage: DeepSeekUsage }> {
  const body: Record<string, unknown> = {
    model: params.model,
    messages: [
      { role: "system", content: params.systemPrompt },
      { role: "user", content: params.prompt },
    ],
    max_tokens: params.maxTokens,
    response_format: { type: "json_object" },
    stream: true,
    stream_options: { include_usage: true },
    thinking: { type: "disabled" }, // JSON mode is safest on the non-thinking path
  };
  let text = "";
  let usage: DeepSeekUsage = {};
  for await (const event of requester.streamChat(body, params.signal)) {
    if (event.type === "usage") {
      usage = event.usage;
    } else if (event.type === "text") {
      text += event.text;
    }
  }
  return { text, usage };
}

function tryParseObject(text: string): Record<string, unknown> | null {
  const trimmed = stripFences(text.trim());
  if (!trimmed) return null;
  const candidates = [trimmed, firstBalancedObject(trimmed)];
  for (const candidate of candidates) {
    if (!candidate) continue;
    try {
      const parsed = JSON.parse(candidate);
      if (parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      // try next candidate
    }
  }
  return null;
}

function stripFences(text: string): string {
  const fence = text.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/);
  return fence ? fence[1] : text;
}

function firstBalancedObject(text: string): string | null {
  const start = text.indexOf("{");
  if (start < 0) return null;
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (ch === "\\") {
      escaped = true;
      continue;
    }
    if (ch === '"') inString = !inString;
    if (inString) continue;
    if (ch === "{") depth += 1;
    if (ch === "}") {
      depth -= 1;
      if (depth === 0) return text.slice(start, i + 1);
    }
  }
  return null;
}


/**
 * Lightweight JSON-Schema subset validation: checks required fields and
 * primitive types (string/number/boolean/array/object). Returns a list of
 * human-readable issues; empty list = valid.
 */
export function validateAgainstSchema(
  value: Record<string, unknown>,
  schema: Record<string, unknown>,
): string[] {
  const issues: string[] = [];
  const properties = (schema.properties ?? {}) as Record<string, { type?: string }>;
  const required = Array.isArray(schema.required) ? (schema.required as string[]) : [];
  for (const field of required) {
    if (!(field in value) || value[field] === undefined || value[field] === null || value[field] === "") {
      issues.push(`missing required field "${field}"`);
    }
  }
  for (const [field, type] of Object.entries(properties)) {
    if (!(field in value) || value[field] === undefined) continue;
    const actual = value[field];
    const expected = type?.type;
    if (!expected) continue;
    const mismatches: Record<string, (v: unknown) => boolean> = {
      string: (v) => typeof v !== "string",
      number: (v) => typeof v !== "number",
      boolean: (v) => typeof v !== "boolean",
      array: (v) => !Array.isArray(v),
      object: (v) => typeof v !== "object" || Array.isArray(v) || v === null,
    };
    const check = mismatches[expected];
    if (check && check(actual)) {
      issues.push(`field "${field}" should be ${expected}`);
    }
  }
  return issues;
}

function mergeUsageClone(a: DeepSeekUsage, b: DeepSeekUsage): DeepSeekUsage {
  const merged: DeepSeekUsage = { ...a };
  mergeDeepSeekUsage(merged, b);
  return merged;
}
