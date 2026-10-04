/**
 * Harness multi-turn memory.
 *
 * The native harness is stateless across turns: OpenClaw owns the transcript,
 * so the harness must rebuild DeepSeek-visible history from the session file
 * before each attempt. This module parses the OpenClaw JSONL transcript,
 * extracts user / assistant / tool-result turns, and emits DeepSeek chat
 * messages honoring the reasoning_content replay contract.
 */
import type { DeepSeekChatMessage } from "./types.js";

export interface RebuiltHistory {
  messages: DeepSeekChatMessage[];
  /** Number of transcript lines that contributed. */
  linesUsed: number;
  /** Turns skipped (system noise, media-only, malformed). */
  linesSkipped: number;
}

interface TranscriptLine {
  role?: string;
  content?: unknown;
  toolName?: string;
  toolCallId?: string;
  isError?: boolean;
  timestamp?: number;
}

/**
 * Rebuilds DeepSeek-visible history from an OpenClaw session JSONL file.
 * Keeps the last `maxTurns` user/assistant exchanges and their tool results.
 */
export async function rebuildHistoryFromSessionFile(
  sessionFile: string | undefined,
  options: { maxTurns?: number } = {},
): Promise<RebuiltHistory> {
  if (!sessionFile) return { messages: [], linesUsed: 0, linesSkipped: 0 };
  const { readFile } = await import("node:fs/promises");
  let raw: string;
  try {
    raw = await readFile(sessionFile, "utf8");
  } catch {
    // Missing/unreadable session file: treat as a fresh conversation. The
    // harness is stateless by design; memory is an enhancement, not a hard
    // dependency.
    return { messages: [], linesUsed: 0, linesSkipped: 0 };
  }
  return rebuildHistoryFromJsonl(raw, options);
}

export function rebuildHistoryFromJsonl(
  jsonl: string,
  options: { maxTurns?: number } = {},
): RebuiltHistory {
  const maxTurns = options.maxTurns ?? 12;
  const messages: DeepSeekChatMessage[] = [];
  let linesUsed = 0;
  let linesSkipped = 0;

  const parsed: Array<{ line: TranscriptLine; keep: boolean }> = [];
  for (const line of jsonl.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    let entry: TranscriptLine;
    try {
      entry = JSON.parse(trimmed) as TranscriptLine;
    } catch {
      // Malformed transcript line: skip it rather than abort history rebuild.
      linesSkipped += 1;
      continue;
    }
    if (!entry || typeof entry.role !== "string") {
      linesSkipped += 1;
      continue;
    }
    if (entry.role === "user") {
      parsed.push({ line: entry, keep: true });
      linesUsed += 1;
    } else if (entry.role === "assistant") {
      parsed.push({ line: entry, keep: true });
      linesUsed += 1;
    } else if (entry.role === "toolResult") {
      parsed.push({ line: entry, keep: true });
      linesUsed += 1;
    } else {
      linesSkipped += 1;
    }
  }

  // Keep only the trailing window of user turns (an exchange = user + everything until next user).
  const userTurnIndexes = parsed.map((p, i) => (p.line.role === "user" ? i : -1)).filter((i) => i >= 0);
  const firstKeptUser = userTurnIndexes.length > maxTurns ? userTurnIndexes[userTurnIndexes.length - maxTurns] : 0;
  const window = parsed.slice(firstKeptUser);

  for (const { line } of window) {
    if (line.role === "user") {
      const text = flattenContent(line.content);
      if (text) messages.push({ role: "user", content: text });
    } else if (line.role === "assistant") {
      const text = flattenContent(line.content);
      // Assistant turns with only thinking blocks still replay as empty-content
      // turns when they carried tool calls; plain text turns skip when empty.
      const toolCalls = extractToolCalls(line.content);
      if (text || toolCalls.length > 0) {
        const msg: DeepSeekChatMessage = { role: "assistant", content: text || null };
        if (toolCalls.length > 0) {
          msg.tool_calls = toolCalls.map((c) => ({
            id: c.id,
            type: "function" as const,
            function: { name: c.name, arguments: JSON.stringify(c.arguments ?? {}) },
          }));
          // DeepSeek contract: replayed assistant tool-call turns keep
          // reasoning_content; a missing field on thinking models returns 400.
          const reasoning = extractThinking(line.content);
          if (reasoning) msg.reasoning_content = reasoning;
        }
        messages.push(msg);
      }
    } else if (line.role === "toolResult") {
      const text = flattenContent(line.content) || "(tool result without text)";
      messages.push({
        role: "tool",
        tool_call_id: line.toolCallId ?? "unknown",
        name: line.toolName ?? "unknown",
        content: text,
      });
    }
  }

  return { messages, linesUsed, linesSkipped };
}

interface TranscriptBlock {
  type?: string;
  text?: string;
  thinking?: string;
  id?: string;
  name?: string;
  arguments?: Record<string, unknown>;
}

function flattenContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((block) => {
        if (typeof block === "string") return block;
        if (isBlock(block, "text")) return block.text ?? "";
        return "";
      })
      .filter(Boolean)
      .join("\n");
  }
  return "";
}

function isBlock(value: unknown, type: string): value is TranscriptBlock & { type: string } {
  return (
    value !== null &&
    typeof value === "object" &&
    (value as TranscriptBlock).type === type
  );
}

function extractToolCalls(content: unknown): Array<{ id: string; name: string; arguments?: Record<string, unknown> }> {
  if (!Array.isArray(content)) return [];
  return content
    .filter((block): block is TranscriptBlock & { type: "toolCall" } => isBlock(block, "toolCall"))
    .map((block) => ({
      id: String(block.id ?? ""),
      name: String(block.name ?? ""),
      arguments: block.arguments,
    }))
    .filter((c) => c.id && c.name);
}

function extractThinking(content: unknown): string {
  if (!Array.isArray(content)) return "";
  return content
    .filter((block): block is TranscriptBlock & { type: "thinking" } => isBlock(block, "thinking"))
    .map((block) => block.thinking ?? "")
    .filter(Boolean)
    .join("\n");
}
