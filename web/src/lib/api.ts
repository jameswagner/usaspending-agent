import type { AskResponse } from "./types";

// One tool-status event from POST /api/ask/stream's SSE body - see
// backend/app/agent/streaming.py's event protocol. Named "tool_name" (not
// camelCase) to match the wire payload verbatim, so askQuestionStream can
// spread `data` straight into this shape with no field renaming.
export type ToolStreamEvent =
  | { type: "tool_call_start"; tool_name: string; args: Record<string, unknown> }
  | { type: "tool_result"; tool_name: string; summary: string }
  | { type: "tool_error"; tool_name: string; message: string };

// No event of a given frame beyond "keep-alive" comment (`: keep-alive`,
// no event:/data: lines) - parseFrame returns null for those, and the
// idle timer below still gets reset on receiving one, since a keep-alive
// is itself proof the connection is alive.
function parseSSEFrame(frame: string): { eventName: string; data: unknown } | null {
  let eventName = "message";
  const dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith(":")) continue;
    if (line.startsWith("event: ")) {
      eventName = line.slice("event: ".length);
    } else if (line.startsWith("data: ")) {
      dataLines.push(line.slice("data: ".length));
    }
  }
  if (dataLines.length === 0) return null;
  return { eventName, data: JSON.parse(dataLines.join("\n")) };
}

// A long multi-tool-call turn with periodic keep-alives shouldn't hit a
// flat wall-clock cap the way a fully-silent unary request should (see
// REQUEST_TIMEOUT_MS below) - this is an IDLE timeout instead, reset on
// every frame received (including keep-alives), well over the backend's
// own 15s keep-alive interval so it only fires on a genuine hang.
const STREAM_IDLE_TIMEOUT_MS = 30_000;

// Calls this Next.js app's own /api/ask/stream route (never FastAPI
// directly - see app/api/ask/stream/route.ts), consuming the SSE body via
// fetch + a stream reader rather than EventSource, since EventSource has
// no way to send a POST body. onEvent fires for each tool-status frame as
// it arrives; the returned promise resolves with the same AskResponse
// shape askQuestion returns, once the server's `done` frame arrives.
export async function askQuestionStream(
  question: string,
  conversationId: string | null,
  onEvent: (event: ToolStreamEvent) => void,
  { signal: externalSignal }: { signal?: AbortSignal } = {}
): Promise<AskResponse> {
  const controller = new AbortController();
  const onExternalAbort = () => controller.abort();
  externalSignal?.addEventListener("abort", onExternalAbort);

  let idleTimeout: ReturnType<typeof setTimeout> | undefined;
  const resetIdleTimer = () => {
    clearTimeout(idleTimeout);
    idleTimeout = setTimeout(() => controller.abort(), STREAM_IDLE_TIMEOUT_MS);
  };

  try {
    resetIdleTimer();
    const resp = await fetch("/api/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, conversation_id: conversationId }),
      signal: controller.signal,
    });
    if (!resp.ok || !resp.body) {
      throw new Error(`Server error: ${resp.status}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      resetIdleTimer();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let frameEnd: number;
      while ((frameEnd = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, frameEnd);
        buffer = buffer.slice(frameEnd + 2);
        const parsed = parseSSEFrame(frame);
        if (!parsed) continue;
        const { eventName, data } = parsed;

        if (eventName === "done") {
          return data as AskResponse;
        }
        if (eventName === "error") {
          const message = (data as { message?: string })?.message ?? "The server reported an error.";
          throw new Error(message);
        }
        if (eventName === "tool_call_start" || eventName === "tool_result" || eventName === "tool_error") {
          onEvent({ type: eventName, ...(data as object) } as ToolStreamEvent);
        }
      }
    }
    throw new Error("Stream ended without a final answer.");
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") {
      throw new Error("Request timed out - the server may be unresponsive.");
    }
    throw err;
  } finally {
    clearTimeout(idleTimeout);
    externalSignal?.removeEventListener("abort", onExternalAbort);
  }
}

// A hung request (dead dev server, dropped connection, anything) must
// still settle the fetch promise - without a bound, sendMessage's loading
// state stays true forever with no way for the UI to recover except a
// page reload. Well above any real agent turn's latency (multiple tool
// calls plus retrieval), so this only fires on a genuine hang.
const REQUEST_TIMEOUT_MS = 90_000;

// Calls this Next.js app's own /api/ask route, never FastAPI directly -
// see app/api/ask/route.ts for why (the proxy/BFF pattern).
export async function askQuestion(
  question: string,
  conversationId: string | null
): Promise<AskResponse> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const resp = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, conversation_id: conversationId }),
      signal: controller.signal,
    });
    if (!resp.ok) {
      throw new Error(`Server error: ${resp.status}`);
    }
    return (await resp.json()) as AskResponse;
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") {
      throw new Error("Request timed out - the server may be unresponsive.");
    }
    throw err;
  } finally {
    clearTimeout(timeout);
  }
}
