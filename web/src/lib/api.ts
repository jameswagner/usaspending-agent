import type { AskResponse } from "./types";

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
