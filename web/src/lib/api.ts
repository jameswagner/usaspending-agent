import type { AskResponse } from "./types";

// Calls this Next.js app's own /api/ask route, never FastAPI directly -
// see app/api/ask/route.ts for why (the proxy/BFF pattern).
export async function askQuestion(
  question: string,
  conversationId: string | null
): Promise<AskResponse> {
  const resp = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, conversation_id: conversationId }),
  });
  if (!resp.ok) {
    throw new Error(`Server error: ${resp.status}`);
  }
  return resp.json() as Promise<AskResponse>;
}
