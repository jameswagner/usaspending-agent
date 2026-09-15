import { NextRequest } from "next/server";

// Same server-side proxy pattern as ../route.ts, streaming instead of
// buffering: FastAPI's /ask/stream response body is already a
// ReadableStream (from fetch), so this pipes it straight through rather
// than awaiting/re-serializing it.
const FASTAPI_BASE_URL = process.env.FASTAPI_BASE_URL ?? "http://localhost:8000";

export async function POST(request: NextRequest) {
  const body = await request.json();

  const resp = await fetch(`${FASTAPI_BASE_URL}/ask/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  return new Response(resp.body, {
    status: resp.status,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
