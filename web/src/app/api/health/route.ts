import { NextResponse } from "next/server";

// Same server-side proxy pattern as ../ask/route.ts. Used by
// useBackendHealth to block Send/QuickQuestions until FastAPI's own
// /health - which only returns 200 once warm_up() has finished loading
// retrievers/models/clients (see backend/app/main.py's lifespan hook) -
// resolves, so a Railway cold start doesn't time out the user's first
// real question.
const FASTAPI_BASE_URL = process.env.FASTAPI_BASE_URL ?? "http://localhost:8000";

export async function GET() {
  try {
    const resp = await fetch(`${FASTAPI_BASE_URL}/health`);
    const data = await resp.json();
    return NextResponse.json(data, { status: resp.status });
  } catch {
    return NextResponse.json({ status: "unreachable" }, { status: 503 });
  }
}
