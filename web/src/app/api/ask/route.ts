import { NextRequest, NextResponse } from "next/server";

// Server-side proxy to FastAPI - the browser only ever calls this route,
// never FASTAPI_BASE_URL directly. This is the seam session/auth
// middleware attaches to later, and it's why no CORS setup is needed on
// the FastAPI side: this call is server-to-server, not browser-to-server.
const FASTAPI_BASE_URL = process.env.FASTAPI_BASE_URL ?? "http://localhost:8000";

export async function POST(request: NextRequest) {
  const body = await request.json();

  const resp = await fetch(`${FASTAPI_BASE_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const data = await resp.json();
  return NextResponse.json(data, { status: resp.status });
}
