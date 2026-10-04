import { NextRequest, NextResponse } from "next/server";

// Proxies to FastAPI's POST /ask/download, same pattern as ../route.ts.
const FASTAPI_BASE_URL = process.env.FASTAPI_BASE_URL ?? "http://localhost:8000";

export async function POST(request: NextRequest) {
  const body = await request.json();

  const resp = await fetch(`${FASTAPI_BASE_URL}/ask/download`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const data = await resp.json();
  return NextResponse.json(data, { status: resp.status });
}
