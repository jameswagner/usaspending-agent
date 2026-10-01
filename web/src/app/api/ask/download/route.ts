import { NextRequest, NextResponse } from "next/server";

// Same server-side proxy pattern as ../route.ts - the "Download this" follow-up
// button's click path, forwarded to FastAPI's POST /ask/download.
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
