import { NextRequest, NextResponse } from "next/server";

// Same server-side proxy pattern as /api/ask - see that route for why.
const FASTAPI_BASE_URL = process.env.FASTAPI_BASE_URL ?? "http://localhost:8000";

export async function POST(request: NextRequest) {
  const body = await request.json();

  const resp = await fetch(`${FASTAPI_BASE_URL}/guided-flow/community-spending/step`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const data = await resp.json();
  return NextResponse.json(data, { status: resp.status });
}
