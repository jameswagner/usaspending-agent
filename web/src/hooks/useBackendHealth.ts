"use client";

import { useEffect, useState } from "react";

// Railway's sleep mode means the first request after idle can take well
// over a typical request timeout while the container cold-starts - this
// polls /api/health (which blocks on FastAPI's own /health, itself gated
// on warm_up() finishing - see that route's comment) until it succeeds,
// so callers can gate Send/QuickQuestions on a real "backend is warm"
// signal instead of racing the user's first question against the cold
// start.
const HEALTH_RETRY_DELAY_MS = 3_000;

export function useBackendHealth(): boolean {
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let retryTimeout: ReturnType<typeof setTimeout>;

    async function check() {
      try {
        const resp = await fetch("/api/health");
        if (!cancelled && resp.ok) {
          setReady(true);
          return;
        }
      } catch {
        // Network error (e.g. Railway still provisioning) - fall through to retry.
      }
      if (!cancelled) {
        retryTimeout = setTimeout(check, HEALTH_RETRY_DELAY_MS);
      }
    }

    void check();
    return () => {
      cancelled = true;
      clearTimeout(retryTimeout);
    };
  }, []);

  return ready;
}
