"use client";

import { useCallback, useRef, useState } from "react";
import { guidedFlowStep } from "@/lib/api";
import type { GuidedFlowFields, GuidedFlowStepResponse } from "@/lib/types";

export function useGuidedFlow() {
  const [active, setActive] = useState(false);
  const [paused, setPaused] = useState(false);
  const [fields, setFields] = useState<GuidedFlowFields>({});
  const [prompt, setPrompt] = useState<string | null>(null);
  const [asideAnswer, setAsideAnswer] = useState<string | null>(null);
  const [result, setResult] = useState<GuidedFlowStepResponse | null>(null);
  const [loading, setLoading] = useState(false);
  // Independent of useConversation's own conversation_id - this flow talks
  // to a separate backend endpoint/state store keyed by its own id, not the
  // main chat's LangGraph thread. An escaped question therefore starts a
  // fresh chat conversation rather than inheriting this flow's fields as
  // context - a known v1 simplification, not an oversight.
  const conversationIdRef = useRef<string | null>(null);

  const applyResponse = useCallback((data: GuidedFlowStepResponse) => {
    setFields(data.fields ?? {});
    if (data.status === "result" || data.status === "breakdown") {
      setResult(data);
      setAsideAnswer(null);
      setPaused(false);
    } else {
      // "collecting"
      setResult(null);
      setPrompt(data.prompt);
      setAsideAnswer(null);
      setPaused(false);
    }
  }, []);

  const start = useCallback(async () => {
    conversationIdRef.current = crypto.randomUUID();
    setActive(true);
    setPaused(false);
    setResult(null);
    setAsideAnswer(null);
    setLoading(true);
    try {
      applyResponse(await guidedFlowStep(conversationIdRef.current, null));
    } finally {
      setLoading(false);
    }
  }, [applyResponse]);

  // Returns the forward question when the message is a tangent, so
  // ChatWindow can hand it to the normal chat pipeline; null otherwise.
  const handleMessage = useCallback(
    async (message: string): Promise<string | null> => {
      if (!conversationIdRef.current) return null;
      setLoading(true);
      try {
        const data = await guidedFlowStep(conversationIdRef.current, message);
        if (data.status === "escaped") {
          setPaused(true);
          return data.forward_question ?? message;
        }
        if (data.status === "aside_answered") {
          setAsideAnswer(data.answer);
          setPrompt(data.prompt);
          setFields(data.fields ?? {});
          return null;
        }
        applyResponse(data);
        return null;
      } finally {
        setLoading(false);
      }
    },
    [applyResponse]
  );

  const resume = useCallback(async () => {
    if (!conversationIdRef.current) return;
    setLoading(true);
    try {
      applyResponse(await guidedFlowStep(conversationIdRef.current, null));
    } finally {
      setLoading(false);
    }
  }, [applyResponse]);

  const dismiss = useCallback(() => {
    setActive(false);
    setPaused(false);
    setResult(null);
    setPrompt(null);
    setAsideAnswer(null);
    setFields({});
    conversationIdRef.current = null;
  }, []);

  return { active, paused, fields, prompt, asideAnswer, result, loading, start, handleMessage, resume, dismiss };
}
