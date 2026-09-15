"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { askQuestionStream } from "@/lib/api";
import type { ConversationTurn, TurnStatus } from "@/lib/types";

function statusFromToolEvent(event: { type: string; tool_name: string; summary?: string; message?: string }): TurnStatus {
  if (event.type === "tool_call_start") {
    return { kind: "tool_call", toolName: event.tool_name };
  }
  if (event.type === "tool_result") {
    return { kind: "tool_result", toolName: event.tool_name, summary: event.summary ?? "" };
  }
  return { kind: "tool_error", toolName: event.tool_name, message: event.message ?? "This query failed." };
}

export function useConversation() {
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [turns, setTurns] = useState<ConversationTurn[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Only one turn should ever stream at a time - a second overlapping
  // request against the same conversation_id/LangGraph thread_id is worth
  // avoiding given the checkpointer's single sqlite connection (see
  // singletons.py), not just a UI nicety.
  const activeControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => {
      activeControllerRef.current?.abort();
    };
  }, []);

  const sendMessage = useCallback(
    async (question: string) => {
      activeControllerRef.current?.abort();
      const controller = new AbortController();
      activeControllerRef.current = controller;

      setLoading(true);
      setError(null);
      const turnId = crypto.randomUUID();
      const pendingTurn: ConversationTurn = { id: turnId, question, response: null, status: { kind: "pending" } };
      setTurns((prev) => [...prev, pendingTurn]);

      const updateStatus = (status: TurnStatus) => {
        setTurns((prev) => prev.map((turn) => (turn.id === turnId ? { ...turn, status } : turn)));
      };

      try {
        const response = await askQuestionStream(
          question,
          conversationId,
          (event) => updateStatus(statusFromToolEvent(event)),
          { signal: controller.signal }
        );
        setConversationId(response.conversation_id);
        setTurns((prev) =>
          prev.map((turn) => (turn.id === turnId ? { id: turnId, question, response, status: { kind: "done" } } : turn))
        );
      } catch (err) {
        if (err instanceof Error && err.name === "AbortError") {
          // Superseded by a newer sendMessage call, or the component
          // unmounted - not a real error, and the turn that triggered it
          // is gone from state already (or about to be) either way.
          return;
        }
        setError(err instanceof Error ? err.message : String(err));
        setTurns((prev) => prev.filter((turn) => turn.id !== turnId));
      } finally {
        if (activeControllerRef.current === controller) {
          setLoading(false);
          activeControllerRef.current = null;
        }
      }
    },
    [conversationId]
  );

  const newConversation = useCallback(() => {
    activeControllerRef.current?.abort();
    setConversationId(null);
    setTurns([]);
    setError(null);
  }, []);

  return { conversationId, turns, loading, error, sendMessage, newConversation };
}
