"use client";

import { useCallback, useState } from "react";
import { askQuestion } from "@/lib/api";
import type { ConversationTurn } from "@/lib/types";

export function useConversation() {
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [turns, setTurns] = useState<ConversationTurn[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendMessage = useCallback(
    async (question: string) => {
      setLoading(true);
      setError(null);
      const pendingTurn: ConversationTurn = { question, response: null };
      setTurns((prev) => [...prev, pendingTurn]);
      try {
        const response = await askQuestion(question, conversationId);
        setConversationId(response.conversation_id);
        setTurns((prev) => prev.map((turn) => (turn === pendingTurn ? { question, response } : turn)));
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
        setTurns((prev) => prev.filter((turn) => turn !== pendingTurn));
      } finally {
        setLoading(false);
      }
    },
    [conversationId]
  );

  const newConversation = useCallback(() => {
    setConversationId(null);
    setTurns([]);
    setError(null);
  }, []);

  return { conversationId, turns, loading, error, sendMessage, newConversation };
}
