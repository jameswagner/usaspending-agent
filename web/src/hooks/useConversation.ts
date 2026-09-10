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
      try {
        const response = await askQuestion(question, conversationId);
        setConversationId(response.conversation_id);
        setTurns((prev) => [...prev, { question, response }]);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
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
