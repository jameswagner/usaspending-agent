"use client";

import { useState } from "react";
import { useConversation } from "@/hooks/useConversation";

// Bare placeholder - proves the proxy round-trip works end-to-end
// (including conversation_id continuity across turns). The real chat UI
// replaces this in part 2 of the Next.js migration.
export default function Home() {
  const { conversationId, turns, loading, error, sendMessage, newConversation } = useConversation();
  const [question, setQuestion] = useState("");

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!question.trim()) return;
    await sendMessage(question);
    setQuestion("");
  }

  return (
    <main style={{ padding: 24, fontFamily: "monospace" }}>
      <h1>USASpending RAG - plumbing check</h1>
      <form onSubmit={handleSubmit}>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question"
          style={{ width: 400 }}
        />
        <button type="submit" disabled={loading}>
          Send
        </button>
        <button type="button" onClick={newConversation}>
          New conversation
        </button>
      </form>
      {error && <p style={{ color: "red" }}>{error}</p>}
      <p>conversation_id: {conversationId ?? "(none yet)"}</p>
      <pre style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(turns, null, 2)}</pre>
    </main>
  );
}
