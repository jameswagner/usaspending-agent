"use client";

import { useState } from "react";

interface MessageInputProps {
  onSend: (question: string) => Promise<void>;
  onNewConversation: () => void;
  loading: boolean;
}

export function MessageInput({ onSend, onNewConversation, loading }: MessageInputProps) {
  const [question, setQuestion] = useState("");

  async function submit() {
    const trimmed = question.trim();
    if (!trimmed || loading) return;
    setQuestion("");
    await onSend(trimmed);
  }

  return (
    <form
      onSubmit={(e: React.SubmitEvent) => {
        e.preventDefault();
        void submit();
      }}
      className="flex items-end gap-2 border-t border-black/10 px-4 py-3 dark:border-white/10"
    >
      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            void submit();
          }
        }}
        placeholder="e.g. What is a sub-award? / How is NSF spending broken down by NAICS code for FY2024?"
        rows={1}
        autoFocus
        className="max-h-40 flex-1 resize-none rounded-md border border-black/15 px-3 py-2 text-sm outline-none focus:border-black/40 dark:border-white/15 dark:focus:border-white/40"
      />
      <button
        type="submit"
        disabled={loading || !question.trim()}
        className="shrink-0 rounded-md bg-black px-4 py-2 text-sm text-white disabled:opacity-50 dark:bg-white dark:text-black"
      >
        Ask
      </button>
      <button
        type="button"
        onClick={onNewConversation}
        className="shrink-0 rounded-md border border-black/15 px-3 py-2 text-sm text-black/70 hover:bg-black/5 dark:border-white/15 dark:text-white/70 dark:hover:bg-white/10"
      >
        New conversation
      </button>
    </form>
  );
}
