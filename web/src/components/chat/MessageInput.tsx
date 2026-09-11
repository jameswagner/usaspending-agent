"use client";

import { useRef, useState } from "react";

interface MessageInputProps {
  onSend: (question: string) => Promise<void>;
  onNewConversation: () => void;
  loading: boolean;
}

export function MessageInput({ onSend, onNewConversation, loading }: MessageInputProps) {
  const [question, setQuestion] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Grows with content up to max-h-40 (matches the class below) instead of
  // staying a fixed one-line box that scrolls its own text internally.
  function resize(el: HTMLTextAreaElement) {
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }

  async function submit() {
    const trimmed = question.trim();
    if (!trimmed || loading) return;
    setQuestion("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    await onSend(trimmed);
  }

  return (
    <form
      onSubmit={(e: React.SubmitEvent) => {
        e.preventDefault();
        void submit();
      }}
      className="flex items-center gap-2 border-t border-black/10 px-4 py-3 dark:border-white/10"
    >
      <textarea
        ref={textareaRef}
        value={question}
        onChange={(e) => {
          setQuestion(e.target.value);
          resize(e.target);
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            void submit();
          }
        }}
        placeholder="e.g. What is a sub-award? / How is NSF spending broken down by NAICS code for FY2024?"
        rows={2}
        autoFocus
        className="max-h-40 flex-1 resize-none overflow-y-auto rounded-md border border-black/15 px-3 py-2 text-sm outline-none focus:border-black/40 dark:border-white/15 dark:focus:border-white/40"
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
