"use client";

import { useConversation } from "@/hooks/useConversation";
import { MessageList } from "./MessageList";
import { MessageInput } from "./MessageInput";

export function ChatWindow() {
  const { turns, loading, error, sendMessage, newConversation } = useConversation();

  return (
    <div className="flex h-dvh flex-col">
      <header className="border-b border-black/10 px-4 py-3 dark:border-white/10">
        <h1 className="text-lg font-semibold">USASpending RAG assistant</h1>
        <p className="text-sm text-black/60 dark:text-white/60">
          Ask about federal spending — definitions from the Analyst&apos;s Guide, or live numbers from
          USASpending.gov.
        </p>
      </header>
      <MessageList turns={turns} loading={loading} error={error} />
      <MessageInput onSend={sendMessage} onNewConversation={newConversation} loading={loading} />
    </div>
  );
}
