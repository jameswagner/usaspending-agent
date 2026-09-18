"use client";

import { useConversation } from "@/hooks/useConversation";
import { useGuidedFlow } from "@/hooks/useGuidedFlow";
import { GuidedFlowCard } from "./GuidedFlowCard";
import { MessageList } from "./MessageList";
import { MessageInput } from "./MessageInput";

export function ChatWindow() {
  const { turns, loading, error, sendMessage, newConversation } = useConversation();
  const guidedFlow = useGuidedFlow();

  // While a flow is active (and not mid-escape), every message - a form
  // value, an aside, or a tangent - goes through the same input box and
  // the same classifier, rather than a separate "flow mode" input. A
  // tangent comes back as a forward question, which then goes through the
  // normal chat pipeline exactly as if the user had typed it directly.
  async function handleSend(question: string) {
    if (guidedFlow.active && !guidedFlow.paused) {
      const forwardQuestion = await guidedFlow.handleMessage(question);
      if (forwardQuestion) {
        await sendMessage(forwardQuestion);
      }
      return;
    }
    await sendMessage(question);
  }

  function handleNewConversation() {
    guidedFlow.dismiss();
    newConversation();
  }

  return (
    <div className="flex h-dvh flex-col">
      <header className="border-b border-black/10 px-4 py-3 dark:border-white/10">
        <h1 className="text-lg font-semibold">USASpending assistant</h1>
        <p className="text-sm text-black/60 dark:text-white/60">
          Ask about federal spending: definitions from the Analyst&apos;s Guide or Glossary, or live numbers from
          USASpending.gov.
        </p>
      </header>
      {!guidedFlow.active && (
        <div className="border-b border-black/10 px-4 py-2 dark:border-white/10">
          <button
            type="button"
            onClick={() => void guidedFlow.start()}
            className="rounded-md border border-black/15 px-3 py-1.5 text-sm text-black/70 hover:bg-black/5 dark:border-white/15 dark:text-white/70 dark:hover:bg-white/10"
          >
            Find spending in your area
          </button>
        </div>
      )}
      {guidedFlow.active && (
        <div className="border-b border-black/10 px-4 py-3 dark:border-white/10">
          <GuidedFlowCard flow={guidedFlow} onEscape={(question) => void sendMessage(question)} />
        </div>
      )}
      <MessageList turns={turns} error={error} onSend={handleSend} />
      <MessageInput onSend={handleSend} onNewConversation={handleNewConversation} loading={loading} />
    </div>
  );
}
