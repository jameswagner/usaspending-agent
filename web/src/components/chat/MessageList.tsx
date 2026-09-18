"use client";

import dynamic from "next/dynamic";
import { useEffect, useRef } from "react";
import type { ConversationTurn } from "@/lib/types";
import { MessageBubble } from "./MessageBubble";

// QuickQuestions shuffles with Math.random() on every render - rendered
// during SSR, that produces a different order than the client's hydration
// pass, which is a guaranteed hydration mismatch (React can't reconcile
// text content that legitimately differs between the two passes). ssr:
// false skips the server render entirely for this component, so it only
// ever renders client-side, after mount - no mismatch is possible because
// there's no server-rendered version to compare against.
const QuickQuestions = dynamic(() => import("./QuickQuestions").then((m) => m.QuickQuestions), { ssr: false });

interface MessageListProps {
  turns: ConversationTurn[];
  error: string | null;
  onSend: (question: string) => Promise<void>;
}

export function MessageList({ turns, error, onSend }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Ref-based bottom scroll, not flex-direction: column-reverse - reverse
  // flow inverts scroll-wheel/scrollbar direction, a real UX quirk this
  // avoids.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4">
      <div className="mx-auto flex max-w-3xl flex-col gap-6">
        {turns.length === 0 && (
          <div className="flex flex-col gap-3">
            <p className="text-sm text-black/50 dark:text-white/50">
              e.g. What is a sub-award? / How is NSF spending broken down by NAICS code for FY2024?
            </p>
            <QuickQuestions onSend={onSend} />
          </div>
        )}
        {turns.map((turn) => (
          <MessageBubble key={turn.id} turn={turn} />
        ))}
        {error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
