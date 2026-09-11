"use client";

import { useEffect, useRef } from "react";
import type { ConversationTurn } from "@/lib/types";
import { MessageBubble } from "./MessageBubble";

interface MessageListProps {
  turns: ConversationTurn[];
  error: string | null;
}

export function MessageList({ turns, error }: MessageListProps) {
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
          <p className="text-sm text-black/50 dark:text-white/50">
            e.g. What is a sub-award? / How is NSF spending broken down by NAICS code for FY2024?
          </p>
        )}
        {turns.map((turn, i) => (
          <MessageBubble key={i} turn={turn} />
        ))}
        {error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
