"use client";

import { useState } from "react";
import { sampleDemoQuestions } from "@/lib/demoQuestions";

interface QuickQuestionsProps {
  onSend: (question: string) => Promise<void>;
}

// sampleDemoQuestions() shuffles with Math.random() - imported with
// { ssr: false } (see MessageList.tsx) so this never renders during SSR,
// only after mount, avoiding the hydration mismatch a random order would
// otherwise cause between the server's render and the client's.
export function QuickQuestions({ onSend }: QuickQuestionsProps) {
  const [questions] = useState(() => sampleDemoQuestions());

  return (
    <div className="flex flex-wrap gap-2">
      {questions.map((question) => (
        <button
          key={question}
          type="button"
          onClick={() => void onSend(question)}
          className="rounded-md border border-black/15 px-3 py-1.5 text-left text-sm text-black/70 hover:bg-black/5 dark:border-white/15 dark:text-white/70 dark:hover:bg-white/10"
        >
          {question}
        </button>
      ))}
    </div>
  );
}
