"use client";

import { useEffect, useState } from "react";
import { sampleDemoQuestions } from "@/lib/demoQuestions";

interface QuickQuestionsProps {
  onSend: (question: string) => Promise<void>;
}

export function QuickQuestions({ onSend }: QuickQuestionsProps) {
  const [questions, setQuestions] = useState<string[]>([]);

  useEffect(() => {
    // Client-only by design: Math.random() must not run during SSR, or hydration mismatches.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setQuestions(sampleDemoQuestions());
  }, []);

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
