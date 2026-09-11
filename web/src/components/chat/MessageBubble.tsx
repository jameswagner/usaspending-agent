import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ConversationTurn } from "@/lib/types";
import { Citations } from "./Citations";
import { ChartBlock } from "./ChartBlock";

interface MessageBubbleProps {
  turn: ConversationTurn;
}

export function MessageBubble({ turn }: MessageBubbleProps) {
  const { question, response } = turn;

  if (!response) {
    return (
      <div className="border-b border-black/10 pb-6 last:border-none dark:border-white/10">
        <p className="font-semibold">{question}</p>
        <p className="mt-2 text-sm text-black/50 dark:text-white/50">Thinking…</p>
      </div>
    );
  }

  const hasCitations = response.citations.length > 0 || response.tool_citations.length > 0;

  return (
    <div className="border-b border-black/10 pb-6 last:border-none dark:border-white/10">
      <p className="font-semibold">{question}</p>
      <div className="markdown-answer mt-2 text-sm leading-relaxed">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{response.answer_text}</ReactMarkdown>
      </div>
      <span className="mt-2 inline-block rounded-full bg-black/5 px-2 py-0.5 text-xs text-black/60 dark:bg-white/10 dark:text-white/60">
        {response.source_type}
      </span>
      {response.charts.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-6">
          {response.charts.map((chart, i) => (
            <ChartBlock key={i} chart={chart} />
          ))}
        </div>
      )}
      {hasCitations && <Citations citations={response.citations} toolCitations={response.tool_citations} />}
    </div>
  );
}
