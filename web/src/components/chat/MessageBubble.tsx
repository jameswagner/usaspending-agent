import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ConversationTurn } from "@/lib/types";
import { Citations } from "./Citations";
import { ChartBlock } from "./ChartBlock";

interface MessageBubbleProps {
  turn: ConversationTurn;
}

type StatusTone = "normal" | "error" | "cancelled";

function statusLine(status: ConversationTurn["status"]): { text: string; tone: StatusTone } {
  switch (status.kind) {
    case "tool_call":
      return { text: `Calling ${status.toolName}…`, tone: "normal" };
    case "tool_result":
      return { text: `${status.toolName}: ${status.summary}`, tone: "normal" };
    case "tool_error":
      return { text: `${status.toolName}: ${status.message}`, tone: "error" };
    case "cancelled":
      return { text: "Cancelled", tone: "cancelled" };
    case "pending":
    case "done":
    default:
      return { text: "Thinking…", tone: "normal" };
  }
}

export function MessageBubble({ turn }: MessageBubbleProps) {
  const { question, response, status } = turn;

  if (!response) {
    const { text, tone } = statusLine(status);
    return (
      <div className="border-b border-black/10 pb-6 last:border-none dark:border-white/10">
        <p className="font-semibold">{question}</p>
        <p
          className={
            tone === "error"
              ? "mt-2 flex items-start gap-2 text-sm text-amber-600 dark:text-amber-400"
              : "mt-2 flex items-start gap-2 text-sm text-black/50 dark:text-white/50"
          }
        >
          {/* A raw tool-result summary (e.g. a wall of category/amount
              pairs) can look enough like a finished answer that this dot
              is the only thing telling a user the turn isn't done yet -
              status text alone wasn't a strong enough signal in practice. */}
          {tone === "normal" && (
            <span className="mt-1 h-2 w-2 shrink-0 animate-pulse rounded-full bg-black/40 dark:bg-white/40" />
          )}
          <span className="font-mono text-xs leading-relaxed">{text}</span>
        </p>
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
