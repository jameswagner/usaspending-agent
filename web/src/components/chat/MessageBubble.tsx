import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ConversationTurn, FollowUp } from "@/lib/types";
import { Citations } from "./Citations";
import { ChartBlock } from "./ChartBlock";
import { DownloadBlock } from "./DownloadBlock";

interface MessageBubbleProps {
  turn: ConversationTurn;
  onDownload: (followUp: FollowUp) => Promise<void>;
}

function statusLine(status: ConversationTurn["status"]): { text: string; isError: boolean } {
  switch (status.kind) {
    case "tool_call":
      return { text: `Calling ${status.toolName}…`, isError: false };
    case "tool_result":
      return { text: `${status.toolName}: ${status.summary}`, isError: false };
    case "tool_error":
      return { text: `${status.toolName}: ${status.message}`, isError: true };
    case "pending":
    case "done":
    default:
      return { text: "Thinking…", isError: false };
  }
}

export function MessageBubble({ turn, onDownload }: MessageBubbleProps) {
  const { question, response, status } = turn;
  // Local: disables only this bubble's button while its click is in flight.
  const [requestingDownload, setRequestingDownload] = useState(false);

  const handleFollowUpClick = async (followUp: FollowUp) => {
    setRequestingDownload(true);
    try {
      await onDownload(followUp);
    } finally {
      setRequestingDownload(false);
    }
  };

  if (!response) {
    const { text, isError } = statusLine(status);
    return (
      <div className="border-b border-black/10 pb-6 last:border-none dark:border-white/10">
        <p className="font-semibold">{question}</p>
        <p
          className={
            isError
              ? "mt-2 flex items-start gap-2 text-sm text-amber-600 dark:text-amber-400"
              : "mt-2 flex items-start gap-2 text-sm text-black/50 dark:text-white/50"
          }
        >
          {/* A raw tool-result summary (e.g. a wall of category/amount
              pairs) can look enough like a finished answer that this dot
              is the only thing telling a user the turn isn't done yet -
              status text alone wasn't a strong enough signal in practice. */}
          {!isError && (
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
      {response.downloads.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-3">
          {response.downloads.map((download, i) => (
            <DownloadBlock key={i} download={download} />
          ))}
        </div>
      )}
      {response.follow_ups.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-2">
          {response.follow_ups.map((followUp, i) => (
            <button
              key={i}
              type="button"
              onClick={() => handleFollowUpClick(followUp)}
              disabled={requestingDownload}
              className="rounded-full border border-black/10 px-3 py-1 text-xs font-medium hover:bg-black/5 disabled:opacity-50 dark:border-white/10 dark:hover:bg-white/10"
              style={{ color: "var(--chart-ink)" }}
            >
              {requestingDownload ? "Preparing download…" : followUp.label}
            </button>
          ))}
        </div>
      )}
      {hasCitations && <Citations citations={response.citations} toolCitations={response.tool_citations} />}
    </div>
  );
}
