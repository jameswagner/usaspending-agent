import type { Citation, ToolCitation } from "@/lib/types";

interface CitationsProps {
  citations: Citation[];
  toolCitations: ToolCitation[];
}

export function Citations({ citations, toolCitations }: CitationsProps) {
  return (
    <details className="mt-4 border-t border-black/10 pt-3 text-sm text-black/70 dark:border-white/10 dark:text-white/70">
      <summary className="cursor-pointer text-xs font-semibold tracking-wide text-black/50 uppercase dark:text-white/50">
        Sources
      </summary>
      <ul className="mt-1.5 space-y-1.5">
        {citations.map((citation, i) => (
          <li key={`guide-${i}`}>
            <GuideCitation citation={citation} />
          </li>
        ))}
        {toolCitations.map((citation, i) => (
          <li key={`tool-${i}`}>
            <ToolCitationItem citation={citation} />
          </li>
        ))}
      </ul>
    </details>
  );
}

// Glossary citations carry a term, linked to the live glossary sidebar's
// per-term deep link. Guide citations carry either a question (the
// common case - Q&A-shaped chunks) or a page (the minority that aren't
// Q&A-shaped) - see Citation in response_shaping.py.
function GuideCitation({ citation }: { citation: Citation }) {
  const label = citation.term
    ? `${citation.source}: ${citation.term}`
    : citation.question
      ? `${citation.source}: "${citation.question}"`
      : `${citation.source}, page ${citation.page}`;

  if (citation.url) {
    return (
      <a href={citation.url} target="_blank" rel="noopener noreferrer" className="underline hover:no-underline">
        {label}
      </a>
    );
  }
  return <>{label}</>;
}

function ToolCitationItem({ citation }: { citation: ToolCitation }) {
  return (
    <div>
      {citation.url ? (
        <a href={citation.url} target="_blank" rel="noopener noreferrer" className="underline hover:no-underline">
          {citation.description}
        </a>
      ) : (
        citation.description
      )}
      {/* curl is a reproducible command, not a clickable link - shown as a copyable code block. */}
      {citation.curl && (
        <pre className="mt-1 overflow-x-auto rounded bg-black/5 px-2 py-1.5 text-xs whitespace-pre-wrap break-all dark:bg-white/10">
          {citation.curl}
        </pre>
      )}
    </div>
  );
}
