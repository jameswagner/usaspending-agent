import type { DownloadSpec } from "@/lib/types";

interface DownloadBlockProps {
  download: DownloadSpec;
}

export function DownloadBlock({ download }: DownloadBlockProps) {
  const ready = download.status === "finished";
  const failed = download.status === "failed";

  return (
    <div className="w-full max-w-xl rounded-lg border border-black/10 p-3 dark:border-white/10">
      <div className="flex items-center justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate text-sm font-medium" style={{ color: "var(--chart-ink)" }}>
            {download.file_name}
          </p>
          {download.total_rows !== null && (
            <p className="text-xs" style={{ color: "var(--chart-muted-ink)" }}>
              {download.total_rows.toLocaleString()} rows
            </p>
          )}
          {!ready && !failed && (
            <p className="text-xs" style={{ color: "var(--chart-muted-ink)" }}>
              Still generating - the link below goes live once it&apos;s ready.
            </p>
          )}
        </div>
        {failed ? (
          <span className="shrink-0 text-xs text-amber-600 dark:text-amber-400">Failed</span>
        ) : (
          <a
            href={ready ? download.url : download.status_url}
            className="shrink-0 rounded bg-black/5 px-3 py-1 text-xs font-medium hover:bg-black/10 dark:bg-white/10 dark:hover:bg-white/15"
            style={{ color: "var(--chart-ink)" }}
          >
            {ready ? "Download CSV" : "Check status"}
          </a>
        )}
      </div>
    </div>
  );
}
