"use client";

import { useState } from "react";
import type { useGuidedFlow } from "@/hooks/useGuidedFlow";
import { Citations } from "./Citations";

// Modeled on USASpending's own "Quickstart" and "Tutorial" videos on
// finding spending to your state - see guided_flow.py's module docstring
// for both URLs and what each adds.
const GUIDE_VIDEO_URL = "https://www.youtube.com/watch?v=b9ABwzIyCNI";

const STATES: readonly [string, string][] = [
  ["AL", "Alabama"], ["AK", "Alaska"], ["AZ", "Arizona"], ["AR", "Arkansas"], ["CA", "California"],
  ["CO", "Colorado"], ["CT", "Connecticut"], ["DE", "Delaware"], ["DC", "District of Columbia"],
  ["FL", "Florida"], ["GA", "Georgia"], ["HI", "Hawaii"], ["ID", "Idaho"], ["IL", "Illinois"],
  ["IN", "Indiana"], ["IA", "Iowa"], ["KS", "Kansas"], ["KY", "Kentucky"], ["LA", "Louisiana"],
  ["ME", "Maine"], ["MD", "Maryland"], ["MA", "Massachusetts"], ["MI", "Michigan"], ["MN", "Minnesota"],
  ["MS", "Mississippi"], ["MO", "Missouri"], ["MT", "Montana"], ["NE", "Nebraska"], ["NV", "Nevada"],
  ["NH", "New Hampshire"], ["NJ", "New Jersey"], ["NM", "New Mexico"], ["NY", "New York"],
  ["NC", "North Carolina"], ["ND", "North Dakota"], ["OH", "Ohio"], ["OK", "Oklahoma"], ["OR", "Oregon"],
  ["PA", "Pennsylvania"], ["RI", "Rhode Island"], ["SC", "South Carolina"], ["SD", "South Dakota"],
  ["TN", "Tennessee"], ["TX", "Texas"], ["UT", "Utah"], ["VT", "Vermont"], ["VA", "Virginia"],
  ["WA", "Washington"], ["WV", "West Virginia"], ["WI", "Wisconsin"], ["WY", "Wyoming"],
];

function formatCurrency(value: number): string {
  return value.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
}

function NamedAmountList({ title, items }: { title: string; items: { name: string; amount: number }[] }) {
  if (items.length === 0) return null;
  return (
    <div>
      <p className="text-xs font-medium text-black/60 dark:text-white/60">{title}</p>
      <ul className="mt-0.5 space-y-0.5">
        {items.map((item, i) => (
          <li key={i} className="flex justify-between gap-2 text-xs">
            <span className="truncate">{item.name}</span>
            <span className="shrink-0 tabular-nums">{formatCurrency(item.amount)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

interface GuidedFlowCardProps {
  flow: ReturnType<typeof useGuidedFlow>;
  onEscape: (question: string) => void;
}

export function GuidedFlowCard({ flow, onEscape }: GuidedFlowCardProps) {
  const [state, setState] = useState("");
  const [district, setDistrict] = useState("");
  const [startYear, setStartYear] = useState("");
  const [endYear, setEndYear] = useState("");

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const districtClause = district.trim() ? `, congressional district ${district}` : "";
    const message = `State ${state}${districtClause}, fiscal year ${startYear} to ${endYear}`;
    const forwardQuestion = await flow.handleMessage(message);
    if (forwardQuestion) onEscape(forwardQuestion);
  }

  function reset() {
    setState("");
    setDistrict("");
    setStartYear("");
    setEndYear("");
    flow.dismiss();
  }

  if (flow.paused) {
    return (
      <div className="w-full max-w-xl rounded-md border border-black/15 p-4 text-sm dark:border-white/15">
        <p className="text-black/70 dark:text-white/70">Still want your community spending lookup?</p>
        <button
          type="button"
          onClick={() => void flow.resume()}
          className="mt-2 rounded-md border border-black/15 px-3 py-1.5 text-sm hover:bg-black/5 dark:border-white/15 dark:hover:bg-white/10"
        >
          Resume lookup
        </button>
      </div>
    );
  }

  const result = flow.result;

  return (
    <div className="w-full max-w-xl rounded-md border border-black/15 p-4 dark:border-white/15">
      <p className="text-sm font-medium">Community spending lookup</p>
      <p className="mt-0.5 text-xs text-black/50 dark:text-white/50">
        Modeled on USASpending&apos;s own{" "}
        <a href={GUIDE_VIDEO_URL} target="_blank" rel="noopener noreferrer" className="underline hover:no-underline">
          Quickstart guide
        </a>
        . Scoped by place of performance (where the work happened) - the live API doesn&apos;t reliably support
        filtering this by recipient business address.
      </p>

      {flow.asideAnswer && <p className="mt-3 rounded bg-black/5 p-2 text-sm dark:bg-white/10">{flow.asideAnswer}</p>}

      {result && result.status === "result" ? (
        <div className="mt-3 space-y-2 text-sm">
          {result.note && (
            <p className="rounded bg-black/5 p-2 text-xs text-black/70 dark:bg-white/10 dark:text-white/70">
              {result.note}
            </p>
          )}
          <p>
            <span className="font-medium">Prime award obligations:</span> {formatCurrency(result.prime_total ?? 0)}
          </p>
          <p>
            <span className="font-medium">Subaward total:</span>{" "}
            {result.subaward_total != null ? formatCurrency(result.subaward_total) : "not available"}
          </p>
          <p className="text-xs text-black/60 dark:text-white/60">
            These are shown separately and should not be added together - subawards can overlap with prime
            awards, so summing them would double-count some of the same federal dollars.
          </p>
          <NamedAmountList title="Top recipients (prime)" items={result.top_recipients} />
          <NamedAmountList title="Top subrecipients (subawards)" items={result.top_subrecipients} />
          <Citations citations={[]} toolCitations={result.tool_citations} />
          <button
            type="button"
            onClick={reset}
            className="mt-2 rounded-md border border-black/15 px-3 py-1.5 text-xs hover:bg-black/5 dark:border-white/15 dark:hover:bg-white/10"
          >
            Look up a different location
          </button>
        </div>
      ) : result && result.status === "breakdown" ? (
        <div className="mt-3 space-y-2 text-sm">
          <p>
            <span className="font-medium">{result.fields.state} total (all districts):</span>{" "}
            {formatCurrency(result.state_prime_total ?? 0)} prime /{" "}
            {formatCurrency(result.state_subaward_total ?? 0)} subaward
          </p>
          <div className="max-h-64 overflow-y-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-black/50 dark:text-white/50">
                  <th className="py-1">District</th>
                  <th className="py-1 text-right">Prime</th>
                  <th className="py-1 text-right">Subaward</th>
                </tr>
              </thead>
              <tbody>
                {result.districts.map((d) => (
                  <tr key={d.code} className="border-t border-black/5 dark:border-white/10">
                    <td className="py-1">{d.name}</td>
                    <td className="py-1 text-right tabular-nums">{formatCurrency(d.prime_total)}</td>
                    <td className="py-1 text-right tabular-nums">{formatCurrency(d.subaward_total)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Citations citations={[]} toolCitations={result.tool_citations} />
          <button
            type="button"
            onClick={reset}
            className="mt-2 rounded-md border border-black/15 px-3 py-1.5 text-xs hover:bg-black/5 dark:border-white/15 dark:hover:bg-white/10"
          >
            Look up a different location
          </button>
        </div>
      ) : (
        <form onSubmit={(e) => void handleSubmit(e)} className="mt-3 space-y-2">
          {flow.prompt && <p className="text-sm text-black/70 dark:text-white/70">{flow.prompt}</p>}
          <div className="flex flex-wrap gap-2">
            <select
              value={state}
              onChange={(e) => setState(e.target.value)}
              className="rounded border border-black/15 px-2 py-1 text-sm dark:border-white/15 dark:bg-transparent"
            >
              <option value="">State</option>
              {STATES.map(([code, name]) => (
                <option key={code} value={code}>
                  {name}
                </option>
              ))}
            </select>
            <input
              value={district}
              onChange={(e) => setDistrict(e.target.value)}
              placeholder="District (blank = all, AL = at-large)"
              className="w-56 rounded border border-black/15 px-2 py-1 text-sm dark:border-white/15 dark:bg-transparent"
            />
            <input
              value={startYear}
              onChange={(e) => setStartYear(e.target.value)}
              placeholder="Start FY"
              inputMode="numeric"
              className="w-24 rounded border border-black/15 px-2 py-1 text-sm dark:border-white/15 dark:bg-transparent"
            />
            <input
              value={endYear}
              onChange={(e) => setEndYear(e.target.value)}
              placeholder="End FY"
              inputMode="numeric"
              className="w-24 rounded border border-black/15 px-2 py-1 text-sm dark:border-white/15 dark:bg-transparent"
            />
          </div>
          <button
            type="submit"
            disabled={flow.loading}
            className="rounded-md border border-black/15 px-3 py-1.5 text-sm hover:bg-black/5 disabled:opacity-50 dark:border-white/15 dark:hover:bg-white/10"
          >
            Look up spending
          </button>
        </form>
      )}
    </div>
  );
}
