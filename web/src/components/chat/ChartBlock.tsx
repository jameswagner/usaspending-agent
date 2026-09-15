"use client";

import { useRef } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ChartSpec } from "@/lib/types";

interface ChartBlockProps {
  chart: ChartSpec;
}

const LABEL_MAX = 32;

// Recharts silently drops overlapping category ticks rather than wrapping
// them, so long agency/NAICS-style labels are truncated with an ellipsis
// (full text still shows in the tooltip) instead of collapsing the axis.
function truncateLabel(label: string, max = LABEL_MAX): string {
  return label.length > max ? `${label.slice(0, max - 1)}…` : label;
}

function formatCurrency(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1e9) return `$${(value / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `$${(value / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `$${(value / 1e3).toFixed(0)}K`;
  return `$${value.toLocaleString()}`;
}

function slugify(title: string): string {
  return title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-+|-+$)/g, "") || "chart";
}

export function ChartBlock({ chart }: ChartBlockProps) {
  const chartRef = useRef<HTMLDivElement>(null);
  const data = chart.labels.map((label, i) => ({ label, value: chart.values[i] }));
  const isBar = chart.chart_type === "bar";
  // Horizontal bars need height proportional to category count, and a
  // category-axis column wide enough for the longest (truncated) label -
  // both computed from the data rather than fixed, since category counts
  // and label lengths vary a lot across tools (NAICS names vs. state names).
  const barChartHeight = Math.max(220, data.length * 34 + 50);
  const longestLabelLen = Math.min(LABEL_MAX, Math.max(4, ...chart.labels.map((l) => l.length)));
  const categoryAxisWidth = Math.min(210, Math.max(90, longestLabelLen * 6.2));

  const tickStyle = { fill: "var(--chart-muted-ink)", fontSize: 12 };
  const tooltipStyle = {
    background: "var(--background)",
    border: "1px solid var(--chart-gridline)",
    borderRadius: 6,
    fontSize: 12,
  };

  function handleDownload() {
    const svg = chartRef.current?.querySelector("svg");
    if (!svg) return;
    const { width, height } = svg.getBoundingClientRect();
    const scale = 2;
    const clone = svg.cloneNode(true) as SVGSVGElement;
    clone.setAttribute("width", String(width));
    clone.setAttribute("height", String(height));
    const bgColor = getComputedStyle(document.documentElement).getPropertyValue("--background").trim() || "#ffffff";
    const svgData = new XMLSerializer().serializeToString(clone);
    const url = URL.createObjectURL(new Blob([svgData], { type: "image/svg+xml;charset=utf-8" }));

    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = width * scale;
      canvas.height = height * scale;
      const ctx = canvas.getContext("2d");
      URL.revokeObjectURL(url);
      if (!ctx) return;
      ctx.scale(scale, scale);
      ctx.fillStyle = bgColor;
      ctx.fillRect(0, 0, width, height);
      ctx.drawImage(img, 0, 0, width, height);
      canvas.toBlob((blob) => {
        if (!blob) return;
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = `${slugify(chart.title)}.png`;
        link.click();
        URL.revokeObjectURL(link.href);
      });
    };
    img.src = url;
  }

  return (
    <div className="w-full max-w-xl">
      <div className="flex items-center justify-between gap-2">
        <p className="text-sm font-medium" style={{ color: "var(--chart-ink)" }}>
          {chart.title}
        </p>
        <button
          type="button"
          onClick={handleDownload}
          aria-label="Download chart as PNG"
          title="Download chart as PNG"
          className="shrink-0 rounded p-1 hover:bg-black/5 dark:hover:bg-white/10"
          style={{ color: "var(--chart-muted-ink)" }}
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M8 1.5v9m0 0L4.5 7M8 10.5L11.5 7" />
            <path d="M2 12.5v1a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1v-1" />
          </svg>
        </button>
      </div>
      <div ref={chartRef}>
      <ResponsiveContainer width="100%" height={isBar ? barChartHeight : 300}>
        {chart.chart_type === "line" ? (
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
            <CartesianGrid stroke="var(--chart-gridline)" vertical={false} />
            <XAxis dataKey="label" tick={tickStyle} axisLine={{ stroke: "var(--chart-gridline)" }} tickLine={false} />
            <YAxis tick={tickStyle} tickFormatter={formatCurrency} width={56} axisLine={false} tickLine={false} />
            <Tooltip
              contentStyle={tooltipStyle}
              labelStyle={{ color: "var(--chart-ink)" }}
              formatter={(value) => [formatCurrency(Number(value)), "value"]}
            />
            <Line type="monotone" dataKey="value" stroke="var(--chart-series)" strokeWidth={2} dot={false} />
          </LineChart>
        ) : (
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 52, bottom: 4, left: 4 }} barCategoryGap="24%">
            <CartesianGrid stroke="var(--chart-gridline)" horizontal={false} />
            <XAxis
              type="number"
              domain={[0, (dataMax: number) => Math.ceil(dataMax * 1.15)]}
              tick={tickStyle}
              tickFormatter={formatCurrency}
              axisLine={{ stroke: "var(--chart-gridline)" }}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="label"
              tick={tickStyle}
              tickFormatter={(label: string) => truncateLabel(label)}
              width={categoryAxisWidth}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              contentStyle={tooltipStyle}
              labelStyle={{ color: "var(--chart-ink)" }}
              formatter={(value) => [formatCurrency(Number(value)), "value"]}
              cursor={{ fill: "var(--chart-gridline)", opacity: 0.4 }}
            />
            <Bar dataKey="value" fill="var(--chart-series)" radius={[0, 4, 4, 0]}>
              <LabelList
                dataKey="value"
                position="right"
                formatter={(value) => formatCurrency(Number(value))}
                style={{ fill: "var(--chart-ink)", fontSize: 12 }}
              />
            </Bar>
          </BarChart>
        )}
      </ResponsiveContainer>
      </div>
    </div>
  );
}
