"use client";

import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ChartSpec } from "@/lib/types";

interface ChartBlockProps {
  chart: ChartSpec;
}

// Recharts silently drops overlapping category ticks rather than wrapping
// them - with long agency/NAICS-style labels that can collapse a whole axis
// down to a single tick, so labels are shortened and angled instead of left
// to the library's own collision handling.
function truncateLabel(label: string, max = 14): string {
  return label.length > max ? `${label.slice(0, max - 1)}…` : label;
}

function formatCurrency(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1e9) return `$${(value / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `$${(value / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `$${(value / 1e3).toFixed(0)}K`;
  return `$${value.toLocaleString()}`;
}

export function ChartBlock({ chart }: ChartBlockProps) {
  const data = chart.labels.map((label, i) => ({ label, value: chart.values[i] }));
  const margin = { top: 8, right: 12, bottom: 56, left: 8 };
  const xAxisTick = {
    fill: "var(--chart-muted-ink)",
    fontSize: 12,
  };
  const tooltipStyle = {
    background: "var(--chart-surface, #fff)",
    border: "1px solid var(--chart-gridline)",
    borderRadius: 6,
    fontSize: 12,
  };

  return (
    <div className="w-full max-w-xl">
      <p className="text-sm font-medium" style={{ color: "var(--chart-ink)" }}>
        {chart.title}
      </p>
      <ResponsiveContainer width="100%" height={340}>
        {chart.chart_type === "line" ? (
          <LineChart data={data} margin={margin}>
            <CartesianGrid stroke="var(--chart-gridline)" vertical={false} />
            <XAxis
              dataKey="label"
              tick={xAxisTick}
              tickFormatter={(label: string) => truncateLabel(label)}
              angle={-35}
              textAnchor="end"
              interval={0}
              height={70}
              axisLine={{ stroke: "var(--chart-gridline)" }}
              tickLine={false}
            />
            <YAxis
              tick={xAxisTick}
              tickFormatter={formatCurrency}
              width={64}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              contentStyle={tooltipStyle}
              labelStyle={{ color: "var(--chart-ink)" }}
              formatter={(value) => [formatCurrency(Number(value)), "value"]}
            />
            <Line type="monotone" dataKey="value" stroke="var(--chart-series)" strokeWidth={2} dot={false} />
          </LineChart>
        ) : (
          <BarChart data={data} margin={margin} barCategoryGap="20%">
            <CartesianGrid stroke="var(--chart-gridline)" vertical={false} />
            <XAxis
              dataKey="label"
              tick={xAxisTick}
              tickFormatter={(label: string) => truncateLabel(label)}
              angle={-35}
              textAnchor="end"
              interval={0}
              height={70}
              axisLine={{ stroke: "var(--chart-gridline)" }}
              tickLine={false}
            />
            <YAxis
              tick={xAxisTick}
              tickFormatter={formatCurrency}
              width={64}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              contentStyle={tooltipStyle}
              labelStyle={{ color: "var(--chart-ink)" }}
              formatter={(value) => [formatCurrency(Number(value)), "value"]}
              cursor={{ fill: "var(--chart-gridline)", opacity: 0.4 }}
            />
            <Bar dataKey="value" fill="var(--chart-series)" radius={[4, 4, 0, 0]} />
          </BarChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}
