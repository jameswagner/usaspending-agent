"use client";

import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ChartSpec } from "@/lib/types";

interface ChartBlockProps {
  chart: ChartSpec;
}

export function ChartBlock({ chart }: ChartBlockProps) {
  const data = chart.labels.map((label, i) => ({ label, value: chart.values[i] }));

  return (
    <div className="w-full max-w-xl">
      <p className="text-sm font-medium" style={{ color: "var(--chart-ink)" }}>
        {chart.title}
      </p>
      <ResponsiveContainer width="100%" height={300}>
        {chart.chart_type === "line" ? (
          <LineChart data={data}>
            <CartesianGrid stroke="var(--chart-gridline)" />
            <XAxis dataKey="label" tick={{ fill: "var(--chart-muted-ink)" }} />
            <YAxis tick={{ fill: "var(--chart-muted-ink)" }} />
            <Tooltip />
            <Line type="monotone" dataKey="value" stroke="var(--chart-series)" dot={false} />
          </LineChart>
        ) : (
          <BarChart data={data}>
            <CartesianGrid stroke="var(--chart-gridline)" />
            <XAxis dataKey="label" tick={{ fill: "var(--chart-muted-ink)" }} />
            <YAxis tick={{ fill: "var(--chart-muted-ink)" }} />
            <Tooltip />
            <Bar dataKey="value" fill="var(--chart-series)" />
          </BarChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}
