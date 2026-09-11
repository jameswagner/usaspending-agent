"use client";

import { useEffect, useState } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Bar, Line } from "react-chartjs-2";
import type { ChartSpec } from "@/lib/types";

ChartJS.register(CategoryScale, LinearScale, BarElement, LineElement, PointElement, Title, Tooltip, Legend);

// Chart.js draws to canvas, so it can't pick up globals.css's
// prefers-color-scheme variables - the tokens below are duplicated from
// there (dataviz skill's sequential-blue + chrome/ink tokens).
const CHART_TOKENS = {
  light: { series: "#2a78d6", primaryInk: "#0b0b0b", mutedInk: "#898781", gridline: "#e1e0d9" },
  dark: { series: "#3987e5", primaryInk: "#ffffff", mutedInk: "#898781", gridline: "#2c2c2a" },
};

function useChartTokens() {
  const [isDark, setIsDark] = useState(
    () => typeof window !== "undefined" && window.matchMedia("(prefers-color-scheme: dark)").matches
  );

  useEffect(() => {
    const query = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = (e: MediaQueryListEvent) => setIsDark(e.matches);
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);

  return isDark ? CHART_TOKENS.dark : CHART_TOKENS.light;
}

interface ChartBlockProps {
  chart: ChartSpec;
}

export function ChartBlock({ chart }: ChartBlockProps) {
  const tokens = useChartTokens();

  const data = {
    labels: chart.labels,
    datasets: [
      {
        label: chart.title,
        data: chart.values,
        backgroundColor: tokens.series,
        borderColor: tokens.series,
      },
    ],
  };
  const options = {
    responsive: true,
    plugins: {
      // Single series - the title already names it, a legend would just repeat it.
      legend: { display: false },
      title: { display: true, text: chart.title, color: tokens.primaryInk },
    },
    scales: {
      x: { ticks: { color: tokens.mutedInk }, grid: { color: tokens.gridline } },
      y: { ticks: { color: tokens.mutedInk }, grid: { color: tokens.gridline } },
    },
  };

  return (
    <div className="w-full max-w-xl">
      {chart.chart_type === "line" ? <Line data={data} options={options} /> : <Bar data={data} options={options} />}
    </div>
  );
}
