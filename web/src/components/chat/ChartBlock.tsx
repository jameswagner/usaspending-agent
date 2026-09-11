"use client";

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

interface ChartBlockProps {
  chart: ChartSpec;
}

export function ChartBlock({ chart }: ChartBlockProps) {
  const data = {
    labels: chart.labels,
    datasets: [{ label: chart.title, data: chart.values }],
  };
  const options = {
    responsive: true,
    plugins: { title: { display: true, text: chart.title } },
  };

  return (
    <div className="w-full max-w-xl">
      {chart.chart_type === "line" ? <Line data={data} options={options} /> : <Bar data={data} options={options} />}
    </div>
  );
}
