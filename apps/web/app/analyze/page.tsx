"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAppStore } from "@/lib/store";
import { PredictiveTab } from "@/components/analyze/PredictiveTab";
import { CausalTab } from "@/components/analyze/CausalTab";
import { InterventionsTab } from "@/components/analyze/InterventionsTab";
import { ExecutiveTab } from "@/components/analyze/ExecutiveTab";
import { EdaTab } from "@/components/analyze/EdaTab";

const TABS = [
  { id: "overview",      label: "Overview" },
  { id: "predictive",    label: "Predictive" },
  { id: "causal",        label: "Causal" },
  { id: "interventions", label: "Interventions" },
  { id: "executive",     label: "Executive" },
];

import { useState } from "react";

export default function AnalyzePage() {
  const router = useRouter();
  const { analysis, dataset, target } = useAppStore();
  const [activeTab, setActiveTab] = useState("overview");

  useEffect(() => {
    if (!analysis) router.replace("/setup");
  }, [analysis, router]);

  if (!analysis || !dataset || !target) return null;

  const best = analysis.predictive.find((p) => p.is_winner) ?? analysis.predictive[0];

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <div className="border-b border-border/40 bg-background/80 backdrop-blur sticky top-0 z-50">
        <div className="container flex items-center justify-between h-14">
          <div className="flex items-center gap-4">
            <button onClick={() => router.push("/setup")} className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              ← Setup
            </button>
            <span className="text-xs text-muted-foreground hidden sm:block">
              {dataset.name} · target: <strong>{target}</strong> · {analysis.row_count.toLocaleString()} rows
            </span>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <span className="text-muted-foreground">Best: {best.display_name}</span>
            <span className={`font-mono font-bold ${best.metrics.r2 > 0.6 ? "text-green-400" : best.metrics.r2 > 0.3 ? "text-yellow-400" : "text-red-400"}`}>
              R²={best.metrics.r2.toFixed(3)}
            </span>
            <span className="text-muted-foreground">· {analysis.runtime_seconds}s</span>
          </div>
        </div>
        {/* Tabs */}
        <div className="container">
          <nav className="flex gap-1 overflow-x-auto pb-0">
            {TABS.map(({ id, label }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
                  activeTab === id
                    ? "border-primary text-foreground"
                    : "border-transparent text-muted-foreground hover:text-foreground"
                }`}
              >
                {label}
                {id === "interventions" && (
                  <span className="ml-1.5 rounded-full bg-primary/20 text-primary text-xs px-1.5 py-0.5">
                    {analysis.interventions.length}
                  </span>
                )}
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Tab content */}
      <div className="container py-6 max-w-6xl">
        {activeTab === "overview"      && <OverviewPanel analysis={analysis} best={best} />}
        {activeTab === "predictive"    && <PredictiveTab results={analysis.predictive} />}
        {activeTab === "causal"        && <CausalTab effects={analysis.causal} target={target} />}
        {activeTab === "interventions" && <InterventionsTab interventions={analysis.interventions} target={target} />}
        {activeTab === "executive"     && <ExecutiveTab executive={analysis.executive} />}
      </div>
    </div>
  );
}

function OverviewPanel({ analysis, best }: { analysis: import("@/lib/types").AnalysisBundle; best: import("@/lib/types").PredictiveResult }) {
  return (
    <div className="space-y-6">
      {/* KPI cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <MetricCard label="Best Model R²" value={best.metrics.r2.toFixed(3)}
          sub={best.display_name}
          color={best.metrics.r2 > 0.6 ? "green" : best.metrics.r2 > 0.3 ? "yellow" : "red"} />
        <MetricCard label="RMSE" value={best.metrics.rmse.toFixed(3)} sub="test set" />
        <MetricCard label="Causal Levers" value={String(analysis.causal.filter(e => e.p_value < 0.05).length)}
          sub="p < 0.05" color="blue" />
        <MetricCard label="Recommendations" value={String(analysis.interventions.length)}
          sub="ranked by impact" color="purple" />
      </div>

      {/* Model comparison table */}
      <div className="rounded-xl border border-border/60 bg-card overflow-hidden">
        <div className="px-4 py-3 border-b border-border/60">
          <p className="font-semibold text-sm">Model Comparison</p>
          <p className="text-xs text-muted-foreground">All models trained on the same 80/20 split. Best by test R² is selected.</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border/40 text-xs text-muted-foreground">
                <th className="px-4 py-2 text-left">Model</th>
                <th className="px-4 py-2 text-right">Test R²</th>
                <th className="px-4 py-2 text-right">RMSE</th>
                <th className="px-4 py-2 text-right">MAE</th>
                <th className="px-4 py-2 text-right">CV R² (±σ)</th>
              </tr>
            </thead>
            <tbody>
              {analysis.predictive.map((r) => (
                <tr key={r.model} className={`border-b border-border/20 ${r.is_winner ? "bg-primary/5" : ""}`}>
                  <td className="px-4 py-2 font-medium text-sm">
                    {r.is_winner && <span className="text-primary mr-1">★</span>}
                    {r.display_name}
                  </td>
                  <td className="px-4 py-2 text-right font-mono text-xs">{r.metrics.r2.toFixed(4)}</td>
                  <td className="px-4 py-2 text-right font-mono text-xs">{r.metrics.rmse.toFixed(4)}</td>
                  <td className="px-4 py-2 text-right font-mono text-xs">{r.metrics.mae.toFixed(4)}</td>
                  <td className="px-4 py-2 text-right font-mono text-xs text-muted-foreground">
                    {r.metrics.cv_r2_mean !== undefined
                      ? `${r.metrics.cv_r2_mean.toFixed(3)} ±${r.metrics.cv_r2_std?.toFixed(3)}`
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Warnings */}
      {analysis.warnings.length > 0 && (
        <div className="rounded-xl border border-yellow-500/30 bg-yellow-500/5 p-4">
          <p className="text-xs font-semibold text-yellow-400 mb-2">Analysis Warnings</p>
          <ul className="space-y-1">
            {analysis.warnings.map((w, i) => (
              <li key={i} className="text-xs text-muted-foreground">• {w}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function MetricCard({ label, value, sub, color }: {
  label: string; value: string; sub?: string; color?: string;
}) {
  const colorMap: Record<string, string> = {
    green: "text-green-400", yellow: "text-yellow-400",
    red: "text-red-400", blue: "text-blue-400", purple: "text-purple-400",
  };
  return (
    <div className="rounded-xl border border-border/60 bg-card p-4">
      <p className="text-xs text-muted-foreground mb-1">{label}</p>
      <p className={`text-2xl font-bold font-mono ${color ? colorMap[color] : ""}`}>{value}</p>
      {sub && <p className="text-xs text-muted-foreground mt-0.5">{sub}</p>}
    </div>
  );
}
