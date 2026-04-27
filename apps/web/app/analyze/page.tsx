"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ScatterChart, Scatter, Cell, ReferenceLine,
} from "recharts";
import {
  ArrowLeft, BarChart3, GitBranch, Lightbulb, Brain,
  Star, AlertTriangle, CheckCircle2, ArrowUpRight, ArrowDownRight, Info,
  Send, Loader2,
} from "lucide-react";
import { useAppStore } from "@/lib/store";
import { askCopilot, ApiError } from "@/lib/api-client";
import type { AnalysisBundle, PredictiveResult, CausalEffect, Intervention, CopilotAnswerResponse } from "@/lib/types";

/* ─── safe formatters — never crash on null/undefined ─────────────────── */
const n = (v: unknown): number => (v == null || isNaN(Number(v)) ? 0 : Number(v));
const fmt  = (v: unknown, d = 3) => n(v).toFixed(d);
const fmtP = (v: unknown)        => (n(v) < 0.001 ? "<.001" : n(v).toFixed(4));
const sign = (v: unknown)        => (n(v) >= 0 ? "+" : "") + n(v).toFixed(3);
const r2Color = (v: unknown) => n(v) > 0.7 ? "text-green-400" : n(v) > 0.4 ? "text-yellow-400" : "text-red-400";

const STRENGTH: Record<string, string> = {
  strong:       "bg-green-500/20 text-green-400 border-green-500/30",
  moderate:     "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  weak:         "bg-orange-500/20 text-orange-400 border-orange-500/30",
  insufficient: "bg-gray-500/20 text-gray-400 border-gray-600/30",
};
const EVTYPE: Record<string, string> = {
  causal:     "bg-blue-500/20 text-blue-400 border-blue-500/30",
  predictive: "bg-purple-500/20 text-purple-400 border-purple-500/30",
  mixed:      "bg-teal-500/20 text-teal-400 border-teal-500/30",
};
const TABS = [
  { id: "overview",      label: "Overview" },
  { id: "predictive",    label: "Predictive Models" },
  { id: "causal",        label: "Causal Analysis" },
  { id: "interventions", label: "Interventions" },
  { id: "executive",     label: "Executive Summary" },
  { id: "copilot",       label: "Copilot" },
];

/* ═══════════════ PAGE ═══════════════════════════════════════════════════ */
export default function AnalyzePage() {
  const router = useRouter();
  const { analysis, target } = useAppStore();
  const [tab, setTab] = useState("overview");
  const [mounted, setMounted] = useState(false);

  useEffect(() => { setMounted(true); }, []);
  useEffect(() => {
    if (mounted && !analysis) router.replace("/setup");
  }, [mounted, analysis, router]);

  if (!mounted || !analysis || !target) return (
    <div className="min-h-screen bg-background flex items-center justify-center">
      <p className="text-muted-foreground text-sm">Loading results…</p>
    </div>
  );

  const best   = analysis.predictive.find(p => p.is_winner) ?? analysis.predictive[0];
  const sigN   = analysis.causal.filter(e => n(e.p_value) < 0.05).length;
  const topIv  = analysis.interventions[0];

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* top bar */}
      <div className="border-b border-border/40 bg-background/90 backdrop-blur sticky top-0 z-50">
        <div className="container flex items-center justify-between h-14 gap-4">
          <button onClick={() => router.push("/setup")}
            className="flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors shrink-0">
            <ArrowLeft className="h-4 w-4" /> Setup
          </button>
          <div className="flex items-center gap-3 text-xs overflow-hidden">
            <span className="text-muted-foreground truncate hidden sm:block">
              {analysis.dataset_name} · <strong className="text-foreground">{target}</strong>
            </span>
            <span className={`font-mono font-bold ${r2Color(best?.metrics?.r2)}`}>
              R²={fmt(best?.metrics?.r2)}
            </span>
            <span className="text-muted-foreground hidden md:block">
              {best?.display_name} · {analysis.row_count?.toLocaleString()} rows · {fmt(analysis.runtime_seconds, 1)}s
            </span>
          </div>
        </div>
        <div className="container">
          <nav className="flex overflow-x-auto gap-0 pb-0">
            {TABS.map(({ id, label }) => (
              <button key={id} onClick={() => setTab(id)}
                className={`px-4 py-2.5 text-sm font-medium border-b-2 whitespace-nowrap transition-colors ${
                  tab === id ? "border-primary text-foreground" : "border-transparent text-muted-foreground hover:text-foreground"
                }`}>
                {label}
                {id === "interventions" && analysis.interventions.length > 0 && (
                  <span className="ml-1.5 rounded-full bg-primary/20 text-primary text-xs px-1.5 py-0.5">
                    {analysis.interventions.length}
                  </span>
                )}
              </button>
            ))}
          </nav>
        </div>
      </div>

      <div className="container py-6 max-w-6xl">
        {tab === "overview"      && <OverviewTab    analysis={analysis} best={best} sigN={sigN} topIv={topIv} />}
        {tab === "predictive"    && <PredictiveTab  results={analysis.predictive} />}
        {tab === "causal"        && <CausalTab      effects={analysis.causal} target={target} />}
        {tab === "interventions" && <InterventionsTab interventions={analysis.interventions} target={target} />}
        {tab === "executive"     && <ExecutiveTab   exec={analysis.executive} />}
        {tab === "copilot"       && <CopilotTab     analysis={analysis} />}
      </div>
    </div>
  );
}

/* ═══════════════ OVERVIEW ═══════════════════════════════════════════════ */
function OverviewTab({ analysis, best, sigN, topIv }: {
  analysis: AnalysisBundle; best: PredictiveResult;
  sigN: number; topIv: Intervention | undefined;
}) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <KpiCard label="Best Model R²" value={fmt(best?.metrics?.r2)}
          sub={best?.display_name ?? "—"}
          color={n(best?.metrics?.r2) > 0.6 ? "green" : n(best?.metrics?.r2) > 0.3 ? "yellow" : "red"} />
        <KpiCard label="Test RMSE"  value={fmt(best?.metrics?.rmse)} sub={`MAE ${fmt(best?.metrics?.mae)}`} />
        <KpiCard label="Causal Levers" value={String(sigN)} sub="p < 0.05" color="blue" />
        <KpiCard label="Top Impact"
          value={topIv ? `${Math.abs(n(topIv.expected_kpi_change_pct)).toFixed(1)}%` : "—"}
          sub={topIv?.feature ?? "No interventions"} color="purple" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model comparison */}
        <Card title="Model Comparison" sub="All models on the same 80/20 train/test split — best by R² marked ★">
          <div className="space-y-2 mt-3">
            {analysis.predictive.map(r => (
              <div key={r.model} className={`flex items-center gap-3 p-2 rounded-lg ${r.is_winner ? "bg-primary/5 border border-primary/20" : ""}`}>
                {r.is_winner ? <Star className="h-3.5 w-3.5 text-primary shrink-0" /> : <div className="h-3.5 w-3.5 shrink-0" />}
                <span className="text-xs font-medium w-28 shrink-0">{r.display_name}</span>
                <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-primary/70 rounded-full" style={{ width: `${Math.max(0, n(r.metrics?.r2)) * 100}%` }} />
                </div>
                <span className={`font-mono text-xs w-14 text-right ${r2Color(r.metrics?.r2)}`}>{fmt(r.metrics?.r2)}</span>
                {r.metrics?.cv_r2_mean != null && (
                  <span className="font-mono text-xs text-muted-foreground hidden md:block w-20 text-right">
                    CV {fmt(r.metrics.cv_r2_mean, 2)}±{fmt(r.metrics.cv_r2_std, 2)}
                  </span>
                )}
              </div>
            ))}
          </div>
          <p className="text-xs text-muted-foreground mt-3">
            R² = fraction of variance explained (0–1, higher = better).
            CV R² = 3-fold cross-validated score (±σ shows stability).
          </p>
        </Card>

        {/* Top interventions */}
        <Card title="Top Recommendations" sub="Ranked by estimated KPI impact — click Interventions tab for full detail">
          <div className="space-y-2 mt-3">
            {analysis.interventions.slice(0, 5).map(iv => (
              <div key={iv.feature} className="flex items-center gap-3 p-2 rounded-lg bg-muted/20">
                <span className="h-5 w-5 rounded-full bg-primary/15 text-primary text-xs font-bold flex items-center justify-center shrink-0">
                  {iv.rank}
                </span>
                <span className="text-xs font-mono flex-1 truncate">{iv.feature}</span>
                <span className={`text-xs font-medium flex items-center gap-0.5 ${iv.direction === "decrease" ? "text-blue-400" : "text-green-400"}`}>
                  {iv.direction === "decrease" ? <ArrowDownRight className="h-3.5 w-3.5" /> : <ArrowUpRight className="h-3.5 w-3.5" />}
                  {iv.direction}
                </span>
                <span className={`font-mono text-xs font-bold ${n(iv.expected_kpi_change) < 0 ? "text-green-400" : "text-red-400"}`}>
                  {n(iv.expected_kpi_change) > 0 ? "+" : ""}{n(iv.expected_kpi_change_pct).toFixed(1)}%
                </span>
              </div>
            ))}
            {analysis.interventions.length === 0 && (
              <p className="text-xs text-muted-foreground py-4 text-center">
                No recommendations — assign controllable numeric columns in Setup.
              </p>
            )}
          </div>
        </Card>
      </div>

      {analysis.warnings?.length > 0 && (
        <div className="rounded-xl border border-yellow-500/30 bg-yellow-500/5 p-4 flex gap-3">
          <AlertTriangle className="h-4 w-4 text-yellow-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            {analysis.warnings.map((w, i) => <p key={i} className="text-xs text-muted-foreground">{w}</p>)}
          </div>
        </div>
      )}
    </div>
  );
}

/* ═══════════════ PREDICTIVE ═════════════════════════════════════════════ */
function PredictiveTab({ results }: { results: PredictiveResult[] }) {
  const [sel, setSel] = useState(results.find(r => r.is_winner)?.model ?? results[0]?.model ?? "");
  const model = results.find(r => r.model === sel) ?? results[0];
  if (!model) return <Empty msg="No model results available." />;

  const impData = (model.importances ?? []).slice(0, 12).map(f => ({
    name: (f.feature ?? "").length > 22 ? (f.feature ?? "").slice(0, 22) + "…" : (f.feature ?? ""),
    value: parseFloat(n(f.importance_norm).toFixed(4)),
  }));

  const scatterData = (model.predictions ?? []).slice(0, 300).map(p => ({
    actual:    n(p.actual),
    predicted: n(p.predicted),
  }));

  return (
    <div className="space-y-6">
      {/* selector */}
      <div className="flex flex-wrap gap-2">
        {results.map(r => (
          <button key={r.model} onClick={() => setSel(r.model)}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium border transition-colors ${
              sel === r.model ? "bg-primary text-primary-foreground border-primary" : "border-border hover:bg-accent"
            }`}>
            {r.is_winner && "★ "}{r.display_name}
            <span className={`ml-2 font-mono text-xs opacity-80 ${r2Color(r.metrics?.r2)}`}>
              R²={fmt(r.metrics?.r2)}
            </span>
          </button>
        ))}
      </div>

      {/* metric cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetCard l="Test R²"    v={fmt(model.metrics?.r2)}    c={r2Color(model.metrics?.r2)} />
        <MetCard l="Adj. R²"   v={model.metrics?.adj_r2 != null ? fmt(model.metrics.adj_r2) : "—"} />
        <MetCard l="RMSE"      v={fmt(model.metrics?.rmse)} />
        <MetCard l="CV R² ±σ"  v={model.metrics?.cv_r2_mean != null
          ? `${fmt(model.metrics.cv_r2_mean, 2)} ±${fmt(model.metrics.cv_r2_std, 2)}` : "—"} />
      </div>
      <div className="text-xs text-muted-foreground rounded-lg border border-border/40 bg-card/50 p-3 leading-relaxed">
        <strong>How to read these metrics:</strong>{" "}
        <strong>R²</strong> (coefficient of determination) — fraction of variance in the target explained by the model.
        1.0 = perfect, 0 = no better than the mean.{" "}
        <strong>RMSE</strong> (root mean square error) — average prediction error in the same units as the target.{" "}
        <strong>Adj. R²</strong> penalises for extra features (OLS/Ridge only).{" "}
        <strong>CV R²</strong> is the 3-fold cross-validated score — tests generalisation, not just in-sample fit.
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Feature Importances" sub="Normalised 0–1. For tree models = mean impurity decrease. For linear = |standardised coefficient|.">
          {impData.length === 0 ? <Empty msg="No importances available." /> : (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart layout="vertical" data={impData} margin={{ left: 4, right: 20, top: 8, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 33% 18%)" />
                <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 10, fill: "hsl(215 20% 55%)" }} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fill: "hsl(215 20% 55%)" }} width={140} />
                <Tooltip formatter={(v: number) => v.toFixed(4)} contentStyle={{ background: "hsl(222 47% 10%)", border: "1px solid hsl(217 33% 18%)", borderRadius: 8, fontSize: 12 }} />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {impData.map((_, i) => <Cell key={i} fill={i === 0 ? "hsl(217 91% 60%)" : "hsl(217 91% 60% / 0.5)"} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Actual vs Predicted (test set)" sub="Points on the diagonal = perfect predictions. Spread = error magnitude.">
          {scatterData.length === 0 ? <Empty msg="No prediction data." /> : (
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart margin={{ left: 4, right: 20, top: 8, bottom: 24 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 33% 18%)" />
                <XAxis dataKey="actual" name="Actual" type="number" tick={{ fontSize: 10, fill: "hsl(215 20% 55%)" }}
                  label={{ value: "Actual", position: "insideBottom", offset: -14, fontSize: 10, fill: "hsl(215 20% 55%)" }} />
                <YAxis dataKey="predicted" name="Predicted" type="number" tick={{ fontSize: 10, fill: "hsl(215 20% 55%)" }}
                  label={{ value: "Predicted", angle: -90, position: "insideLeft", fontSize: 10, fill: "hsl(215 20% 55%)" }} />
                <Tooltip formatter={(v: number) => v.toFixed(3)} contentStyle={{ background: "hsl(222 47% 10%)", border: "1px solid hsl(217 33% 18%)", borderRadius: 8, fontSize: 12 }} />
                <Scatter data={scatterData} fill="hsl(217 91% 60%)" fillOpacity={0.5} />
              </ScatterChart>
            </ResponsiveContainer>
          )}
        </Card>
      </div>

      {/* Coefficients table */}
      {(model.coefficients ?? []).length > 1 && (
        <Card title="Regression Coefficients" sub="Features are standardised (μ=0, σ=1) so coefficients are directly comparable in magnitude.">
          <div className="text-xs text-muted-foreground rounded-lg bg-muted/20 p-3 mt-2 mb-3 leading-relaxed">
            <strong>How to read:</strong> β = change in target per +1 SD increase in feature, holding all others constant.
            A positive β means the feature pushes the target up. p-value tests whether β ≠ 0.
            ★ = statistically significant at α = 0.05.
          </div>
          <div className="overflow-x-auto max-h-72 overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-card">
                <tr className="border-b border-border/40 text-muted-foreground">
                  {["Feature","β (coef)","Std Err","t-stat","p-value","Sig"].map(h => (
                    <th key={h} className="px-3 py-2 text-left font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(model.coefficients ?? []).filter(c => c.feature !== "(intercept)").slice(0, 25).map(c => (
                  <tr key={c.feature} className={`border-b border-border/20 hover:bg-muted/10 ${!c.significant ? "opacity-50" : ""}`}>
                    <td className="px-3 py-1.5 font-mono">{c.feature ?? "—"}</td>
                    <td className={`px-3 py-1.5 font-mono font-bold ${n(c.coef) > 0 ? "text-red-400" : "text-green-400"}`}>{sign(c.coef)}</td>
                    <td className="px-3 py-1.5 font-mono text-muted-foreground">{fmt(c.std_err)}</td>
                    <td className="px-3 py-1.5 font-mono">{fmt(c.t_stat, 2)}</td>
                    <td className={`px-3 py-1.5 font-mono ${n(c.p_value) < 0.05 ? "text-green-400" : "text-muted-foreground"}`}>
                      {fmtP(c.p_value)}
                    </td>
                    <td className="px-3 py-1.5 text-yellow-400">{c.significant ? "★" : ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}

/* ═══════════════ CAUSAL ═════════════════════════════════════════════════ */
function CausalTab({ effects, target }: { effects: CausalEffect[]; target: string }) {
  if (!effects?.length) return (
    <Empty msg="No causal effects computed. Ensure you have controllable numeric columns with at least 30 non-missing rows and at least one confounder assigned." />
  );

  const chartData = effects.slice(0, 10).map(e => ({
    name: (e.feature ?? "").length > 20 ? (e.feature ?? "").slice(0, 20) + "…" : (e.feature ?? ""),
    effect: parseFloat(n(e.effect_per_std).toFixed(4)),
    sig: n(e.p_value) < 0.05,
  }));

  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-blue-500/20 bg-blue-500/5 p-4 text-xs text-muted-foreground leading-relaxed">
        <strong className="text-blue-400">Statistical method:</strong> Back-door adjusted OLS (statsmodels).
        For each controllable variable we fit:{" "}
        <code className="bg-muted px-1 rounded">{target} ~ feature + confounders + DAG_parents + context</code>.{" "}
        β is the standardised coefficient: <em>effect on {target} per +1 SD increase in the feature</em>, with all adjustment variables held constant.{" "}
        Mediators are excluded (blocking the causal path would absorb the effect).
        95% CIs and p-values are from OLS inference under homoskedasticity assumptions.
        This is <strong>observational</strong> — unobserved confounders may bias estimates.
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Adjusted Effect Sizes (β/SD)" sub="Red = increases target · Green = decreases target · Grey = not significant (p ≥ 0.05)">
          <ResponsiveContainer width="100%" height={Math.max(200, chartData.length * 38 + 40)}>
            <BarChart layout="vertical" data={chartData} margin={{ left: 4, right: 24, top: 8, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 33% 18%)" />
              <XAxis type="number" tick={{ fontSize: 10, fill: "hsl(215 20% 55%)" }} />
              <ReferenceLine x={0} stroke="hsl(215 20% 55%)" strokeDasharray="4 4" />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fill: "hsl(215 20% 55%)" }} width={140} />
              <Tooltip formatter={(v: number) => v.toFixed(4)} contentStyle={{ background: "hsl(222 47% 10%)", border: "1px solid hsl(217 33% 18%)", borderRadius: 8, fontSize: 12 }} />
              <Bar dataKey="effect" radius={[0, 4, 4, 0]}>
                {chartData.map((d, i) => (
                  <Cell key={i} fill={!d.sig ? "hsl(215 20% 35%)" : d.effect > 0 ? "hsl(0 63% 55%)" : "hsl(142 71% 45%)"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Inference Table" sub="β = effect per +1 SD · 95% CI · evidence strength">
          <div className="overflow-x-auto mt-3 max-h-80 overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-card">
                <tr className="border-b border-border/40 text-muted-foreground">
                  <th className="px-3 py-2 text-left">Feature</th>
                  <th className="px-3 py-2 text-right">β/SD</th>
                  <th className="px-3 py-2 text-right">95% CI</th>
                  <th className="px-3 py-2 text-right">p-val</th>
                  <th className="px-3 py-2 text-left">Evidence</th>
                </tr>
              </thead>
              <tbody>
                {effects.map(e => (
                  <tr key={e.feature} className="border-b border-border/20 hover:bg-muted/10 transition-colors">
                    <td className="px-3 py-2 font-mono font-semibold text-xs">{e.feature ?? "—"}</td>
                    <td className={`px-3 py-2 text-right font-mono font-bold ${n(e.effect_per_std) > 0 ? "text-red-400" : "text-green-400"}`}>
                      {sign(e.effect_per_std)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-muted-foreground text-xs">
                      [{fmt(e.conf_int_lo, 3)}, {fmt(e.conf_int_hi, 3)}]
                    </td>
                    <td className={`px-3 py-2 text-right font-mono ${n(e.p_value) < 0.05 ? "text-green-400" : "text-muted-foreground"}`}>
                      {fmtP(e.p_value)}
                    </td>
                    <td className="px-3 py-2">
                      <span className={`rounded-full border px-1.5 py-0.5 text-xs ${STRENGTH[e.evidence_strength] ?? ""}`}>
                        {e.evidence_strength}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>

      {effects.filter(e => e.warning).map(e => (
        <div key={e.feature} className="rounded-lg border border-yellow-500/30 bg-yellow-500/5 p-3 text-xs flex gap-2">
          <AlertTriangle className="h-3.5 w-3.5 text-yellow-400 shrink-0 mt-0.5" />
          <span><strong className="text-yellow-400">{e.feature}:</strong> {e.warning}</span>
        </div>
      ))}
    </div>
  );
}

/* ═══════════════ INTERVENTIONS ══════════════════════════════════════════ */
function InterventionsTab({ interventions, target }: { interventions: Intervention[]; target: string }) {
  if (!interventions?.length) return (
    <Empty msg="No recommendations — ensure you have controllable numeric columns in Setup." />
  );
  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-border/60 bg-card p-4 text-xs text-muted-foreground leading-relaxed">
        <strong className="text-foreground">How recommendations are generated:</strong>{" "}
        A gradient boosted regressor simulates shifting each controllable variable by ±1 SD while holding all others at their mean
        (counterfactual prediction). Where back-door OLS is significant (p&lt;0.05) the estimate is labelled{" "}
        <span className="text-blue-400 font-medium">causal</span>, otherwise{" "}
        <span className="text-purple-400 font-medium">predictive</span>.
        Ranked by |estimated KPI change|. Always validate with controlled experiments.
      </div>
      {interventions.map(iv => <IvCard key={iv.feature} iv={iv} target={target} />)}
    </div>
  );
}

function IvCard({ iv, target }: { iv: Intervention; target: string }) {
  const [open, setOpen] = useState(false);
  const improving = n(iv.expected_kpi_change) < 0;
  return (
    <div className="rounded-xl border border-border/60 bg-card overflow-hidden">
      <div className="flex items-center gap-3 px-4 py-3 bg-muted/10 border-b border-border/40">
        <span className="h-7 w-7 rounded-full bg-primary/15 text-primary text-sm font-bold flex items-center justify-center shrink-0">
          {iv.rank}
        </span>
        <div className="flex-1 min-w-0">
          <p className="font-semibold text-sm font-mono truncate">{iv.feature}</p>
        </div>
        <div className="flex items-center gap-2 flex-wrap justify-end">
          <Badge cls={EVTYPE[iv.evidence_type] ?? ""}>{iv.evidence_type}</Badge>
          <Badge cls={STRENGTH[iv.evidence_strength] ?? ""}>{iv.evidence_strength}</Badge>
          <span className={`font-mono font-bold text-sm ${improving ? "text-green-400" : "text-red-400"}`}>
            {n(iv.expected_kpi_change) > 0 ? "+" : ""}{n(iv.expected_kpi_change_pct).toFixed(1)}% {target}
          </span>
        </div>
      </div>
      <div className="p-4 grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
        <div className="space-y-2">
          <Row label="Action">
            <span className={iv.direction === "decrease" ? "text-blue-400 font-semibold" : "text-green-400 font-semibold"}>
              {iv.direction === "decrease"
                ? <><ArrowDownRight className="h-3.5 w-3.5 inline mr-0.5" />Decrease</>
                : <><ArrowUpRight className="h-3.5 w-3.5 inline mr-0.5" />Increase</>}
            </span>
          </Row>
          <Row label="Current mean">
            <code>{n(iv.current_mean).toFixed(3)}</code>
            <span className="text-muted-foreground ml-1 text-xs">
              [p10={n(iv.current_p10).toFixed(2)}, p90={n(iv.current_p90).toFixed(2)}]
            </span>
          </Row>
          <Row label="Suggested value">
            <code className="text-primary font-bold">{n(iv.suggested_value).toFixed(3)}</code>
            <span className="text-muted-foreground ml-1">
              ({n(iv.delta) >= 0 ? "+" : ""}{n(iv.delta).toFixed(3)}, {n(iv.delta_pct).toFixed(1)}%)
            </span>
          </Row>
          <Row label="Est. KPI change">
            <code className={improving ? "text-green-400 font-bold" : "text-red-400 font-bold"}>
              {n(iv.expected_kpi_change) > 0 ? "+" : ""}{n(iv.expected_kpi_change).toFixed(4)}
            </code>
          </Row>
        </div>
        <div className="space-y-2">
          <div>
            <p className="text-muted-foreground font-medium mb-0.5">Rationale</p>
            <p className="leading-relaxed">{iv.rationale ?? "—"}</p>
          </div>
          <div>
            <p className="text-muted-foreground font-medium mb-0.5">Tradeoff</p>
            <p className="leading-relaxed text-yellow-400/80">{iv.tradeoff ?? "—"}</p>
          </div>
        </div>
      </div>
      <button onClick={() => setOpen(o => !o)}
        className="w-full px-4 pb-2 text-left text-xs text-muted-foreground hover:text-foreground flex items-center gap-1 transition-colors">
        <Info className="h-3 w-3" /> {open ? "Hide" : "Show"} assumptions &amp; caveats
      </button>
      {open && (
        <div className="px-4 pb-4 border-t border-border/30 pt-3 space-y-1 text-xs text-muted-foreground">
          {(iv.assumptions ?? []).map((a, i) => <p key={i}>· {a}</p>)}
          <p className="italic mt-2 opacity-70">{iv.caveat ?? ""}</p>
        </div>
      )}
    </div>
  );
}

/* ═══════════════ EXECUTIVE ══════════════════════════════════════════════ */
function ExecutiveTab({ exec }: { exec: AnalysisBundle["executive"] }) {
  if (!exec) return <Empty msg="No executive summary available." />;
  return (
    <div className="max-w-2xl mx-auto space-y-5">
      <div className="rounded-xl border border-primary/30 bg-primary/5 p-6 text-center">
        <h2 className="text-xl font-bold mb-2">{exec.headline}</h2>
        <p className="text-sm text-muted-foreground">{exec.sub_headline}</p>
      </div>
      <Card title="Key Findings">
        <ul className="mt-3 space-y-2">
          {(exec.bullets ?? []).map((b, i) => (
            <li key={i} className="flex gap-3 text-sm">
              <CheckCircle2 className="h-4 w-4 text-primary shrink-0 mt-0.5" />
              <span>{b}</span>
            </li>
          ))}
        </ul>
      </Card>
      {(exec.top_levers ?? []).length > 0 && (
        <Card title="Top Levers to Pull">
          <div className="flex flex-wrap gap-2 mt-3">
            {exec.top_levers.map(l => (
              <span key={l} className="rounded-lg border border-green-500/30 bg-green-500/10 px-3 py-1 text-sm font-mono text-green-400">{l}</span>
            ))}
          </div>
        </Card>
      )}
      <Card title="Important Caveats">
        <ul className="mt-3 space-y-2">
          {(exec.cautions ?? []).map((c, i) => (
            <li key={i} className="flex gap-3 text-sm">
              <AlertTriangle className="h-4 w-4 text-yellow-400 shrink-0 mt-0.5" />
              <span className="text-muted-foreground">{c}</span>
            </li>
          ))}
        </ul>
      </Card>
      <Card title="Methodology">
        <p className="text-xs text-muted-foreground leading-relaxed mt-2">{exec.methodology_note}</p>
        <p className="text-xs text-muted-foreground leading-relaxed mt-2 italic">{exec.disclaimer}</p>
      </Card>
    </div>
  );
}

/* ═══════════════ COPILOT ═══════════════════════════════════════════════ */
function CopilotTab({ analysis }: { analysis: AnalysisBundle }) {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<CopilotAnswerResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const submit = async () => {
    const q = question.trim();
    if (!q || loading) return;
    setLoading(true);
    setError("");
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 45_000);
    try {
      const res = await askCopilot({
        analysis_id: analysis.request_id,
        question: q,
        max_citations: 5,
      }, controller.signal);
      setAnswer(res);
    } catch (err) {
      let msg = "Copilot request failed.";
      if (err instanceof Error && err.name === "AbortError") {
        msg = "Copilot timed out after 45s. Try a narrower question.";
      } else if (err instanceof ApiError) {
        msg = err.message;
      } else if (err instanceof Error) {
        msg = err.message;
      }
      setError(msg);
    } finally {
      clearTimeout(timeout);
      setLoading(false);
    }
  };

  return (
    <div className="space-y-5 max-w-3xl mx-auto">
      <div className="rounded-xl border border-border/60 bg-card p-5">
        <div className="flex items-start gap-3">
          <Brain className="h-5 w-5 text-primary shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="font-semibold text-sm">Analysis Copilot</p>
            <p className="text-xs text-muted-foreground mt-1 leading-relaxed">
              Ask about this analysis. Answers are grounded in retrieved dataset, model, causal, intervention, and summary artifacts.
            </p>
          </div>
        </div>
        <div className="mt-4 flex gap-2">
          <input
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter") submit(); }}
            placeholder="Ask about drivers, caveats, interventions, or model quality…"
            className="flex-1 bg-background border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/40"
          />
          <button
            onClick={submit}
            disabled={!question.trim() || loading}
            className="h-10 w-10 rounded-lg bg-primary text-primary-foreground flex items-center justify-center disabled:opacity-40"
            aria-label="Ask copilot"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          </button>
        </div>
        {error && (
          <div className="mt-3 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-400">
            {error}
          </div>
        )}
      </div>

      {answer && (
        <div className="rounded-xl border border-border/60 bg-card p-5 space-y-4">
          <div>
            <div className="flex items-center justify-between gap-3 mb-2">
              <p className="font-semibold text-sm">Answer</p>
              <span className="text-xs text-muted-foreground">
                {answer.used_llm ? answer.model ?? "Groq" : "retrieval only"}
              </span>
            </div>
            <p className="text-sm leading-relaxed whitespace-pre-wrap">{answer.answer}</p>
            {answer.warnings.map((w, i) => (
              <p key={i} className="text-xs text-yellow-400 mt-2">{w}</p>
            ))}
          </div>

          <div>
            <p className="font-semibold text-sm mb-2">Citations</p>
            <div className="space-y-2">
              {answer.citations.map((c, i) => (
                <div key={`${c.artifact_id}-${i}`} className="rounded-lg border border-border/50 bg-muted/10 p-3">
                  <div className="flex items-center justify-between gap-3 mb-1">
                    <span className="text-xs font-semibold">{c.title}</span>
                    <span className="text-xs text-muted-foreground">{c.kind} · score {fmt(c.score, 2)}</span>
                  </div>
                  <p className="text-xs text-muted-foreground leading-relaxed">{c.snippet}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ═══════════════ shared UI ══════════════════════════════════════════════ */
function Card({ title, sub, children }: { title: string; sub?: string; children?: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-border/60 bg-card p-5">
      <p className="font-semibold text-sm">{title}</p>
      {sub && <p className="text-xs text-muted-foreground mt-0.5">{sub}</p>}
      {children}
    </div>
  );
}
function KpiCard({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  const colors: Record<string, string> = { green:"text-green-400", yellow:"text-yellow-400", red:"text-red-400", blue:"text-blue-400", purple:"text-purple-400" };
  return (
    <div className="rounded-xl border border-border/60 bg-card p-4">
      <p className="text-xs text-muted-foreground mb-1">{label}</p>
      <p className={`text-2xl font-bold font-mono ${color ? (colors[color] ?? "") : ""}`}>{value}</p>
      {sub && <p className="text-xs text-muted-foreground mt-0.5">{sub}</p>}
    </div>
  );
}
function MetCard({ l, v, c }: { l: string; v: string; c?: string }) {
  return (
    <div className="rounded-xl border border-border/60 bg-card p-3 text-center">
      <p className="text-xs text-muted-foreground mb-1">{l}</p>
      <p className={`font-mono font-bold text-sm ${c ?? ""}`}>{v}</p>
    </div>
  );
}
function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-baseline gap-2 text-xs">
      <span className="text-muted-foreground w-28 shrink-0">{label}</span>
      <span className="font-mono">{children}</span>
    </div>
  );
}
function Badge({ cls, children }: { cls: string; children: React.ReactNode }) {
  return <span className={`rounded-full border px-2 py-0.5 text-xs font-medium ${cls}`}>{children}</span>;
}
function Empty({ msg }: { msg: string }) {
  return (
    <div className="rounded-xl border border-border/60 bg-card p-10 text-center text-muted-foreground text-sm">{msg}</div>
  );
}
