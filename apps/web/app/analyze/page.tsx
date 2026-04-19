"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ScatterChart, Scatter, Cell,
  LineChart, Line, ReferenceLine, ErrorBar,
} from "recharts";
import {
  ArrowLeft, TrendingDown, TrendingUp, BarChart3, GitBranch,
  Lightbulb, Brain, Star, AlertTriangle, CheckCircle2,
  ArrowUpRight, ArrowDownRight, Info, Clock,
} from "lucide-react";
import { useAppStore } from "@/lib/store";
import type { AnalysisBundle, PredictiveResult, CausalEffect, Intervention } from "@/lib/types";

/* ─────────────────────────── helpers ──────────────────────────────────── */
const fmt = (n: number, d = 3) => n.toFixed(d);
const pct = (n: number) => `${(n * 100).toFixed(1)}%`;
const sign = (n: number) => (n >= 0 ? "+" : "") + n.toFixed(3);
const r2Color = (r2: number) =>
  r2 > 0.7 ? "text-green-400" : r2 > 0.4 ? "text-yellow-400" : "text-red-400";

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
  { id: "overview",      icon: BarChart3,  label: "Overview" },
  { id: "predictive",    icon: BarChart3,  label: "Predictive Models" },
  { id: "causal",        icon: GitBranch,  label: "Causal Analysis" },
  { id: "interventions", icon: Lightbulb,  label: "Interventions" },
  { id: "executive",     icon: Brain,      label: "Executive Summary" },
];

/* ─────────────────────────── page ─────────────────────────────────────── */
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
      <div className="text-muted-foreground text-sm">Loading…</div>
    </div>
  );

  const best = analysis.predictive.find((p) => p.is_winner) ?? analysis.predictive[0];
  const sigCausal = analysis.causal.filter((e) => e.p_value < 0.05).length;

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* ── top bar ──────────────────────────────────────────────────── */}
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
            <span className={`font-mono font-bold ${r2Color(best.metrics.r2)}`}>
              R²={fmt(best.metrics.r2)}
            </span>
            <span className="text-muted-foreground hidden md:block">
              {best.display_name} · {analysis.row_count.toLocaleString()} rows · {analysis.runtime_seconds}s
            </span>
          </div>
        </div>
        {/* tabs */}
        <div className="container">
          <nav className="flex overflow-x-auto gap-0 pb-0 scrollbar-hide">
            {TABS.map(({ id, label }) => (
              <button key={id} onClick={() => setTab(id)}
                className={`px-4 py-2.5 text-sm font-medium border-b-2 whitespace-nowrap transition-colors ${
                  tab === id
                    ? "border-primary text-foreground"
                    : "border-transparent text-muted-foreground hover:text-foreground"
                }`}>
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

      {/* ── content ──────────────────────────────────────────────────── */}
      <div className="container py-6 max-w-6xl">
        {tab === "overview"      && <OverviewTab analysis={analysis} best={best} target={target} sigCausal={sigCausal} />}
        {tab === "predictive"    && <PredictiveTab results={analysis.predictive} />}
        {tab === "causal"        && <CausalTab effects={analysis.causal} target={target} />}
        {tab === "interventions" && <InterventionsTab interventions={analysis.interventions} target={target} improve={analysis.predictive[0]?.task ?? "regression"} />}
        {tab === "executive"     && <ExecutiveTab exec={analysis.executive} target={target} />}
      </div>
    </div>
  );
}

/* ═══════════════════════════ OVERVIEW ═══════════════════════════════════ */
function OverviewTab({ analysis, best, target, sigCausal }: {
  analysis: AnalysisBundle; best: PredictiveResult; target: string; sigCausal: number;
}) {
  const topIv = analysis.interventions[0];
  return (
    <div className="space-y-6">
      {/* KPI cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <KpiCard label="Best Model R²" value={fmt(best.metrics.r2)}
          sub={best.display_name}
          color={best.metrics.r2 > 0.6 ? "green" : best.metrics.r2 > 0.3 ? "yellow" : "red"}
          icon={<BarChart3 className="h-4 w-4" />} />
        <KpiCard label="Test RMSE" value={fmt(best.metrics.rmse)} sub={`MAE ${fmt(best.metrics.mae)}`} icon={<BarChart3 className="h-4 w-4" />} />
        <KpiCard label="Causal Levers" value={String(sigCausal)} sub="p < 0.05 significance" color="blue" icon={<GitBranch className="h-4 w-4" />} />
        <KpiCard label="Top Impact" value={topIv ? `${Math.abs(topIv.expected_kpi_change_pct).toFixed(1)}%` : "—"}
          sub={topIv?.feature ?? "No interventions"} color="purple" icon={<Lightbulb className="h-4 w-4" />} />
      </div>

      {/* Model comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Model Comparison" sub="All 5 models on the same 80/20 train/test split">
          <div className="space-y-2 mt-3">
            {analysis.predictive.map((r) => (
              <div key={r.model} className={`flex items-center gap-3 p-2 rounded-lg ${r.is_winner ? "bg-primary/5 border border-primary/20" : ""}`}>
                {r.is_winner && <Star className="h-3.5 w-3.5 text-primary shrink-0" />}
                {!r.is_winner && <div className="h-3.5 w-3.5 shrink-0" />}
                <span className="text-xs font-medium w-28 shrink-0">{r.display_name}</span>
                <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-primary/70 rounded-full" style={{ width: `${Math.max(0, r.metrics.r2) * 100}%` }} />
                </div>
                <span className={`font-mono text-xs w-14 text-right ${r2Color(r.metrics.r2)}`}>{fmt(r.metrics.r2)}</span>
                {r.metrics.cv_r2_mean !== undefined && (
                  <span className="font-mono text-xs text-muted-foreground hidden md:block w-20 text-right">
                    CV {fmt(r.metrics.cv_r2_mean, 2)}
                  </span>
                )}
              </div>
            ))}
          </div>
        </Card>

        {/* Top interventions preview */}
        <Card title="Top Recommendations" sub="Ranked by estimated KPI impact">
          <div className="space-y-2 mt-3">
            {analysis.interventions.slice(0, 5).map((iv) => (
              <div key={iv.feature} className="flex items-center gap-3 p-2 rounded-lg bg-muted/20">
                <span className="h-5 w-5 rounded-full bg-primary/15 text-primary text-xs font-bold flex items-center justify-center shrink-0">
                  {iv.rank}
                </span>
                <span className="text-xs font-mono flex-1 truncate">{iv.feature}</span>
                <span className={`text-xs font-medium ${iv.direction === "decrease" ? "text-blue-400" : "text-green-400"}`}>
                  {iv.direction === "decrease" ? <ArrowDownRight className="h-3.5 w-3.5 inline" /> : <ArrowUpRight className="h-3.5 w-3.5 inline" />}
                  {iv.direction}
                </span>
                <span className={`font-mono text-xs font-bold ${iv.expected_kpi_change < 0 ? "text-green-400" : "text-red-400"}`}>
                  {sign(iv.expected_kpi_change_pct)}%
                </span>
              </div>
            ))}
            {analysis.interventions.length === 0 && (
              <p className="text-xs text-muted-foreground py-4 text-center">No recommendations — add controllable numeric features.</p>
            )}
          </div>
        </Card>
      </div>

      {/* Warnings */}
      {analysis.warnings.length > 0 && (
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

/* ═══════════════════════════ PREDICTIVE ═════════════════════════════════ */
function PredictiveTab({ results }: { results: PredictiveResult[] }) {
  const [sel, setSel] = useState(results.find(r => r.is_winner)?.model ?? results[0]?.model);
  const model = results.find(r => r.model === sel) ?? results[0];
  if (!model) return null;

  const impData = model.importances.slice(0, 12).map(f => ({
    name: f.feature.length > 20 ? f.feature.slice(0, 20) + "…" : f.feature,
    value: parseFloat(f.importance_norm.toFixed(4)),
  }));

  return (
    <div className="space-y-6">
      {/* Model selector */}
      <div className="flex flex-wrap gap-2">
        {results.map(r => (
          <button key={r.model} onClick={() => setSel(r.model)}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium border transition-colors ${
              sel === r.model ? "bg-primary text-primary-foreground border-primary" : "border-border hover:bg-accent"
            }`}>
            {r.is_winner && "★ "}{r.display_name}
            <span className={`ml-2 font-mono text-xs opacity-80 ${r2Color(r.metrics.r2)}`}>
              R²={fmt(r.metrics.r2)}
            </span>
          </button>
        ))}
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { l: "Test R²",    v: fmt(model.metrics.r2),      c: r2Color(model.metrics.r2) },
          { l: "Adj. R²",   v: model.metrics.adj_r2 != null ? fmt(model.metrics.adj_r2) : "—" },
          { l: "RMSE",       v: fmt(model.metrics.rmse) },
          { l: "CV R² (3-fold)", v: model.metrics.cv_r2_mean != null
              ? `${fmt(model.metrics.cv_r2_mean, 2)} ±${fmt(model.metrics.cv_r2_std ?? 0, 2)}` : "—" },
        ].map(m => (
          <div key={m.l} className="rounded-xl border border-border/60 bg-card p-3 text-center">
            <p className="text-xs text-muted-foreground mb-1">{m.l}</p>
            <p className={`font-mono font-bold text-sm ${m.c ?? ""}`}>{m.v}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Feature importances */}
        <Card title="Feature Importances" sub="Normalised — 1.0 = most important feature">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart layout="vertical" data={impData} margin={{ left: 4, right: 20, top: 8, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 33% 18%)" />
              <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 10, fill: "hsl(215 20% 55%)" }} />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fill: "hsl(215 20% 55%)" }} width={130} />
              <Tooltip formatter={(v: number) => v.toFixed(4)} contentStyle={{ background: "hsl(222 47% 10%)", border: "1px solid hsl(217 33% 18%)", borderRadius: 8, fontSize: 12 }} />
              <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                {impData.map((_, i) => <Cell key={i} fill={i === 0 ? "hsl(217 91% 60%)" : "hsl(217 91% 60% / 0.5)"} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* Actual vs Predicted */}
        <Card title="Actual vs Predicted" sub="Test set — ideal fit = diagonal line">
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart margin={{ left: 4, right: 20, top: 8, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 33% 18%)" />
              <XAxis dataKey="actual" name="Actual" type="number" tick={{ fontSize: 10, fill: "hsl(215 20% 55%)" }}
                label={{ value: "Actual", position: "insideBottom", offset: -12, fontSize: 10, fill: "hsl(215 20% 55%)" }} />
              <YAxis dataKey="predicted" name="Predicted" type="number" tick={{ fontSize: 10, fill: "hsl(215 20% 55%)" }}
                label={{ value: "Predicted", angle: -90, position: "insideLeft", fontSize: 10, fill: "hsl(215 20% 55%)" }} />
              <Tooltip contentStyle={{ background: "hsl(222 47% 10%)", border: "1px solid hsl(217 33% 18%)", borderRadius: 8, fontSize: 12 }}
                formatter={(v: number) => v.toFixed(3)} />
              <Scatter data={model.predictions.slice(0, 300)} fill="hsl(217 91% 60%)" fillOpacity={0.5} />
            </ScatterChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Coefficients table */}
      {model.coefficients && model.coefficients.length > 1 && (
        <Card title="Regression Coefficients" sub="Standardised features — coefficients are comparable in magnitude">
          <div className="overflow-x-auto mt-3">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border/40 text-muted-foreground">
                  {["Feature","Coefficient","Std Error","t-stat","p-value","Sig"].map(h => (
                    <th key={h} className="px-3 py-2 text-left font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {model.coefficients.filter(c => c.feature !== "(intercept)").slice(0, 20).map(c => (
                  <tr key={c.feature} className={`border-b border-border/20 hover:bg-muted/10 ${!c.significant ? "opacity-50" : ""}`}>
                    <td className="px-3 py-1.5 font-mono">{c.feature}</td>
                    <td className={`px-3 py-1.5 font-mono font-bold ${c.coef > 0 ? "text-red-400" : "text-green-400"}`}>{sign(c.coef)}</td>
                    <td className="px-3 py-1.5 font-mono text-muted-foreground">{fmt(c.std_err)}</td>
                    <td className="px-3 py-1.5 font-mono">{fmt(c.t_stat, 2)}</td>
                    <td className={`px-3 py-1.5 font-mono ${c.p_value < 0.05 ? "text-green-400" : "text-muted-foreground"}`}>
                      {c.p_value < 0.001 ? "<0.001" : fmt(c.p_value)}
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

/* ═══════════════════════════ CAUSAL ═════════════════════════════════════ */
function CausalTab({ effects, target }: { effects: CausalEffect[]; target: string }) {
  if (!effects.length) return (
    <div className="rounded-xl border border-border/60 bg-card p-10 text-center text-muted-foreground text-sm">
      No causal effects computed. Ensure controllable numeric columns exist with at least 30 non-missing rows.
    </div>
  );

  const maxAbs = Math.max(...effects.map(e => Math.abs(e.effect_per_std)), 0.01);
  const chartData = effects.slice(0, 10).map(e => ({
    name: e.feature.length > 18 ? e.feature.slice(0, 18) + "…" : e.feature,
    effect: parseFloat(e.effect_per_std.toFixed(4)),
    ci_lo: parseFloat((e.effect_per_std - 1.96 * e.std_err).toFixed(4)),
    ci_hi: parseFloat((e.effect_per_std + 1.96 * e.std_err).toFixed(4)),
    sig: e.p_value < 0.05,
  }));

  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-blue-500/20 bg-blue-500/5 p-4 text-xs text-muted-foreground leading-relaxed">
        <strong className="text-blue-400">Method:</strong> Back-door adjusted OLS via statsmodels.
        For each controllable variable: <code className="bg-muted px-1 rounded">{target} ~ feature + confounders + DAG_parents + context</code>.
        β = effect of <strong>+1 standard deviation</strong> increase in the feature on {target} (also standardised).
        Mediators are excluded from adjustment sets. These are observational estimates — unobserved confounders may bias results.
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Effect size chart */}
        <Card title="Adjusted Effect Sizes" sub="β per +1 SD change · bars = 95% CI · grey = not significant">
          <ResponsiveContainer width="100%" height={Math.max(200, effects.slice(0, 10).length * 36 + 40)}>
            <BarChart layout="vertical" data={chartData} margin={{ left: 4, right: 24, top: 8, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 33% 18%)" />
              <XAxis type="number" tick={{ fontSize: 10, fill: "hsl(215 20% 55%)" }} />
              <ReferenceLine x={0} stroke="hsl(215 20% 55%)" strokeDasharray="3 3" />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fill: "hsl(215 20% 55%)" }} width={130} />
              <Tooltip formatter={(v: number) => v.toFixed(4)} contentStyle={{ background: "hsl(222 47% 10%)", border: "1px solid hsl(217 33% 18%)", borderRadius: 8, fontSize: 12 }} />
              <Bar dataKey="effect" radius={[0, 4, 4, 0]}>
                {chartData.map((d, i) => (
                  <Cell key={i} fill={
                    !d.sig ? "hsl(215 20% 40%)" :
                    d.effect > 0 ? "hsl(0 63% 55%)" : "hsl(142 71% 45%)"
                  } />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* Significance table */}
        <Card title="Inference Table" sub="p-value · 95% CI · adjustment variables">
          <div className="overflow-x-auto mt-3 max-h-80 overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-card">
                <tr className="border-b border-border/40 text-muted-foreground">
                  <th className="px-3 py-2 text-left">Feature</th>
                  <th className="px-3 py-2 text-right">β/SD</th>
                  <th className="px-3 py-2 text-right">p-val</th>
                  <th className="px-3 py-2 text-left">Strength</th>
                </tr>
              </thead>
              <tbody>
                {effects.map(e => (
                  <tr key={e.feature} className="border-b border-border/20 hover:bg-muted/10 transition-colors">
                    <td className="px-3 py-2 font-mono font-semibold">{e.feature}</td>
                    <td className={`px-3 py-2 text-right font-mono font-bold ${e.effect_per_std > 0 ? "text-red-400" : "text-green-400"}`}>
                      {sign(e.effect_per_std)}
                    </td>
                    <td className={`px-3 py-2 text-right font-mono ${e.p_value < 0.05 ? "text-green-400" : "text-muted-foreground"}`}>
                      {e.p_value < 0.001 ? "<.001" : fmt(e.p_value)}
                    </td>
                    <td className="px-3 py-2">
                      <span className={`rounded-full border px-1.5 py-0.5 text-xs ${STRENGTH[e.evidence_strength]}`}>
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

      {/* Warnings */}
      {effects.filter(e => e.warning).map(e => (
        <div key={e.feature} className="rounded-lg border border-yellow-500/30 bg-yellow-500/5 p-3 text-xs flex gap-2">
          <AlertTriangle className="h-3.5 w-3.5 text-yellow-400 shrink-0 mt-0.5" />
          <span><strong className="text-yellow-400">{e.feature}:</strong> {e.warning}</span>
        </div>
      ))}
    </div>
  );
}

/* ═══════════════════════════ INTERVENTIONS ══════════════════════════════ */
function InterventionsTab({ interventions, target }: {
  interventions: Intervention[]; target: string; improve: string;
}) {
  if (!interventions.length) return (
    <div className="rounded-xl border border-border/60 bg-card p-10 text-center text-muted-foreground text-sm">
      No intervention recommendations generated. Ensure controllable numeric features exist with meaningful variance.
    </div>
  );

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-border/60 bg-card p-4 text-xs text-muted-foreground leading-relaxed">
        <strong className="text-foreground">How recommendations are generated:</strong> A GBR model simulates shifting each controllable variable by ±1 SD while holding others constant.
        Where back-door adjusted OLS is statistically significant (p&lt;0.05), the estimate is labelled <span className="text-blue-400 font-medium">causal</span>.
        Otherwise it is labelled <span className="text-purple-400 font-medium">predictive</span>.
        All estimates are observational — validate with experiments before large-scale changes.
      </div>

      {interventions.map(iv => <InterventionCard key={iv.feature} iv={iv} target={target} />)}
    </div>
  );
}

function InterventionCard({ iv, target }: { iv: Intervention; target: string }) {
  const [open, setOpen] = useState(false);
  const improving = iv.expected_kpi_change < 0;

  return (
    <div className="rounded-xl border border-border/60 bg-card overflow-hidden hover:border-border transition-colors">
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-3 bg-muted/10 border-b border-border/40">
        <span className="h-7 w-7 rounded-full bg-primary/15 text-primary text-sm font-bold flex items-center justify-center shrink-0">
          {iv.rank}
        </span>
        <div className="flex-1 min-w-0">
          <p className="font-semibold text-sm font-mono truncate">{iv.feature}</p>
        </div>
        <div className="flex items-center gap-2 flex-wrap justify-end">
          <Badge style={EVTYPE[iv.evidence_type]}>{iv.evidence_type}</Badge>
          <Badge style={STRENGTH[iv.evidence_strength]}>{iv.evidence_strength}</Badge>
          <span className={`font-mono font-bold text-sm ${improving ? "text-green-400" : "text-red-400"}`}>
            {iv.expected_kpi_change > 0 ? "+" : ""}{iv.expected_kpi_change_pct.toFixed(1)}% {target}
          </span>
        </div>
      </div>

      {/* Body */}
      <div className="p-4 grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
        {/* Numbers */}
        <div className="space-y-2">
          <Row label="Action">
            <span className={`font-semibold ${iv.direction === "decrease" ? "text-blue-400" : "text-green-400"}`}>
              {iv.direction === "decrease"
                ? <><ArrowDownRight className="h-3.5 w-3.5 inline mr-0.5" />Decrease</>
                : <><ArrowUpRight className="h-3.5 w-3.5 inline mr-0.5" />Increase</>}
            </span>
          </Row>
          <Row label="Current mean"><code>{iv.current_mean.toFixed(3)}</code>
            <span className="text-muted-foreground ml-1">[p10={iv.current_p10.toFixed(2)}, p90={iv.current_p90.toFixed(2)}]</span>
          </Row>
          <Row label="Suggested value">
            <code className="text-primary font-bold">{iv.suggested_value.toFixed(3)}</code>
            <span className="text-muted-foreground ml-1">({iv.delta > 0 ? "+" : ""}{iv.delta.toFixed(3)}, {iv.delta_pct.toFixed(1)}%)</span>
          </Row>
          <Row label="Est. KPI change">
            <code className={improving ? "text-green-400 font-bold" : "text-red-400 font-bold"}>
              {iv.expected_kpi_change > 0 ? "+" : ""}{iv.expected_kpi_change.toFixed(4)}
            </code>
          </Row>
        </div>

        {/* Qualitative */}
        <div className="space-y-2">
          <div>
            <p className="text-muted-foreground font-medium mb-0.5">Rationale</p>
            <p className="leading-relaxed">{iv.rationale}</p>
          </div>
          <div>
            <p className="text-muted-foreground font-medium mb-0.5">Tradeoff</p>
            <p className="leading-relaxed text-yellow-400/80">{iv.tradeoff}</p>
          </div>
        </div>
      </div>

      {/* Expandable assumptions */}
      <button onClick={() => setOpen(o => !o)}
        className="w-full px-4 pb-2 text-left text-xs text-muted-foreground hover:text-foreground flex items-center gap-1 transition-colors">
        <Info className="h-3 w-3" /> {open ? "Hide" : "Show"} assumptions &amp; caveats
      </button>
      {open && (
        <div className="px-4 pb-4 border-t border-border/30 pt-3 space-y-1 text-xs text-muted-foreground">
          {iv.assumptions.map((a, i) => <p key={i}>· {a}</p>)}
          <p className="italic mt-2 opacity-70">{iv.caveat}</p>
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════ EXECUTIVE ══════════════════════════════════ */
function ExecutiveTab({ exec, target }: { exec: AnalysisBundle["executive"]; target: string }) {
  return (
    <div className="max-w-2xl mx-auto space-y-5">
      <div className="rounded-xl border border-primary/30 bg-primary/5 p-6 text-center">
        <h2 className="text-xl font-bold mb-2">{exec.headline}</h2>
        <p className="text-sm text-muted-foreground">{exec.sub_headline}</p>
      </div>

      <Card title="Key Findings">
        <ul className="mt-3 space-y-2">
          {exec.bullets.map((b, i) => (
            <li key={i} className="flex gap-3 text-sm">
              <CheckCircle2 className="h-4 w-4 text-primary shrink-0 mt-0.5" />
              <span>{b}</span>
            </li>
          ))}
        </ul>
      </Card>

      {exec.top_levers.length > 0 && (
        <Card title="Top Levers to Pull">
          <div className="flex flex-wrap gap-2 mt-3">
            {exec.top_levers.map(l => (
              <span key={l} className="rounded-lg border border-green-500/30 bg-green-500/10 px-3 py-1 text-sm font-mono text-green-400">
                {l}
              </span>
            ))}
          </div>
        </Card>
      )}

      <Card title="Important Caveats">
        <ul className="mt-3 space-y-2">
          {exec.cautions.map((c, i) => (
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

/* ═══════════════════════════ shared UI ══════════════════════════════════ */
function Card({ title, sub, children }: { title: string; sub?: string; children?: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-border/60 bg-card p-5">
      <p className="font-semibold text-sm">{title}</p>
      {sub && <p className="text-xs text-muted-foreground mt-0.5">{sub}</p>}
      {children}
    </div>
  );
}

function KpiCard({ label, value, sub, color, icon }: {
  label: string; value: string; sub?: string; color?: string; icon?: React.ReactNode;
}) {
  const colors: Record<string, string> = {
    green: "text-green-400", yellow: "text-yellow-400",
    red: "text-red-400", blue: "text-blue-400", purple: "text-purple-400",
  };
  return (
    <div className="rounded-xl border border-border/60 bg-card p-4">
      <div className="flex items-center gap-2 mb-1">
        {icon && <span className="text-muted-foreground">{icon}</span>}
        <p className="text-xs text-muted-foreground">{label}</p>
      </div>
      <p className={`text-2xl font-bold font-mono ${color ? colors[color] : ""}`}>{value}</p>
      {sub && <p className="text-xs text-muted-foreground mt-0.5">{sub}</p>}
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-baseline gap-2">
      <span className="text-muted-foreground w-28 shrink-0">{label}</span>
      <span className="font-mono">{children}</span>
    </div>
  );
}

function Badge({ style, children }: { style: string; children: React.ReactNode }) {
  return (
    <span className={`rounded-full border px-2 py-0.5 text-xs font-medium ${style}`}>
      {children}
    </span>
  );
}
