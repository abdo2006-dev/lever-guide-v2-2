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
import type {
  AnalysisBundle, PredictiveResult, CausalEffect, Intervention,
  CopilotAnswerResponse, ModelStatus, ResultType,
} from "@/lib/types";
import { RESULT_TYPE_LABEL, RESULT_TYPE_NOTE } from "@/lib/types";

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

/* Whether the adjusted estimate agrees with the simulated direction. */
const ADJ_SUPPORT: Record<string, string> = {
  aligned:      "bg-blue-500/20 text-blue-400 border-blue-500/30",
  conflicting:  "bg-red-500/20 text-red-400 border-red-500/30",
  inconclusive: "bg-gray-500/20 text-gray-400 border-gray-600/30",
  none:         "bg-purple-500/20 text-purple-400 border-purple-500/30",
};
const ADJ_SUPPORT_LABEL: Record<string, string> = {
  aligned:      "adjusted estimate agrees",
  conflicting:  "adjusted estimate disagrees",
  inconclusive: "adjusted estimate inconclusive",
  none:         "no adjusted estimate",
};

const IV_STATUS: Record<string, string> = {
  eligible:             "bg-green-500/20 text-green-400 border-green-500/30",
  exploratory:          "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  unsupported:          "bg-orange-500/20 text-orange-400 border-orange-500/30",
  infeasible:           "bg-red-500/20 text-red-400 border-red-500/30",
  conflicting_evidence: "bg-red-500/20 text-red-400 border-red-500/30",
};
const IV_STATUS_LABEL: Record<string, string> = {
  eligible:             "eligible",
  exploratory:          "exploratory",
  unsupported:          "unsupported",
  infeasible:           "infeasible",
  conflicting_evidence: "conflicting evidence",
};

const SUPPORT_LABEL: Record<string, string> = {
  within_observed:                  "inside observed range",
  outside_observed_within_declared: "outside observed range (extrapolating)",
  outside_declared:                 "outside declared operating range",
  unknown:                          "support unknown",
};

const MODEL_STATUS: Record<string, string> = {
  succeeded:                "text-green-400",
  unavailable_dependency:   "text-orange-400",
  training_failed:          "text-red-400",
  skipped_by_configuration: "text-muted-foreground",
};
const MODEL_STATUS_LABEL: Record<string, string> = {
  succeeded:                "ran",
  unavailable_dependency:   "did not run — library unavailable here",
  training_failed:          "did not run — training failed",
  skipped_by_configuration: "skipped by configuration",
};

const RESULT_TYPE_STYLE: Record<ResultType, string> = {
  association:              "bg-slate-500/20 text-slate-300 border-slate-500/30",
  adjusted_effect_estimate: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  predictive_what_if:       "bg-purple-500/20 text-purple-400 border-purple-500/30",
};

const TABS = [
  { id: "overview",      label: "Overview" },
  { id: "predictive",    label: "Predictive Models" },
  { id: "causal",        label: "Adjusted Effect Estimates" },
  { id: "interventions", label: "What-If Simulations" },
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
  // Count estimates whose interval excludes zero, not those under p<0.05:
  // an interval that straddles zero does not establish a direction.
  const sigN   = analysis.causal.filter(e => e.interval_excludes_zero).length;
  const eligible = analysis.interventions.filter(iv => iv.status === "eligible");
  const topIv  = eligible[0];

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
                {id === "interventions" && eligible.length > 0 && (
                  <span className="ml-1.5 rounded-full bg-primary/20 text-primary text-xs px-1.5 py-0.5">
                    {eligible.length}
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
        {tab === "causal"        && <CausalTab      effects={analysis.causal} target={target} dag={analysis.dag_validation} />}
        {tab === "interventions" && <InterventionsTab interventions={analysis.interventions} target={target} />}
        {tab === "executive"     && <ExecutiveTab   exec={analysis.executive} prov={analysis.provenance} />}
        {tab === "copilot"       && <CopilotTab     analysis={analysis} />}
      </div>
    </div>
  );
}

/* ═══════════════ OVERVIEW ═══════════════════════════════════════════════ */
function OverviewTab({ analysis, best, sigN }: {
  analysis: AnalysisBundle; best: PredictiveResult;
  sigN: number; topIv: Intervention | undefined;
}) {
  const eligible = analysis.interventions.filter(iv => iv.status === "eligible");
  const demoted  = analysis.interventions.filter(iv => iv.status !== "eligible");
  const statuses = analysis.model_statuses ?? [];
  const ranModels = statuses.filter(s => s.status === "succeeded").length || analysis.predictive.length;
  const allModels = statuses.length || analysis.predictive.length;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <KpiCard label="Best Model R²" value={fmt(best?.metrics?.r2)}
          sub={`${best?.display_name ?? "—"} · ${ranModels}/${allModels} models ran`}
          color={n(best?.metrics?.r2) > 0.6 ? "green" : n(best?.metrics?.r2) > 0.3 ? "yellow" : "red"} />
        <KpiCard label="Test RMSE"  value={fmt(best?.metrics?.rmse)} sub={`MAE ${fmt(best?.metrics?.mae)}`} />
        <KpiCard label="Directional Estimates" value={String(sigN)}
          sub="95% interval excludes zero" color="blue" />
        <KpiCard label="Eligible Candidates" value={String(eligible.length)}
          sub={`${demoted.length} set aside`} color="purple" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model comparison */}
        <Card
          title="Model Comparison"
          sub={`Same random train/test split (${best?.metrics?.n_train ?? "?"} train / ${best?.metrics?.n_test ?? "?"} test) — best by test R² marked ★`}
        >
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

          {/* Which models actually ran — a model that failed is never omitted */}
          {(analysis.model_statuses ?? []).some(s => s.status !== "succeeded") && (
            <div className="mt-3 rounded-lg border border-border/50 bg-muted/10 p-3 space-y-1">
              <p className="text-xs font-medium">Model run status</p>
              {(analysis.model_statuses ?? []).map((s: ModelStatus) => (
                <p key={s.model} className="text-xs">
                  <span className="font-mono">{s.display_name}</span>{" "}
                  <span className={MODEL_STATUS[s.status] ?? ""}>
                    {MODEL_STATUS_LABEL[s.status] ?? s.status}
                  </span>
                  {s.detail && (
                    <span className="text-muted-foreground"> — {s.detail}</span>
                  )}
                </p>
              ))}
            </div>
          )}

          <p className="text-xs text-muted-foreground mt-3">
            R² = fraction of variance explained (0–1, higher = better).
            CV R² = 3-fold cross-validated score (±σ shows stability).
            The winner is chosen on the same held-out set its score is reported
            on, so that score is optimistic. The split is random, not grouped by
            machine and not time-ordered.
          </p>
        </Card>

        {/* Eligible candidates only */}
        <Card
          title="Eligible Candidate Changes"
          sub="Predictive what-if simulations that passed feasibility, support and evidence-agreement screening"
        >
          <div className="space-y-2 mt-3">
            {eligible.slice(0, 5).map(iv => (
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
            {eligible.length === 0 && (
              <p className="text-xs text-muted-foreground py-4 text-center">
                {demoted.length > 0
                  ? `No candidate passed screening. ${demoted.length} were assessed and set aside — see the What-If Simulations tab for the reason on each.`
                  : "No candidate changes were produced. Assign at least one controllable numeric column in Setup."}
              </p>
            )}
            {demoted.length > 0 && eligible.length > 0 && (
              <p className="text-xs text-muted-foreground pt-1">
                {demoted.length} further candidate(s) were assessed and set aside.
              </p>
            )}
          </div>
          <p className="text-xs text-muted-foreground mt-3">
            {RESULT_TYPE_NOTE.predictive_what_if}
          </p>
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
function CausalTab({ effects, target, dag }: {
  effects: CausalEffect[]; target: string;
  dag?: AnalysisBundle["dag_validation"];
}) {
  const [openMethod, setOpenMethod] = useState(false);
  if (!effects?.length) return (
    <Empty msg="No adjusted effect estimates were computed. This needs at least one lever column that is numeric with 30+ non-missing rows, and at least one adjuster (confounder or context)." />
  );

  const chartData = effects.slice(0, 10).map(e => ({
    name: (e.feature ?? "").length > 20 ? (e.feature ?? "").slice(0, 20) + "…" : (e.feature ?? ""),
    effect: parseFloat(n(e.effect_per_std).toFixed(4)),
    lo: parseFloat(n(e.conf_int_lo).toFixed(4)),
    hi: parseFloat(n(e.conf_int_hi).toFixed(4)),
    directional: e.interval_excludes_zero,
  }));

  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-blue-500/20 bg-blue-500/5 p-4 text-xs text-muted-foreground leading-relaxed space-y-2">
        <p>
          <Badge cls={RESULT_TYPE_STYLE.adjusted_effect_estimate}>
            {RESULT_TYPE_LABEL.adjusted_effect_estimate}
          </Badge>
        </p>
        <p>
          β is the change in <strong>{target}</strong> (in standard deviations) per +1 SD
          of the lever, holding its adjustment set constant.{" "}
          <strong className="text-blue-400">{RESULT_TYPE_NOTE.adjusted_effect_estimate}</strong>
        </p>
        {dag?.graph_assumption && (
          <p className="text-muted-foreground/90">{dag.graph_assumption}</p>
        )}
        <button
          onClick={() => setOpenMethod(o => !o)}
          className="text-xs text-blue-400 hover:underline flex items-center gap-1"
        >
          <Info className="h-3 w-3" /> {openMethod ? "Hide" : "Show"} method details
        </button>
        {openMethod && (
          <div className="pt-2 border-t border-blue-500/20 space-y-2">
            <p>
              For each lever we fit{" "}
              <code className="bg-muted px-1 rounded">{target} ~ lever + adjustment set</code>{" "}
              by ordinary least squares on standardised columns. The adjustment
              set is either <strong>declared</strong> by this dataset&apos;s ontology,
              or <strong>derived</strong> from the assumed graph and the roles you
              assigned — each estimate below says which, and lists the set it used.
              No back-door search is performed and no minimal valid set is claimed.
            </p>
            <p>
              Any column labelled <em>mediator</em> is removed from every adjustment
              set, because conditioning on it turns a total effect into a direct
              effect. Labelling mediators correctly is part of the configuration —
              a mediator you leave labelled &quot;confounder&quot; will be adjusted for.
              Each estimate lists the mediators it dropped, if any.
            </p>
            <p>
              95% intervals and p-values are ordinary OLS inference assuming
              independent, homoskedastic errors. For clustered panel data —
              repeated intervals on the same machines — these intervals are too
              narrow. There are no fixed effects and no cluster-robust standard
              errors.
            </p>
            <p>
              Evidence strength is capped at &quot;weak&quot; whenever the interval
              includes zero, regardless of the p-value.
            </p>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Adjusted Effect Sizes (β/SD)" sub="Red = raises the outcome · Green = lowers it · Grey = interval includes zero, no direction established">
          <ResponsiveContainer width="100%" height={Math.max(200, chartData.length * 38 + 40)}>
            <BarChart layout="vertical" data={chartData} margin={{ left: 4, right: 24, top: 8, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 33% 18%)" />
              <XAxis type="number" tick={{ fontSize: 10, fill: "hsl(215 20% 55%)" }} />
              <ReferenceLine x={0} stroke="hsl(215 20% 55%)" strokeDasharray="4 4" />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fill: "hsl(215 20% 55%)" }} width={140} />
              <Tooltip formatter={(v: number) => v.toFixed(4)} contentStyle={{ background: "hsl(222 47% 10%)", border: "1px solid hsl(217 33% 18%)", borderRadius: 8, fontSize: 12 }} />
              <Bar dataKey="effect" radius={[0, 4, 4, 0]}>
                {chartData.map((d, i) => (
                  <Cell key={i} fill={!d.directional ? "hsl(215 20% 35%)" : d.effect > 0 ? "hsl(0 63% 55%)" : "hsl(142 71% 45%)"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="text-xs text-muted-foreground mt-2">
            Bars show point estimates only; the 95% intervals are in the table.
            A grey bar means the interval includes zero.
          </p>
        </Card>

        <Card title="Inference Table" sub="β per +1 SD · 95% interval · what was adjusted for">
          <div className="overflow-x-auto mt-3 max-h-96 overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-card">
                <tr className="border-b border-border/40 text-muted-foreground">
                  <th className="px-3 py-2 text-left">Lever</th>
                  <th className="px-3 py-2 text-right">β/SD</th>
                  <th className="px-3 py-2 text-right">95% CI</th>
                  <th className="px-3 py-2 text-right">p-val</th>
                  <th className="px-3 py-2 text-left">Evidence</th>
                </tr>
              </thead>
              <tbody>
                {effects.map(e => (
                  <tr key={e.feature} className="border-b border-border/20 hover:bg-muted/10 transition-colors align-top">
                    <td className="px-3 py-2 font-mono font-semibold text-xs">
                      {e.feature ?? "—"}
                      {e.causal_role && (
                        <span className="block text-muted-foreground font-sans font-normal">
                          {e.causal_role.replace(/_/g, " ")}
                        </span>
                      )}
                    </td>
                    <td className={`px-3 py-2 text-right font-mono font-bold ${
                      !e.interval_excludes_zero ? "text-muted-foreground"
                        : n(e.effect_per_std) > 0 ? "text-red-400" : "text-green-400"}`}>
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

      {/* What each estimate actually adjusted for — the claim lives or dies here */}
      <Card title="Adjustment sets" sub="Every estimate above, and exactly what it conditioned on">
        <div className="mt-3 space-y-3">
          {effects.map(e => (
            <div key={e.feature} className="rounded-lg border border-border/40 bg-muted/10 p-3 text-xs space-y-1">
              <div className="flex items-center justify-between gap-2 flex-wrap">
                <span className="font-mono font-semibold">{e.feature}</span>
                <Badge cls={e.adjustment_set_source === "declared_domain_dag"
                  ? "bg-blue-500/20 text-blue-400 border-blue-500/30"
                  : "bg-slate-500/20 text-slate-300 border-slate-500/30"}>
                  {e.adjustment_set_source === "declared_domain_dag"
                    ? "declared by the dataset ontology"
                    : "derived from the assumed graph"}
                </Badge>
              </div>
              <p className="text-muted-foreground">
                Estimand: {e.estimand} · n = {e.n_observations?.toLocaleString?.() ?? e.n_observations} ·
                interval method: {e.interval_method}
              </p>
              <p>
                <span className="text-muted-foreground">Adjusted for:</span>{" "}
                {e.adjusted_for?.length
                  ? <span className="font-mono">{e.adjusted_for.join(", ")}</span>
                  : <span className="text-yellow-400">nothing — this estimate is unadjusted</span>}
              </p>
              {(e.notes ?? []).map((note, i) => (
                <p key={i} className="text-muted-foreground/80">· {note}</p>
              ))}
            </div>
          ))}
        </div>
      </Card>

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
  const [openMethod, setOpenMethod] = useState(false);
  if (!interventions?.length) return (
    <Empty msg="No what-if simulations were produced. This needs at least one controllable numeric column in Setup." />
  );
  const eligible = interventions.filter(iv => iv.status === "eligible");
  const demoted  = interventions.filter(iv => iv.status !== "eligible");

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-border/60 bg-card p-4 text-xs text-muted-foreground leading-relaxed space-y-2">
        <p>
          <Badge cls={RESULT_TYPE_STYLE.predictive_what_if}>
            {RESULT_TYPE_LABEL.predictive_what_if}
          </Badge>
        </p>
        <p>
          <strong className="text-foreground">{RESULT_TYPE_NOTE.predictive_what_if}</strong>{" "}
          Each candidate is screened for physical feasibility, whether the value
          is inside the range the model has seen, and whether the adjusted effect
          estimate agrees with the direction. Only candidates that clear all
          three are ranked.
        </p>
        <button
          onClick={() => setOpenMethod(o => !o)}
          className="text-xs text-primary hover:underline flex items-center gap-1"
        >
          <Info className="h-3 w-3" /> {openMethod ? "Hide" : "Show"} method details
        </button>
        {openMethod && (
          <div className="pt-2 border-t border-border/40 space-y-2">
            <p>
              A gradient-boosted regressor is fitted on the analysed rows. For each
              lever, its column is set to a value about one standard deviation from
              its mean — clipped to declared physical limits and the observed range —
              and the mean prediction is compared with the baseline.
            </p>
            <p>
              <strong>Every other column keeps its own observed value per row.</strong>{" "}
              Only the lever changes, and the change is averaged across rows. Columns
              physically coupled to the lever are not updated, which is why coupling
              constraints are checked separately.
            </p>
            <p>
              The model is fitted and evaluated on the same rows, so magnitudes are
              optimistic relative to new data. Intervals are row-resampling
              percentile intervals that hold the fitted model fixed: they capture
              variation across production intervals, not uncertainty in the model,
              so the true interval is wider.
            </p>
          </div>
        )}
      </div>

      <div>
        <h2 className="text-sm font-semibold mb-1">
          Eligible candidate changes <span className="text-muted-foreground font-normal">({eligible.length})</span>
        </h2>
        <p className="text-xs text-muted-foreground mb-3">
          Ranked by simulated magnitude. Validate with a controlled test before acting.
        </p>
        <div className="space-y-4">
          {eligible.map(iv => <IvCard key={iv.feature} iv={iv} target={target} />)}
          {eligible.length === 0 && (
            <Empty msg="No candidate passed screening. Every simulation that was run is listed below with the reason it was set aside." />
          )}
        </div>
      </div>

      {demoted.length > 0 && (
        <div className="pt-2">
          <h2 className="text-sm font-semibold mb-1">
            Assessed and set aside <span className="text-muted-foreground font-normal">({demoted.length})</span>
          </h2>
          <p className="text-xs text-muted-foreground mb-3">
            These simulations ran and their numbers are shown, but they are not
            offered as actions. They are diagnostics: a rejected candidate is
            information about the process, not an error.
          </p>
          <div className="space-y-4">
            {demoted.map(iv => <IvCard key={iv.feature} iv={iv} target={target} />)}
          </div>
        </div>
      )}
    </div>
  );
}

function IvCard({ iv, target }: { iv: Intervention; target: string }) {
  const [open, setOpen] = useState(false);
  const improving = n(iv.expected_kpi_change) < 0;
  const hasInterval =
    iv.expected_kpi_change_lo != null && iv.expected_kpi_change_hi != null;
  const actionable = iv.status === "eligible";

  return (
    <div className={`rounded-xl border bg-card overflow-hidden ${
      actionable ? "border-border/60" : "border-border/40 opacity-95"}`}>
      <div className="flex items-center gap-3 px-4 py-3 bg-muted/10 border-b border-border/40">
        <span className={`h-7 w-7 rounded-full text-sm font-bold flex items-center justify-center shrink-0 ${
          iv.rank ? "bg-primary/15 text-primary" : "bg-muted text-muted-foreground"}`}>
          {iv.rank || "—"}
        </span>
        <div className="flex-1 min-w-0">
          <p className="font-semibold text-sm font-mono truncate">{iv.feature}</p>
        </div>
        <div className="flex items-center gap-2 flex-wrap justify-end">
          <Badge cls={IV_STATUS[iv.status] ?? ""}>{IV_STATUS_LABEL[iv.status] ?? iv.status}</Badge>
          <Badge cls={ADJ_SUPPORT[iv.adjustment_support] ?? ""}>
            {ADJ_SUPPORT_LABEL[iv.adjustment_support] ?? iv.adjustment_support}
          </Badge>
          {actionable && (
            <Badge cls={STRENGTH[iv.evidence_strength] ?? ""}>{iv.evidence_strength}</Badge>
          )}
          <span className={`font-mono font-bold text-sm ${
            !actionable ? "text-muted-foreground" : improving ? "text-green-400" : "text-red-400"}`}>
            {n(iv.expected_kpi_change) > 0 ? "+" : ""}{n(iv.expected_kpi_change_pct).toFixed(1)}% {target}
          </span>
        </div>
      </div>

      {/* Why this candidate is, or is not, offered as an action */}
      <div className={`px-4 py-2 text-xs border-b border-border/30 ${
        actionable ? "text-muted-foreground" : "text-yellow-400/90 bg-yellow-500/5"}`}>
        {!actionable && <AlertTriangle className="h-3.5 w-3.5 inline mr-1.5 mb-0.5" />}
        {iv.status_reason}
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
          <Row label="Simulated value">
            <code className={actionable ? "text-primary font-bold" : "font-bold"}>
              {n(iv.suggested_value).toFixed(3)}
            </code>
            <span className="text-muted-foreground ml-1">
              ({n(iv.delta) >= 0 ? "+" : ""}{n(iv.delta).toFixed(3)}, {n(iv.delta_pct).toFixed(1)}%)
            </span>
          </Row>
          <Row label="Support">
            <span className={iv.support_status === "within_observed"
              ? "text-muted-foreground" : "text-yellow-400"}>
              {SUPPORT_LABEL[iv.support_status] ?? iv.support_status}
            </span>
          </Row>
          <Row label="Simulated change">
            <code className={
              !actionable ? "text-muted-foreground font-bold"
                : improving ? "text-green-400 font-bold" : "text-red-400 font-bold"}>
              {n(iv.expected_kpi_change) > 0 ? "+" : ""}{n(iv.expected_kpi_change).toFixed(4)}
            </code>
          </Row>
          {/* A confidence-interval badge is never rendered without an interval. */}
          <Row label="95% interval">
            {hasInterval ? (
              <span className="text-muted-foreground">
                [{n(iv.expected_kpi_change_lo).toFixed(4)}, {n(iv.expected_kpi_change_hi).toFixed(4)}]
                <span className="ml-1 opacity-70">row resampling, model held fixed</span>
              </span>
            ) : (
              <span className="text-yellow-400">
                not computed — {iv.uncertainty_status}
              </span>
            )}
          </Row>
        </div>
        <div className="space-y-2">
          <div>
            <p className="text-muted-foreground font-medium mb-0.5">What was simulated</p>
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
        <Info className="h-3 w-3" /> {open ? "Hide" : "Show"} feasibility checks, assumptions &amp; caveats
      </button>
      {open && (
        <div className="px-4 pb-4 border-t border-border/30 pt-3 space-y-2 text-xs text-muted-foreground">
          {(iv.feasibility_checks ?? []).length > 0 && (
            <div className="space-y-1">
              <p className="font-medium text-foreground">Feasibility checks</p>
              {iv.feasibility_checks.map((c, i) => (
                <p key={i} className="flex gap-1.5">
                  <span className={c.passed ? "text-green-400" : "text-red-400"}>
                    {c.passed ? "✓" : "✗"}
                  </span>
                  <span>
                    <span className="font-mono">{c.check}</span> — {c.detail}
                  </span>
                </p>
              ))}
            </div>
          )}
          <div className="space-y-1">
            <p className="font-medium text-foreground">Assumptions</p>
            {(iv.assumptions ?? []).map((a, i) => <p key={i}>· {a}</p>)}
          </div>
          <p className="opacity-80">
            Model: {iv.simulation_model} · {iv.simulation_evaluation}
          </p>
          <p className="italic opacity-70">{iv.caveat ?? ""}</p>
        </div>
      )}
    </div>
  );
}

/* ═══════════════ EXECUTIVE ══════════════════════════════════════════════ */
function ExecutiveTab({ exec, prov }: {
  exec: AnalysisBundle["executive"];
  prov?: AnalysisBundle["provenance"];
}) {
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
        <Card
          title="Highest-ranked candidate levers"
          sub="Ranked by simulated magnitude. Validate with a controlled test before acting."
        >
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
      {prov && (
        <Card title="Provenance" sub="What produced the numbers on this page">
          <dl className="mt-3 space-y-1.5 text-xs">
            <ProvRow k="Question asked" v={
              prov.analysis_mode === "causal"
                ? "causal — adjusted effects and screened what-if simulations"
                : "descriptive & predictive only — no causal claims"} />
            <ProvRow k="Causal graph" v={`${prov.dag_source.replace(/_/g, " ")}${
              prov.ontology_version ? ` · ontology ${prov.ontology_version}` : ""}`} />
            <ProvRow k="Adjustment sets" v={prov.adjustment_set_source.replace(/_/g, " ")} />
            <ProvRow k="Effect estimator" v={prov.effect_estimator} />
            {prov.effect_interval_method && (
              <ProvRow k="Effect intervals" v={prov.effect_interval_method.replace(/_/g, " ")} />
            )}
            {prov.simulation_model && (
              <ProvRow k="Simulation model" v={`${prov.simulation_model.replace(/_/g, " ")} · ${prov.simulation_evaluation}`} />
            )}
            {prov.simulation_interval_method && (
              <ProvRow k="Simulation intervals" v={prov.simulation_interval_method.replace(/_/g, " ")} />
            )}
            <ProvRow k="Rows" v={`${prov.n_rows_analysed.toLocaleString()} analysed of ${prov.n_rows_supplied.toLocaleString()} supplied`} />
            {prov.sampling_note && <ProvRow k="Sampling" v={prov.sampling_note} />}
            <ProvRow k="Train / evaluation" v={prov.train_eval_strategy} />
            <ProvRow k="Random seed" v={String(prov.random_seed)} />
          </dl>
          {prov.graph_assumption && (
            <p className="text-xs text-muted-foreground mt-3 leading-relaxed">{prov.graph_assumption}</p>
          )}
        </Card>
      )}
    </div>
  );
}

function ProvRow({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex gap-3">
      <dt className="text-muted-foreground w-40 shrink-0">{k}</dt>
      <dd className="flex-1">{v}</dd>
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
