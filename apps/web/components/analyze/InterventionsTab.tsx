"use client";

import type { Intervention } from "@/lib/types";

const STRENGTH_STYLES = {
  strong:       "bg-green-500/20 text-green-400 border-green-500/30",
  moderate:     "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  weak:         "bg-orange-500/20 text-orange-400 border-orange-500/30",
  insufficient: "bg-gray-500/20 text-gray-400 border-gray-500/30",
};

const TYPE_STYLES = {
  causal:     "bg-blue-500/20 text-blue-400 border-blue-500/30",
  predictive: "bg-purple-500/20 text-purple-400 border-purple-500/30",
  mixed:      "bg-teal-500/20 text-teal-400 border-teal-500/30",
};

export function InterventionsTab({
  interventions,
  target,
}: {
  interventions: Intervention[];
  target: string;
}) {
  if (interventions.length === 0) {
    return (
      <div className="rounded-xl border border-border/60 bg-card p-8 text-center text-muted-foreground text-sm">
        No intervention recommendations generated. Ensure at least one controllable
        numeric feature has meaningful variance.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-border/60 bg-card p-4">
        <p className="text-sm font-semibold mb-1">Intervention Recommendations</p>
        <p className="text-xs text-muted-foreground">
          Ranked by estimated impact on <strong>{target}</strong>. Each recommendation
          combines a GBR counterfactual simulation (predictive) with back-door adjusted
          OLS coefficients (causal) where available. Validate with controlled experiments
          before operational changes.
        </p>
      </div>

      {interventions.map((iv) => (
        <InterventionCard key={iv.feature} iv={iv} target={target} />
      ))}
    </div>
  );
}

function InterventionCard({ iv, target }: { iv: Intervention; target: string }) {
  const impactPositive = iv.expected_kpi_change < 0; // decrease = improvement for scrap-like KPIs
  const impactLabel = `${iv.expected_kpi_change > 0 ? "+" : ""}${iv.expected_kpi_change_pct.toFixed(1)}%`;

  return (
    <div className="rounded-xl border border-border/60 bg-card overflow-hidden">
      {/* Header bar */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/40 bg-muted/20">
        <div className="flex items-center gap-3">
          <span className="h-6 w-6 rounded-full bg-primary/20 text-primary text-xs font-bold flex items-center justify-center">
            {iv.rank}
          </span>
          <span className="font-semibold text-sm font-mono">{iv.feature}</span>
          <span className={`text-xs rounded-full border px-2 py-0.5 font-medium ${TYPE_STYLES[iv.evidence_type]}`}>
            {iv.evidence_type}
          </span>
          <span className={`text-xs rounded-full border px-2 py-0.5 font-medium ${STRENGTH_STYLES[iv.evidence_strength]}`}>
            {iv.evidence_strength}
          </span>
        </div>
        <div className={`text-lg font-bold font-mono ${impactPositive ? "text-green-400" : "text-red-400"}`}>
          {impactLabel} {target}
        </div>
      </div>

      {/* Body */}
      <div className="p-4 grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Left: numbers */}
        <div className="space-y-2">
          <Row label="Direction">
            <DirectionBadge dir={iv.direction} />
          </Row>
          <Row label="Current mean">
            <code>{iv.current_mean.toFixed(3)}</code>
            <span className="text-muted-foreground ml-1 text-xs">
              [p10={iv.current_p10.toFixed(2)}, p90={iv.current_p90.toFixed(2)}]
            </span>
          </Row>
          <Row label="Suggested value">
            <code className="text-primary font-bold">{iv.suggested_value.toFixed(3)}</code>
            <span className="text-muted-foreground ml-1 text-xs">
              (Δ {iv.delta > 0 ? "+" : ""}{iv.delta.toFixed(3)}, {iv.delta_pct.toFixed(1)}%)
            </span>
          </Row>
          <Row label="Estimated KPI change">
            <code className={impactPositive ? "text-green-400" : "text-red-400"}>
              {iv.expected_kpi_change > 0 ? "+" : ""}{iv.expected_kpi_change.toFixed(4)}
            </code>
          </Row>
        </div>

        {/* Right: qualitative */}
        <div className="space-y-2 text-xs">
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

      {/* Assumptions + caveat footer */}
      <div className="px-4 pb-3 pt-1 border-t border-border/30 bg-muted/10">
        <p className="text-xs text-muted-foreground mb-1 font-medium">Assumptions</p>
        <ul className="space-y-0.5">
          {iv.assumptions.map((a, i) => (
            <li key={i} className="text-xs text-muted-foreground">· {a}</li>
          ))}
        </ul>
        <p className="text-xs text-muted-foreground/60 mt-2 italic">{iv.caveat}</p>
      </div>
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-baseline gap-2 text-xs">
      <span className="text-muted-foreground w-32 shrink-0">{label}</span>
      <span className="font-mono">{children}</span>
    </div>
  );
}

function DirectionBadge({ dir }: { dir: "increase" | "decrease" }) {
  return (
    <span className={`rounded px-1.5 py-0.5 text-xs font-medium ${
      dir === "increase"
        ? "bg-green-500/20 text-green-400"
        : "bg-blue-500/20 text-blue-400"
    }`}>
      {dir === "increase" ? "↑ increase" : "↓ decrease"}
    </span>
  );
}
