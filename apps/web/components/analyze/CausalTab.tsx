"use client";

import type { CausalEffect } from "@/lib/types";

const STRENGTH_STYLES = {
  strong:       "bg-green-500/20 text-green-400 border-green-500/30",
  moderate:     "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  weak:         "bg-orange-500/20 text-orange-400 border-orange-500/30",
  insufficient: "bg-gray-500/20 text-gray-400 border-gray-500/30",
};

export function CausalTab({
  effects,
  target,
}: {
  effects: CausalEffect[];
  target: string;
}) {
  if (effects.length === 0) {
    return (
      <div className="rounded-xl border border-border/60 bg-card p-8 text-center text-muted-foreground text-sm">
        No causal effects computed. Ensure controllable numeric columns exist
        and have at least 30 non-missing rows.
      </div>
    );
  }

  const maxAbs = Math.max(...effects.map((e) => Math.abs(e.effect_per_std)), 0.01);

  return (
    <div className="space-y-4">
      {/* Methodology note */}
      <div className="rounded-xl border border-blue-500/30 bg-blue-500/5 p-4 text-xs text-muted-foreground leading-relaxed">
        <strong className="text-blue-400">Causal estimation method:</strong> For each
        controllable variable we fit back-door adjusted OLS:{" "}
        <code>{target} ~ feature + adjustment_set</code>.
        The adjustment set is derived from the DAG (confounders + DAG parents + context
        variables, excluding mediators and descendants). All features are standardised —
        coefficients represent the effect of a <strong>+1 SD increase</strong> in the
        feature on <strong>{target}</strong>. p-values and 95% CIs come from
        statsmodels OLS inference. This is observational data — unobserved confounders
        may bias estimates.
      </div>

      {/* Effect table */}
      <div className="rounded-xl border border-border/60 bg-card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border/40 text-xs text-muted-foreground">
                <th className="px-4 py-2 text-left">Feature</th>
                <th className="px-4 py-2 text-left">Effect / SD</th>
                <th className="px-4 py-2 text-left w-40">Magnitude</th>
                <th className="px-4 py-2 text-right">p-value</th>
                <th className="px-4 py-2 text-right">95% CI</th>
                <th className="px-4 py-2 text-left">Adjusted for</th>
                <th className="px-4 py-2 text-left">Strength</th>
              </tr>
            </thead>
            <tbody>
              {effects.map((e) => (
                <tr key={e.feature} className="border-b border-border/20 hover:bg-muted/10 transition-colors">
                  <td className="px-4 py-2 font-mono text-xs font-semibold">{e.feature}</td>
                  <td className={`px-4 py-2 font-mono text-xs font-bold ${
                    e.effect_per_std > 0 ? "text-red-400" : "text-green-400"
                  }`}>
                    {e.effect_per_std > 0 ? "+" : ""}{e.effect_per_std.toFixed(4)}
                  </td>
                  <td className="px-4 py-2">
                    <div className="flex items-center gap-1">
                      <div className="w-32 h-3 bg-muted rounded-full overflow-hidden relative">
                        {/* zero line */}
                        <div className="absolute top-0 bottom-0 left-1/2 w-px bg-border" />
                        {/* bar */}
                        <div
                          className={`absolute top-0 bottom-0 ${
                            e.effect_per_std > 0 ? "left-1/2" : "right-1/2"
                          } ${e.p_value < 0.05 ? "opacity-100" : "opacity-40"} ${
                            e.effect_per_std > 0 ? "bg-red-400" : "bg-green-400"
                          }`}
                          style={{
                            width: `${(Math.abs(e.effect_per_std) / maxAbs) * 50}%`,
                          }}
                        />
                      </div>
                    </div>
                  </td>
                  <td className={`px-4 py-2 text-right font-mono text-xs ${
                    e.p_value < 0.05 ? "text-green-400" : "text-muted-foreground"
                  }`}>
                    {e.p_value < 0.001 ? "<0.001" : e.p_value.toFixed(4)}
                  </td>
                  <td className="px-4 py-2 text-right font-mono text-xs text-muted-foreground">
                    [{e.conf_int_lo.toFixed(3)}, {e.conf_int_hi.toFixed(3)}]
                  </td>
                  <td className="px-4 py-2 max-w-xs">
                    {e.adjusted_for.length > 0 ? (
                      <span className="text-xs text-muted-foreground">
                        {e.adjusted_for.slice(0, 3).join(", ")}
                        {e.adjusted_for.length > 3 && ` +${e.adjusted_for.length - 3}`}
                      </span>
                    ) : (
                      <span className="text-xs text-muted-foreground italic">none</span>
                    )}
                  </td>
                  <td className="px-4 py-2">
                    <span className={`text-xs rounded-full border px-2 py-0.5 ${STRENGTH_STYLES[e.evidence_strength]}`}>
                      {e.evidence_strength}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Warnings */}
      {effects.filter((e) => e.warning).map((e) => (
        <div key={e.feature} className="rounded-lg border border-yellow-500/30 bg-yellow-500/5 p-3 text-xs">
          <span className="font-semibold text-yellow-400">{e.feature}:</span>{" "}
          <span className="text-muted-foreground">{e.warning}</span>
        </div>
      ))}
    </div>
  );
}
