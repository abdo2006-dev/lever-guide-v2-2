"use client";

import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, BarChart, Bar, Cell
} from "recharts";
import type { PredictiveResult } from "@/lib/types";
import { useState } from "react";

export function PredictiveTab({ results }: { results: PredictiveResult[] }) {
  const [selected, setSelected] = useState<string>(
    results.find((r) => r.is_winner)?.model ?? results[0]?.model ?? ""
  );

  const model = results.find((r) => r.model === selected) ?? results[0];
  if (!model) return null;

  return (
    <div className="space-y-6">
      {/* Model selector */}
      <div className="flex flex-wrap gap-2">
        {results.map((r) => (
          <button
            key={r.model}
            onClick={() => setSelected(r.model)}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium border transition-colors ${
              selected === r.model
                ? "bg-primary text-primary-foreground border-primary"
                : "border-border hover:bg-accent"
            }`}
          >
            {r.is_winner && "★ "}{r.display_name}
            <span className="ml-2 font-mono text-xs opacity-70">R²={r.metrics.r2.toFixed(3)}</span>
          </button>
        ))}
      </div>

      {/* Metrics row */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        {[
          { label: "R²", val: model.metrics.r2.toFixed(4) },
          { label: "Adj. R²", val: model.metrics.adj_r2?.toFixed(4) ?? "—" },
          { label: "RMSE", val: model.metrics.rmse.toFixed(4) },
          { label: "MAE", val: model.metrics.mae.toFixed(4) },
          { label: "CV R² (3-fold)", val: model.metrics.cv_r2_mean !== undefined
            ? `${model.metrics.cv_r2_mean.toFixed(3)} ±${model.metrics.cv_r2_std?.toFixed(3)}`
            : "—" },
        ].map(({ label, val }) => (
          <div key={label} className="rounded-xl border border-border/60 bg-card p-3 text-center">
            <p className="text-xs text-muted-foreground mb-1">{label}</p>
            <p className="font-mono font-bold text-sm">{val}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Feature importances */}
        <div className="rounded-xl border border-border/60 bg-card p-4">
          <p className="font-semibold text-sm mb-3">Feature Importances</p>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart
              layout="vertical"
              data={model.importances.slice(0, 12).map((f) => ({
                name: f.feature.length > 22 ? f.feature.slice(0, 22) + "…" : f.feature,
                value: f.importance_norm,
              }))}
              margin={{ left: 8, right: 16, top: 4, bottom: 4 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 10 }} />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 10 }} width={140} />
              <Tooltip formatter={(v: number) => v.toFixed(4)} />
              <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                {model.importances.slice(0, 12).map((_, i) => (
                  <Cell key={i} fill={i === 0 ? "hsl(var(--primary))" : "hsl(var(--primary) / 0.5)"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Actual vs predicted scatter */}
        <div className="rounded-xl border border-border/60 bg-card p-4">
          <p className="font-semibold text-sm mb-1">Actual vs Predicted (test set)</p>
          <p className="text-xs text-muted-foreground mb-3">
            Points clustered along the diagonal indicate good model fit.
          </p>
          <ResponsiveContainer width="100%" height={260}>
            <ScatterChart margin={{ left: 8, right: 8, top: 4, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis dataKey="actual" name="Actual" type="number" tick={{ fontSize: 10 }} label={{ value: "Actual", position: "insideBottom", offset: -2, fontSize: 10 }} />
              <YAxis dataKey="predicted" name="Predicted" type="number" tick={{ fontSize: 10 }} label={{ value: "Predicted", angle: -90, position: "insideLeft", fontSize: 10 }} />
              <Tooltip cursor={{ strokeDasharray: "3 3" }} formatter={(v: number) => v.toFixed(4)} />
              <Scatter
                data={model.predictions.slice(0, 400)}
                fill="hsl(var(--primary))"
                fillOpacity={0.5}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Coefficients table (OLS / Ridge only) */}
      {model.coefficients && model.coefficients.length > 0 && (
        <div className="rounded-xl border border-border/60 bg-card overflow-hidden">
          <div className="px-4 py-3 border-b border-border/60">
            <p className="font-semibold text-sm">Regression Coefficients</p>
            <p className="text-xs text-muted-foreground">Standardised features — coefficients are comparable in magnitude.</p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border/40 text-muted-foreground">
                  <th className="px-4 py-2 text-left">Feature</th>
                  <th className="px-4 py-2 text-right">Coef</th>
                  <th className="px-4 py-2 text-right">Std Err</th>
                  <th className="px-4 py-2 text-right">t</th>
                  <th className="px-4 py-2 text-right">p-value</th>
                  <th className="px-4 py-2 text-right">Sig</th>
                </tr>
              </thead>
              <tbody>
                {model.coefficients.filter((c) => c.feature !== "(intercept)").map((c) => (
                  <tr key={c.feature} className={`border-b border-border/20 ${c.significant ? "" : "opacity-60"}`}>
                    <td className="px-4 py-1.5 font-mono">{c.feature}</td>
                    <td className={`px-4 py-1.5 text-right font-mono ${c.coef > 0 ? "text-red-400" : "text-green-400"}`}>
                      {c.coef.toFixed(4)}
                    </td>
                    <td className="px-4 py-1.5 text-right font-mono text-muted-foreground">{c.std_err.toFixed(4)}</td>
                    <td className="px-4 py-1.5 text-right font-mono">{c.t_stat.toFixed(3)}</td>
                    <td className={`px-4 py-1.5 text-right font-mono ${c.p_value < 0.05 ? "text-green-400" : "text-muted-foreground"}`}>
                      {c.p_value < 0.001 ? "<0.001" : c.p_value.toFixed(4)}
                    </td>
                    <td className="px-4 py-1.5 text-right">{c.significant ? "★" : ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
