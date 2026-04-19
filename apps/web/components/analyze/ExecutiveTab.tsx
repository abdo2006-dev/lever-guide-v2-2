"use client";

import type { ExecutiveSummary, FeatureDistribution } from "@/lib/types";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

export function ExecutiveTab({ executive }: { executive: ExecutiveSummary }) {
  return (
    <div className="max-w-2xl mx-auto space-y-6">
      {/* Headline */}
      <div className="rounded-xl border border-primary/30 bg-primary/5 p-6 text-center">
        <h2 className="text-xl font-bold mb-2">{executive.headline}</h2>
        <p className="text-sm text-muted-foreground">{executive.sub_headline}</p>
      </div>

      {/* Key findings */}
      <div className="rounded-xl border border-border/60 bg-card p-5 space-y-3">
        <p className="font-semibold text-sm">Key Findings</p>
        <ul className="space-y-2">
          {executive.bullets.map((b, i) => (
            <li key={i} className="flex gap-3 text-sm">
              <span className="text-primary mt-0.5 shrink-0">→</span>
              <span>{b}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Top levers */}
      {executive.top_levers.length > 0 && (
        <div className="rounded-xl border border-green-500/30 bg-green-500/5 p-5">
          <p className="font-semibold text-sm mb-3 text-green-400">Top Levers to Pull</p>
          <div className="flex flex-wrap gap-2">
            {executive.top_levers.map((l) => (
              <span key={l} className="rounded-lg border border-green-500/30 bg-green-500/10 px-3 py-1 text-sm font-mono">
                {l}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Cautions */}
      <div className="rounded-xl border border-yellow-500/30 bg-yellow-500/5 p-5 space-y-2">
        <p className="font-semibold text-sm text-yellow-400">Important Caveats</p>
        {executive.cautions.map((c, i) => (
          <p key={i} className="text-sm text-muted-foreground">⚠ {c}</p>
        ))}
      </div>

      {/* Methodology */}
      <div className="rounded-xl border border-border/60 bg-card p-5 text-xs text-muted-foreground space-y-2">
        <p className="font-semibold text-foreground text-sm">Methodology Note</p>
        <p className="leading-relaxed">{executive.methodology_note}</p>
        <p className="leading-relaxed italic">{executive.disclaimer}</p>
      </div>
    </div>
  );
}


export function EdaTab({ distributions }: { distributions: FeatureDistribution[] }) {
  if (!distributions || distributions.length === 0) {
    return (
      <div className="rounded-xl border border-border/60 bg-card p-8 text-center text-muted-foreground text-sm">
        No distribution data available.
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
      {distributions.map((dist) => (
        <DistCard key={dist.feature} dist={dist} />
      ))}
    </div>
  );
}

function DistCard({ dist }: { dist: FeatureDistribution }) {
  if (dist.kind === "categorical") {
    const data = dist.categorical_counts.slice(0, 8).map((v) => ({
      name: v.value.length > 14 ? v.value.slice(0, 14) + "…" : v.value,
      count: v.count,
    }));
    return (
      <div className="rounded-xl border border-border/60 bg-card p-4">
        <p className="font-mono text-xs font-semibold mb-2 truncate">{dist.feature}</p>
        <ResponsiveContainer width="100%" height={140}>
          <BarChart data={data} margin={{ left: 0, right: 0, top: 2, bottom: 16 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis dataKey="name" tick={{ fontSize: 8 }} angle={-30} textAnchor="end" />
            <YAxis tick={{ fontSize: 8 }} />
            <Tooltip />
            <Bar dataKey="count" fill="hsl(var(--primary))" fillOpacity={0.7} radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  }

  const data = dist.distribution.map((b) => ({
    name: b.bin_lo.toFixed(2),
    count: b.count,
  }));

  return (
    <div className="rounded-xl border border-border/60 bg-card p-4">
      <p className="font-mono text-xs font-semibold mb-2 truncate">{dist.feature}</p>
      <ResponsiveContainer width="100%" height={140}>
        <BarChart data={data} margin={{ left: 0, right: 0, top: 2, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis dataKey="name" tick={{ fontSize: 8 }} />
          <YAxis tick={{ fontSize: 8 }} />
          <Tooltip formatter={(v: number) => v} labelFormatter={(l) => `≥${l}`} />
          <Bar dataKey="count" fill="hsl(var(--primary))" fillOpacity={0.7} radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
