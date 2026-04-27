"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ArrowRight, BarChart3, Brain, GitBranch, Lightbulb, ShieldCheck, Upload, Zap } from "lucide-react";
import { useAppStore } from "@/lib/store";

const FEATURES = [
  {
    icon: BarChart3,
    title: "Five-model comparison",
    body: "OLS, Ridge, Random Forest, XGBoost, and LightGBM trained and evaluated on a held-out test set.",
  },
  {
    icon: GitBranch,
    title: "DAG-aware causal analysis",
    body: "Assign column roles and optionally submit DAG edges through the API. The engine validates the graph before causal analysis.",
  },
  {
    icon: Lightbulb,
    title: "Ranked recommendations",
    body: "Direction, magnitude, evidence type, strength, tradeoffs, and assumptions — for every lever.",
  },
  {
    icon: Brain,
    title: "Executive summary",
    body: "Plain-language summary for non-technical stakeholders with honest caveats included.",
  },
  {
    icon: ShieldCheck,
    title: "Honest uncertainty",
    body: "Confidence intervals, p-values, and model quality metrics always visible.",
  },
  {
    icon: Upload,
    title: "Bring your own data",
    body: "Upload CSV files up to 50 MB. Large datasets are sampled to 2,000 rows for analysis.",
  },
];

export function HomeClient() {
  const router = useRouter();
  const { setDataset, setTarget } = useAppStore();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleDemo = async () => {
    setLoading(true);
    setError("");
    try {
      const { loadDemoDataset, DEMO_TARGET } = await import("@/lib/csv");
      const ds = await loadDemoDataset();
      setDataset(ds);
      setTarget(DEMO_TARGET);
      router.push("/setup");
    } catch (e) {
      setError("Failed to load demo. Please try again.");
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-background text-foreground">
      {/* Nav */}
      <nav className="border-b border-border/40 backdrop-blur-sm sticky top-0 z-50 bg-background/80">
        <div className="container flex h-14 items-center justify-between">
          <span className="font-bold text-lg tracking-tight">
            LeverGuide
          </span>
          <Link href="/setup" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
            Launch App
          </Link>
        </div>
      </nav>

      {/* Hero */}
      <section className="container py-24 md:py-36 text-center max-w-4xl mx-auto">
        <div className="inline-flex items-center gap-2 rounded-full border border-primary/30 bg-primary/5 px-3 py-1 text-xs text-primary mb-6">
          <Zap className="h-3 w-3" /> v2 — production-grade ML + causal analysis
        </div>
        <h1 className="text-5xl md:text-7xl font-bold tracking-tight mb-6 leading-tight">
          Don&apos;t just predict.{" "}
          <span className="bg-gradient-to-r from-blue-500 to-violet-500 bg-clip-text text-transparent">
            Decide what to change.
          </span>
        </h1>
        <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-10 leading-relaxed">
          Upload any tabular dataset, pick a KPI, and get ranked, explainable
          recommendations — with predictive and causal evidence shown side by side.
        </p>

        {error && (
          <p className="text-sm text-red-400 mb-4">{error}</p>
        )}

        <div className="flex flex-col sm:flex-row gap-3 justify-center">
          <button
            onClick={handleDemo}
            disabled={loading}
            className="inline-flex items-center justify-center gap-2 h-12 px-8 rounded-lg bg-primary text-primary-foreground font-semibold hover:opacity-90 transition-opacity disabled:opacity-50"
          >
            {loading ? "Loading…" : "Try Demo Dataset"}
            <ArrowRight className="h-4 w-4" />
          </button>
          <Link
            href="/setup"
            className="inline-flex items-center justify-center gap-2 h-12 px-8 rounded-lg border border-border hover:bg-accent transition-colors font-semibold"
          >
            <Upload className="h-4 w-4" /> Upload your data
          </Link>
        </div>
      </section>

      {/* Feature grid */}
      <section className="container pb-24">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 max-w-5xl mx-auto">
          {FEATURES.map(({ icon: Icon, title, body }) => (
            <div
              key={title}
              className="rounded-xl border border-border/60 bg-card p-5 hover:border-primary/40 transition-colors"
            >
              <div className="flex items-center gap-3 mb-3">
                <div className="h-8 w-8 rounded-lg bg-primary/10 flex items-center justify-center">
                  <Icon className="h-4 w-4 text-primary" />
                </div>
                <h3 className="font-semibold text-sm">{title}</h3>
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed">{body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Stack */}
      <section className="border-t border-border/40 py-16">
        <div className="container max-w-3xl mx-auto text-center">
          <h2 className="text-2xl font-bold mb-3">Serious ML under the hood</h2>
          <p className="text-muted-foreground text-sm leading-relaxed max-w-xl mx-auto">
            Python FastAPI backend running scikit-learn, XGBoost, and LightGBM.
            Causal estimates use back-door adjusted OLS with statsmodels.
          </p>
          <div className="mt-8 flex flex-wrap justify-center gap-2 text-xs text-muted-foreground">
            {["Next.js 15","TypeScript","FastAPI","scikit-learn","XGBoost","LightGBM","statsmodels","Vercel","Render"].map((t) => (
              <span key={t} className="rounded-full border border-border/60 px-3 py-1">{t}</span>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
