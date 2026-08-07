"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ArrowRight, BarChart3, Brain, GitBranch, Lightbulb, ShieldCheck, Upload, Zap } from "lucide-react";
import { useAppStore } from "@/lib/store";

const FEATURES = [
  {
    icon: BarChart3,
    title: "Model comparison, with run status",
    body: "OLS, Ridge and Random Forest, plus XGBoost and LightGBM where their libraries are available. Every configured model reports whether it actually ran.",
  },
  {
    icon: GitBranch,
    title: "Three kinds of result, kept apart",
    body: "Associations, adjusted observational effect estimates, and predictive what-if simulations are labelled separately — because they support different claims.",
  },
  {
    icon: Lightbulb,
    title: "Screened candidate changes",
    body: "A simulated change is only ranked if it is physically feasible, inside observed support, and agrees with its own adjusted estimate. The rest are shown with the reason they were set aside.",
  },
  {
    icon: Brain,
    title: "Executive summary",
    body: "Plain-language summary for non-technical stakeholders, with the caveats attached to the numbers rather than to a footnote.",
  },
  {
    icon: ShieldCheck,
    title: "Uncertainty where we have it",
    body: "Confidence intervals and p-values on every adjusted effect estimate. Simulated changes carry a row-resampling interval that holds the model fixed, and say so.",
  },
  {
    icon: Upload,
    title: "Bring your own data",
    body: "CSV up to about 5 MB (browser storage limit). Datasets over 2,000 rows are sub-sampled. You specify the outcome and the roles — nothing is inferred for you.",
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
          <Zap className="h-3 w-3" /> v2 - ML decision support with transparent caveats
        </div>
        <h1 className="text-5xl md:text-7xl font-bold tracking-tight mb-6 leading-tight">
          Don&apos;t just predict.{" "}
          <span className="bg-gradient-to-r from-blue-500 to-violet-500 bg-clip-text text-transparent">
            Decide what to change.
          </span>
        </h1>
        <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-10 leading-relaxed">
          Upload a tabular dataset, pick a KPI, and get ranked, explainable
          candidate changes — with predictive signal and DAG-adjusted effect
          estimates shown side by side, and the difference between them made
          explicit.
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
            Python FastAPI backend running scikit-learn, with XGBoost and LightGBM
            as optional extras. Effect estimates use adjusted OLS with statsmodels
            under a declared causal graph — an assumption stated up front, not a
            structure discovered from data.
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
