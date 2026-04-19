"use client";

import { useRef, useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import {
  Upload, ArrowRight, CheckCircle2, AlertCircle, Circle,
  Loader2, ChevronRight, Database, Target, Tag, Zap
} from "lucide-react";
import { useAppStore } from "@/lib/store";
import { parseCsvFile, loadDemoDataset, DEMO_TARGET } from "@/lib/csv";
import { runAnalysis, ApiError } from "@/lib/api-client";
import type { ColumnRole } from "@/lib/types";

const ROLES: ColumnRole[] = [
  "outcome","controllable","confounder","mediator","context","identifier","ignore",
];
const ROLE_COLORS: Record<ColumnRole, string> = {
  outcome:      "bg-blue-500/20 text-blue-400 border-blue-500/40",
  controllable: "bg-green-500/20 text-green-400 border-green-500/40",
  confounder:   "bg-orange-500/20 text-orange-400 border-orange-500/40",
  mediator:     "bg-purple-500/20 text-purple-400 border-purple-500/40",
  context:      "bg-yellow-500/20 text-yellow-400 border-yellow-500/40",
  identifier:   "bg-gray-500/20 text-gray-400 border-gray-500/40",
  ignore:       "bg-gray-500/10 text-gray-500 border-gray-600/20",
};
const ROLE_HELP: Record<ColumnRole, string> = {
  outcome:      "The KPI you want to improve",
  controllable: "Levers you can actually change",
  confounder:   "Causes both controls and outcome",
  mediator:     "On the causal path — don't adjust",
  context:      "Fixed factors per run",
  identifier:   "ID / timestamp — excluded",
  ignore:       "Excluded from analysis",
};

const STEPS = [
  { id: 1, icon: Database, label: "Load Data" },
  { id: 2, icon: Target,   label: "Select Target" },
  { id: 3, icon: Tag,      label: "Assign Roles" },
  { id: 4, icon: Zap,      label: "Analyze" },
];

const ANALYSIS_MESSAGES = [
  "Parsing dataset…",
  "Building feature matrix…",
  "Training OLS regression…",
  "Training Ridge regression…",
  "Training Random Forest…",
  "Training XGBoost…",
  "Training LightGBM…",
  "Running causal analysis…",
  "Computing interventions…",
  "Building executive summary…",
  "Finalising results…",
];

export default function SetupPage() {
  const router = useRouter();
  const fileRef = useRef<HTMLInputElement>(null);
  const store = useAppStore();
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisMsg, setAnalysisMsg] = useState("");
  const [msgIdx, setMsgIdx] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [analyzeError, setAnalyzeError] = useState("");

  // Cycle through status messages while analyzing
  useEffect(() => {
    if (!analyzing) { setMsgIdx(0); setElapsed(0); return; }
    setAnalysisMsg(ANALYSIS_MESSAGES[0]);
    const msgTimer = setInterval(() => {
      setMsgIdx(i => {
        const next = Math.min(i + 1, ANALYSIS_MESSAGES.length - 1);
        setAnalysisMsg(ANALYSIS_MESSAGES[next]);
        return next;
      });
    }, 4000);
    const elapsedTimer = setInterval(() => setElapsed(e => e + 1), 1000);
    return () => { clearInterval(msgTimer); clearInterval(elapsedTimer); };
  }, [analyzing]);

  const handleFile = async (file: File) => {
    if (!file.name.endsWith(".csv")) { toast.error("Please upload a CSV file."); return; }
    if (file.size > 50 * 1024 * 1024) { toast.error("File too large — max 50 MB."); return; }
    try {
      const ds = await parseCsvFile(file);
      store.setDataset(ds);
      toast.success(`Loaded ${ds.row_count.toLocaleString()} rows × ${ds.columns.length} columns`);
    } catch (e) { toast.error(String(e)); }
  };

  const handleDemo = async () => {
    try {
      const ds = await loadDemoDataset();
      store.setDataset(ds);
      store.setTarget(DEMO_TARGET);
      toast.success("Demo loaded — 5,000 injection-moulding rows");
    } catch { toast.error("Failed to load demo"); }
  };

  const currentStep = (): number => {
    if (!store.dataset) return 1;
    if (!store.target) return 2;
    const hasControllable = store.dataset.columns.some(c => c.role === "controllable");
    if (!hasControllable) return 3;
    return 4;
  };

  const step = currentStep();
  const controllable = store.dataset?.columns.filter(c => c.role === "controllable") ?? [];
  const confounders  = store.dataset?.columns.filter(c => c.role === "confounder")  ?? [];

  const handleAnalyze = async () => {
    if (!store.dataset || !store.target) return;
    setAnalyzing(true);
    setAnalyzeError("");

    const column_roles: Record<string, ColumnRole> = {};
    for (const col of store.dataset.columns) column_roles[col.name] = col.role;

    // 90-second client-side timeout
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 90_000);

    try {
      const bundle = await runAnalysis({
        dataset_csv: store.dataset.csv_content,
        dataset_name: store.dataset.name,
        target: store.target,
        task: "regression",
        improve_direction: store.improveDirection,
        column_roles,
        dag_edges: store.dagEdges,
        random_seed: 42,
      });
      clearTimeout(timeout);
      store.setAnalysis(bundle);
      router.push("/analyze");
    } catch (err) {
      clearTimeout(timeout);
      let msg = "Analysis failed.";
      if (err instanceof Error && err.name === "AbortError") {
        msg = "Request timed out after 90s. The server may be under load — try again.";
      } else if (err instanceof ApiError) {
        msg = `Server error: ${err.message}`;
      } else if (err instanceof Error) {
        msg = err.message;
      }
      setAnalyzeError(msg);
      toast.error(msg);
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Top bar */}
      <div className="border-b border-border/40 bg-background/80 backdrop-blur sticky top-0 z-50">
        <div className="container h-14 flex items-center justify-between">
          <span className="font-bold text-sm">LeverGuide</span>
          <button onClick={() => { store.reset(); }} className="text-xs text-muted-foreground hover:text-foreground transition-colors">
            ← Start over
          </button>
        </div>
      </div>

      <div className="container py-8 max-w-4xl space-y-8">

        {/* ── Step indicator ─────────────────────────────────────────────── */}
        <div className="flex items-center gap-0">
          {STEPS.map((s, i) => {
            const done    = step > s.id;
            const active  = step === s.id;
            const Icon    = s.icon;
            return (
              <div key={s.id} className="flex items-center flex-1 last:flex-none">
                <div className={`flex flex-col items-center gap-1 ${active ? "text-primary" : done ? "text-green-400" : "text-muted-foreground"}`}>
                  <div className={`h-9 w-9 rounded-full flex items-center justify-center border-2 transition-all ${
                    active ? "border-primary bg-primary/10" :
                    done   ? "border-green-400 bg-green-400/10" :
                             "border-border bg-muted/30"
                  }`}>
                    {done
                      ? <CheckCircle2 className="h-4 w-4" />
                      : <Icon className="h-4 w-4" />
                    }
                  </div>
                  <span className="text-xs font-medium hidden sm:block">{s.label}</span>
                </div>
                {i < STEPS.length - 1 && (
                  <div className={`h-0.5 flex-1 mx-2 rounded ${step > s.id ? "bg-green-400/60" : "bg-border"}`} />
                )}
              </div>
            );
          })}
        </div>

        {/* ── STEP 1: Load data ──────────────────────────────────────────── */}
        {!store.dataset ? (
          <div
            className="rounded-xl border-2 border-dashed border-border hover:border-primary/50 transition-colors p-14 text-center cursor-pointer"
            onClick={() => fileRef.current?.click()}
            onDragOver={e => e.preventDefault()}
            onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) handleFile(f); }}
          >
            <Upload className="h-10 w-10 text-muted-foreground mx-auto mb-4" />
            <p className="font-semibold mb-1">Drop a CSV file or click to browse</p>
            <p className="text-sm text-muted-foreground mb-5">Max 50 MB · numeric and categorical columns supported</p>
            <button
              onClick={e => { e.stopPropagation(); handleDemo(); }}
              className="inline-flex items-center gap-1.5 text-sm text-primary hover:underline"
            >
              <Zap className="h-3.5 w-3.5" /> Use the injection-moulding demo dataset
            </button>
            <input ref={fileRef} type="file" accept=".csv" className="hidden"
              onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])} />
          </div>
        ) : (
          <>
            {/* Dataset badge */}
            <div className="rounded-xl border border-border/60 bg-card px-4 py-3 flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <CheckCircle2 className="h-4 w-4 text-green-400 shrink-0" />
                <div>
                  <p className="font-semibold text-sm">{store.dataset.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {store.dataset.row_count.toLocaleString()} rows · {store.dataset.columns.length} columns
                    {store.dataset.row_count > 2000 && (
                      <span className="ml-2 text-yellow-400">· sampled to 2,000 for analysis</span>
                    )}
                  </p>
                </div>
              </div>
              <div className="flex gap-2 text-xs flex-wrap">
                <Chip color="green">{controllable.length} controllable</Chip>
                <Chip color="orange">{confounders.length} confounder</Chip>
                {store.target && <Chip color="blue">target: {store.target}</Chip>}
              </div>
            </div>

            {/* ── STEP 2: Target ─────────────────────────────────────────── */}
            <Section
              step={2}
              active={step === 2}
              done={!!store.target}
              title="Select your target KPI"
              subtitle="The numeric column you want to improve."
            >
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4">
                <div>
                  <label className="block text-xs font-semibold mb-2 text-muted-foreground uppercase tracking-wide">
                    Target column
                  </label>
                  <select
                    value={store.target ?? ""}
                    onChange={e => store.setTarget(e.target.value || null)}
                    className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/40"
                  >
                    <option value="">Select a numeric column…</option>
                    {store.dataset.columns.filter(c => c.kind === "numeric").map(c => (
                      <option key={c.name} value={c.name}>{c.name}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-semibold mb-2 text-muted-foreground uppercase tracking-wide">
                    Goal direction
                  </label>
                  <div className="flex gap-2">
                    {(["decrease","increase"] as const).map(d => (
                      <button key={d} onClick={() => store.setImproveDirection(d)}
                        className={`flex-1 py-2 rounded-lg text-sm font-medium border transition-colors ${
                          store.improveDirection === d
                            ? "bg-primary text-primary-foreground border-primary"
                            : "border-border hover:bg-accent"
                        }`}>
                        {d === "decrease" ? "↓ Minimise" : "↑ Maximise"}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </Section>

            {/* ── STEP 3: Roles ──────────────────────────────────────────── */}
            <Section
              step={3}
              active={step === 3}
              done={step >= 4}
              title="Assign column roles"
              subtitle={`Set at least one column to controllable — these are the levers the engine will recommend. Currently ${controllable.length} controllable.`}
            >
              <div className="mt-4 rounded-lg border border-border/60 overflow-hidden">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border/40 text-xs text-muted-foreground bg-muted/20">
                        <th className="px-3 py-2 text-left">Column</th>
                        <th className="px-3 py-2 text-left">Type</th>
                        <th className="px-3 py-2 text-left hidden md:table-cell">Sample</th>
                        <th className="px-3 py-2 text-left">Role</th>
                      </tr>
                    </thead>
                    <tbody>
                      {store.dataset.columns.map(col => (
                        <tr key={col.name} className="border-b border-border/20 hover:bg-muted/10 transition-colors">
                          <td className="px-3 py-2 font-mono text-xs max-w-[140px] truncate">{col.name}</td>
                          <td className="px-3 py-2 text-xs text-muted-foreground">{col.kind}</td>
                          <td className="px-3 py-2 text-xs text-muted-foreground hidden md:table-cell">
                            {col.kind === "numeric" && col.mean !== undefined
                              ? `μ=${col.mean.toFixed(1)}`
                              : col.top_values?.[0]?.value ?? "—"}
                          </td>
                          <td className="px-3 py-2">
                            <select
                              value={col.role}
                              onChange={e => {
                                const r = e.target.value as ColumnRole;
                                if (r === "outcome") store.setTarget(col.name);
                                else store.setColumnRole(col.name, r);
                              }}
                              className={`rounded px-2 py-1 text-xs border font-medium focus:outline-none bg-transparent cursor-pointer ${ROLE_COLORS[col.role]}`}
                            >
                              {ROLES.map(r => (
                                <option key={r} value={r} className="bg-background text-foreground">{r}</option>
                              ))}
                            </select>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Role legend */}
              <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-2">
                {(Object.entries(ROLE_HELP) as [ColumnRole, string][]).map(([role, help]) => (
                  <div key={role} className="text-xs">
                    <span className={`inline-block rounded px-1.5 py-0.5 border font-medium mb-0.5 ${ROLE_COLORS[role]}`}>{role}</span>
                    <p className="text-muted-foreground leading-snug">{help}</p>
                  </div>
                ))}
              </div>
            </Section>

            {/* ── STEP 4: Analyze ────────────────────────────────────────── */}
            <Section
              step={4}
              active={step === 4}
              done={false}
              title="Run analysis"
              subtitle="Trains 5 models, runs causal adjustment, and generates intervention recommendations."
            >
              {/* Checklist */}
              <div className="mt-4 space-y-1.5">
                <Check ok={!!store.target}        label={store.target ? `Target: ${store.target}` : "No target selected"} />
                <Check ok={controllable.length>0} label={`${controllable.length} controllable variable${controllable.length!==1?"s":""}`} />
                <Check ok={confounders.length>0}  label={`${confounders.length} confounder${confounders.length!==1?"s":""} — improves causal estimates`} warn />
              </div>

              {/* Error */}
              {analyzeError && (
                <div className="mt-4 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-400">
                  <AlertCircle className="inline h-4 w-4 mr-1.5 mb-0.5" />
                  {analyzeError}
                </div>
              )}

              {/* Analyzing state */}
              {analyzing && (
                <div className="mt-4 rounded-xl border border-primary/30 bg-primary/5 p-5 space-y-3">
                  <div className="flex items-center gap-3">
                    <Loader2 className="h-5 w-5 text-primary animate-spin shrink-0" />
                    <div>
                      <p className="text-sm font-semibold text-primary">{analysisMsg}</p>
                      <p className="text-xs text-muted-foreground">
                        {elapsed}s elapsed · this takes 20–60s on the first run
                      </p>
                    </div>
                  </div>
                  {/* Progress bar */}
                  <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                    <div
                      className="h-full bg-primary rounded-full transition-all duration-1000"
                      style={{ width: `${Math.min(95, (msgIdx / (ANALYSIS_MESSAGES.length - 1)) * 100)}%` }}
                    />
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Running: OLS · Ridge · Random Forest · XGBoost · LightGBM + causal analysis
                  </p>
                </div>
              )}

              {!analyzing && (
                <button
                  onClick={handleAnalyze}
                  disabled={!store.target || controllable.length === 0}
                  className="mt-5 inline-flex items-center gap-2 h-11 px-8 rounded-lg bg-primary text-primary-foreground font-semibold hover:opacity-90 transition-opacity disabled:opacity-40"
                >
                  Run Analysis <ArrowRight className="h-4 w-4" />
                </button>
              )}
            </Section>
          </>
        )}
      </div>
    </div>
  );
}

function Section({ step, active, done, title, subtitle, children }: {
  step: number; active: boolean; done: boolean; title: string; subtitle: string; children?: React.ReactNode;
}) {
  return (
    <div className={`rounded-xl border p-5 transition-all ${
      active ? "border-primary/50 bg-card shadow-sm" :
      done   ? "border-green-500/20 bg-card/50" :
               "border-border/40 bg-card/30 opacity-60"
    }`}>
      <div className="flex items-start gap-3">
        <div className={`h-6 w-6 rounded-full flex items-center justify-center text-xs font-bold shrink-0 mt-0.5 ${
          active ? "bg-primary text-primary-foreground" :
          done   ? "bg-green-500/20 text-green-400" :
                   "bg-muted text-muted-foreground"
        }`}>
          {done ? <CheckCircle2 className="h-3.5 w-3.5" /> : step}
        </div>
        <div className="flex-1 min-w-0">
          <p className={`font-semibold text-sm ${active ? "" : done ? "text-green-400/80" : "text-muted-foreground"}`}>
            {title}
          </p>
          <p className="text-xs text-muted-foreground mt-0.5 leading-relaxed">{subtitle}</p>
          {children}
        </div>
      </div>
    </div>
  );
}

function Check({ ok, label, warn }: { ok: boolean; label: string; warn?: boolean }) {
  return (
    <div className="flex items-center gap-2 text-xs">
      {ok
        ? <CheckCircle2 className="h-3.5 w-3.5 text-green-400 shrink-0" />
        : warn
          ? <Circle className="h-3.5 w-3.5 text-muted-foreground/40 shrink-0" />
          : <AlertCircle className="h-3.5 w-3.5 text-yellow-400 shrink-0" />
      }
      <span className={ok ? "text-foreground" : warn ? "text-muted-foreground" : "text-yellow-400"}>{label}</span>
    </div>
  );
}

function Chip({ children, color }: { children: React.ReactNode; color: string }) {
  const map: Record<string,string> = {
    green:  "bg-green-500/10 text-green-400 border-green-500/30",
    orange: "bg-orange-500/10 text-orange-400 border-orange-500/30",
    blue:   "bg-blue-500/10 text-blue-400 border-blue-500/30",
  };
  return <span className={`rounded-full border px-2 py-0.5 text-xs ${map[color]??""}`}>{children}</span>;
}
