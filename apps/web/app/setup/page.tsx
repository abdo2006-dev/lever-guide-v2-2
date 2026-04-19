"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { Upload, ArrowRight, ChevronDown, CheckCircle2, AlertCircle } from "lucide-react";
import { useAppStore } from "@/lib/store";
import { parseCsvFile, loadDemoDataset, DEMO_TARGET } from "@/lib/csv";
import { runAnalysis, ApiError } from "@/lib/api-client";
import type { ColumnRole, DagEdge } from "@/lib/types";

const ROLES: ColumnRole[] = [
  "outcome", "controllable", "confounder", "mediator", "context", "identifier", "ignore",
];

const ROLE_COLORS: Record<ColumnRole, string> = {
  outcome:      "bg-blue-500/20 text-blue-400 border-blue-500/30",
  controllable: "bg-green-500/20 text-green-400 border-green-500/30",
  confounder:   "bg-orange-500/20 text-orange-400 border-orange-500/30",
  mediator:     "bg-purple-500/20 text-purple-400 border-purple-500/30",
  context:      "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  identifier:   "bg-gray-500/20 text-gray-400 border-gray-500/30",
  ignore:       "bg-gray-500/10 text-gray-500 border-gray-600/30",
};

const ROLE_HELP: Record<ColumnRole, string> = {
  outcome:      "The KPI you want to improve",
  controllable: "Variables you can actively change",
  confounder:   "Causes both controls and outcome",
  mediator:     "On the causal path — don't adjust",
  context:      "Fixed design factors for the run",
  identifier:   "Row ID / timestamp — not used",
  ignore:       "Exclude from analysis",
};

export default function SetupPage() {
  const router = useRouter();
  const fileRef = useRef<HTMLInputElement>(null);
  const store = useAppStore();
  const [analyzing, setAnalyzing] = useState(false);

  const handleFile = async (file: File) => {
    if (!file.name.endsWith(".csv")) {
      toast.error("Please upload a CSV file.");
      return;
    }
    if (file.size > 50 * 1024 * 1024) {
      toast.error("File too large — max 50 MB.");
      return;
    }
    try {
      const ds = await parseCsvFile(file);
      store.setDataset(ds);
      toast.success(`Loaded ${ds.row_count.toLocaleString()} rows × ${ds.columns.length} columns`);
    } catch (e) {
      toast.error(String(e));
    }
  };

  const handleDemo = async () => {
    try {
      const ds = await loadDemoDataset();
      store.setDataset(ds);
      store.setTarget(DEMO_TARGET);
      toast.success("Demo dataset loaded — injection moulding, 5,000 rows");
    } catch {
      toast.error("Failed to load demo dataset");
    }
  };

  const canAnalyze = (): boolean => {
    if (!store.dataset || !store.target) return false;
    const controllable = store.dataset.columns.filter((c) => c.role === "controllable");
    return controllable.length > 0;
  };

  const handleAnalyze = async () => {
    if (!store.dataset || !store.target) return;
    setAnalyzing(true);
    store.setIsAnalyzing(true);
    store.setAnalyzeError(null);

    const column_roles: Record<string, ColumnRole> = {};
    for (const col of store.dataset.columns) {
      column_roles[col.name] = col.role;
    }

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
      store.setAnalysis(bundle);
      router.push("/analyze");
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : "Analysis failed — check your API connection.";
      toast.error(msg);
      store.setAnalyzeError(msg);
    } finally {
      setAnalyzing(false);
      store.setIsAnalyzing(false);
    }
  };

  const { dataset, target, improveDirection } = store;
  const controllableCount = dataset?.columns.filter((c) => c.role === "controllable").length ?? 0;
  const confoundersCount = dataset?.columns.filter((c) => c.role === "confounder").length ?? 0;

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <div className="border-b border-border/40 bg-background/80 backdrop-blur sticky top-0 z-50">
        <div className="container flex items-center justify-between h-14">
          <span className="font-bold text-sm">LeverGuide</span>
          <div className="flex gap-2 text-xs text-muted-foreground items-center">
            <span className={dataset ? "text-green-400" : ""}>1. Data</span>
            <span>→</span>
            <span className={target ? "text-green-400" : ""}>2. Target</span>
            <span>→</span>
            <span className={controllableCount > 0 ? "text-green-400" : ""}>3. Roles</span>
            <span>→</span>
            <span>4. Analyze</span>
          </div>
        </div>
      </div>

      <div className="container py-8 max-w-5xl space-y-8">
        {/* Upload card */}
        {!dataset && (
          <div
            className="rounded-xl border-2 border-dashed border-border hover:border-primary/50 transition-colors p-12 text-center cursor-pointer"
            onClick={() => fileRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const f = e.dataTransfer.files[0];
              if (f) handleFile(f);
            }}
          >
            <Upload className="h-10 w-10 text-muted-foreground mx-auto mb-4" />
            <p className="font-semibold mb-1">Drop a CSV file here or click to browse</p>
            <p className="text-sm text-muted-foreground mb-4">Max 50 MB · numeric &amp; categorical columns supported</p>
            <button
              onClick={(e) => { e.stopPropagation(); handleDemo(); }}
              className="text-sm text-primary hover:underline"
            >
              Or load the injection-moulding demo dataset →
            </button>
            <input
              ref={fileRef}
              type="file"
              accept=".csv"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
          </div>
        )}

        {dataset && (
          <>
            {/* Dataset summary bar */}
            <div className="rounded-xl border border-border/60 bg-card p-4 flex flex-wrap gap-4 items-center justify-between">
              <div>
                <p className="font-semibold text-sm">{dataset.name}</p>
                <p className="text-xs text-muted-foreground">
                  {dataset.row_count.toLocaleString()} rows · {dataset.columns.length} columns
                </p>
              </div>
              <div className="flex gap-3 text-xs">
                <Pill color="green">{controllableCount} controllable</Pill>
                <Pill color="orange">{confoundersCount} confounder</Pill>
                {target && <Pill color="blue">target: {target}</Pill>}
              </div>
              <button
                onClick={() => { store.reset(); }}
                className="text-xs text-muted-foreground hover:text-destructive transition-colors"
              >
                Clear &amp; reload
              </button>
            </div>

            {/* Target + direction */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="rounded-xl border border-border/60 bg-card p-4">
                <label className="block text-xs font-semibold mb-2 text-muted-foreground uppercase tracking-wide">
                  Target KPI
                </label>
                <select
                  value={target ?? ""}
                  onChange={(e) => store.setTarget(e.target.value || null)}
                  className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
                >
                  <option value="">Select a column…</option>
                  {dataset.columns
                    .filter((c) => c.kind === "numeric")
                    .map((c) => (
                      <option key={c.name} value={c.name}>{c.name}</option>
                    ))}
                </select>
              </div>
              <div className="rounded-xl border border-border/60 bg-card p-4">
                <label className="block text-xs font-semibold mb-2 text-muted-foreground uppercase tracking-wide">
                  Improvement direction
                </label>
                <div className="flex gap-2">
                  {(["decrease", "increase"] as const).map((d) => (
                    <button
                      key={d}
                      onClick={() => store.setImproveDirection(d)}
                      className={`flex-1 py-2 rounded-lg text-sm font-medium border transition-colors ${
                        improveDirection === d
                          ? "bg-primary text-primary-foreground border-primary"
                          : "border-border hover:bg-accent"
                      }`}
                    >
                      {d === "decrease" ? "↓ Minimise" : "↑ Maximise"}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Column roles table */}
            <div className="rounded-xl border border-border/60 bg-card overflow-hidden">
              <div className="px-4 py-3 border-b border-border/60">
                <p className="font-semibold text-sm">Column Roles</p>
                <p className="text-xs text-muted-foreground mt-0.5">
                  Set at least one column to <strong>controllable</strong> — these are the levers the engine will recommend.
                </p>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border/40 text-xs text-muted-foreground">
                      <th className="px-4 py-2 text-left">Column</th>
                      <th className="px-4 py-2 text-left">Type</th>
                      <th className="px-4 py-2 text-left">Stats</th>
                      <th className="px-4 py-2 text-left">Role</th>
                    </tr>
                  </thead>
                  <tbody>
                    {dataset.columns.map((col) => (
                      <tr key={col.name} className="border-b border-border/20 hover:bg-muted/20 transition-colors">
                        <td className="px-4 py-2 font-mono text-xs">{col.name}</td>
                        <td className="px-4 py-2">
                          <span className="text-xs text-muted-foreground">{col.kind}</span>
                        </td>
                        <td className="px-4 py-2 text-xs text-muted-foreground">
                          {col.kind === "numeric" && col.mean !== undefined
                            ? `μ=${col.mean.toFixed(2)} σ=${col.std?.toFixed(2)}`
                            : col.top_values?.[0]?.value ?? "—"}
                        </td>
                        <td className="px-4 py-2">
                          <RoleSelect
                            value={col.role}
                            onChange={(r) => {
                              if (r === "outcome") {
                                store.setTarget(col.name);
                              } else {
                                store.setColumnRole(col.name, r);
                              }
                            }}
                          />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Role legend */}
            <div className="rounded-xl border border-border/60 bg-card p-4">
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">Role Reference</p>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
                {(Object.entries(ROLE_HELP) as [ColumnRole, string][]).map(([role, help]) => (
                  <div key={role} className="text-xs">
                    <span className={`inline-block rounded px-1.5 py-0.5 border text-xs font-medium mb-1 ${ROLE_COLORS[role]}`}>
                      {role}
                    </span>
                    <p className="text-muted-foreground">{help}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Validation + CTA */}
            <div className="rounded-xl border border-border/60 bg-card p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div className="text-sm space-y-1">
                <ValidationRow ok={!!target} label="Target selected" />
                <ValidationRow ok={controllableCount > 0} label={`${controllableCount} controllable variable(s)`} />
                <ValidationRow ok={confoundersCount >= 0} label={`${confoundersCount} confounder(s) — adjusts causal estimates`} />
              </div>
              <button
                onClick={handleAnalyze}
                disabled={!canAnalyze() || analyzing}
                className="inline-flex items-center gap-2 h-11 px-7 rounded-lg bg-primary text-primary-foreground font-semibold hover:opacity-90 transition-opacity disabled:opacity-40 shrink-0"
              >
                {analyzing ? "Analyzing…" : "Run Analysis"}
                <ArrowRight className="h-4 w-4" />
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function RoleSelect({ value, onChange }: { value: ColumnRole; onChange: (r: ColumnRole) => void }) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value as ColumnRole)}
      className={`rounded px-2 py-1 text-xs border font-medium focus:outline-none ${ROLE_COLORS[value]} bg-transparent cursor-pointer`}
    >
      {ROLES.map((r) => (
        <option key={r} value={r} className="bg-background text-foreground">
          {r}
        </option>
      ))}
    </select>
  );
}

function Pill({ children, color }: { children: React.ReactNode; color: string }) {
  const map: Record<string, string> = {
    green:  "bg-green-500/10 text-green-400 border-green-500/30",
    orange: "bg-orange-500/10 text-orange-400 border-orange-500/30",
    blue:   "bg-blue-500/10 text-blue-400 border-blue-500/30",
  };
  return (
    <span className={`rounded-full border px-2 py-0.5 ${map[color] ?? ""}`}>{children}</span>
  );
}

function ValidationRow({ ok, label }: { ok: boolean; label: string }) {
  return (
    <div className="flex items-center gap-2 text-xs">
      {ok
        ? <CheckCircle2 className="h-3.5 w-3.5 text-green-400 shrink-0" />
        : <AlertCircle className="h-3.5 w-3.5 text-muted-foreground shrink-0" />}
      <span className={ok ? "text-foreground" : "text-muted-foreground"}>{label}</span>
    </div>
  );
}
