import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import type {
  AnalysisBundle, AnalysisMode, ColumnRole, DagEdge, ParsedDataset,
} from "./types";

interface AppState {
  dataset: ParsedDataset | null;
  target: string | null;
  improveDirection: "decrease" | "increase";
  analysisMode: AnalysisMode;
  dagEdges: DagEdge[];
  analysis: AnalysisBundle | null;
  isAnalyzing: boolean;
  analyzeError: string | null;

  setDataset: (ds: ParsedDataset | null) => void;
  setColumnRole: (column: string, role: ColumnRole) => void;
  setTarget: (t: string | null) => void;
  setImproveDirection: (d: "decrease" | "increase") => void;
  setAnalysisMode: (m: AnalysisMode) => void;
  setDagEdges: (edges: DagEdge[]) => void;
  setAnalysis: (a: AnalysisBundle | null) => void;
  setIsAnalyzing: (b: boolean) => void;
  setAnalyzeError: (e: string | null) => void;
  reset: () => void;
}

const initial = {
  dataset: null, target: null,
  improveDirection: "decrease" as const,
  analysisMode: "causal" as AnalysisMode,
  dagEdges: [], analysis: null,
  isAnalyzing: false, analyzeError: null,
};

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      ...initial,
      setDataset: (ds) => set({ dataset: ds, analysis: null, analyzeError: null }),
      setColumnRole: (column, role) =>
        set((s) => {
          if (!s.dataset) return s;
          const columns = s.dataset.columns.map((c) =>
            c.name === column ? { ...c, role } : c);
          return { dataset: { ...s.dataset, columns }, analysis: null };
        }),
      setTarget: (t) =>
        set((s) => {
          if (!s.dataset) return { target: t };
          const columns = s.dataset.columns.map((c) => {
            if (c.name === t) return { ...c, role: "outcome" as ColumnRole };
            // A column that stops being the outcome returns to "unassigned".
            // Promoting it to "confounder" would assert a causal role the user
            // never gave it.
            if (c.role === "outcome") return { ...c, role: "unassigned" as ColumnRole };
            return c;
          });
          return { target: t, dataset: { ...s.dataset, columns }, analysis: null };
        }),
      setImproveDirection: (d) => set({ improveDirection: d }),
      setAnalysisMode: (m) => set({ analysisMode: m, analysis: null }),
      setDagEdges: (edges) => set({ dagEdges: edges, analysis: null }),
      setAnalysis: (a) => set({ analysis: a, analyzeError: null }),
      setIsAnalyzing: (b) => set({ isAnalyzing: b }),
      setAnalyzeError: (e) => set({ analyzeError: e, isAnalyzing: false }),
      reset: () => set(initial),
    }),
    {
      name: "leverguide-state",
      storage: createJSONStorage(() =>
        typeof window !== "undefined" ? sessionStorage : {
          getItem: () => null,
          setItem: () => {},
          removeItem: () => {},
        }
      ),
      // Persist EVERYTHING including csv_content — sessionStorage handles ~5MB fine
      // The analysis bundle is ~200KB, CSV is typically <2MB
      partialize: (s) => ({
        target: s.target,
        improveDirection: s.improveDirection,
        analysisMode: s.analysisMode,
        dagEdges: s.dagEdges,
        analysis: s.analysis,
        dataset: s.dataset,   // ← full dataset including csv_content
      }),
      // A session stored before the role vocabulary grew can hold a role this
      // build no longer knows. Fall back to "unassigned" rather than trusting it.
      merge: (persisted, current) => {
        const state = { ...current, ...(persisted as Partial<AppState>) };
        if (state.dataset) {
          const known: ColumnRole[] = [
            "outcome", "controllable", "planning_lever", "confounder",
            "mediator", "context", "identifier", "ignore", "unassigned",
          ];
          state.dataset = {
            ...state.dataset,
            columns: state.dataset.columns.map((c) =>
              known.includes(c.role) ? c : { ...c, role: "unassigned" as ColumnRole }),
          };
        }
        return state;
      },
    }
  )
);
