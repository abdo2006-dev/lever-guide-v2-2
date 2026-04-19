import { create } from "zustand";
import type { AnalysisBundle, ColumnRole, DagEdge, ParsedDataset } from "./types";

interface AppState {
  // Step 1: parsed dataset (client-side)
  dataset: ParsedDataset | null;
  // Step 2: configuration
  target: string | null;
  improveDirection: "decrease" | "increase";
  dagEdges: DagEdge[];
  // Step 3: analysis results
  analysis: AnalysisBundle | null;
  isAnalyzing: boolean;
  analyzeError: string | null;

  // Actions
  setDataset: (ds: ParsedDataset | null) => void;
  setColumnRole: (column: string, role: ColumnRole) => void;
  setTarget: (t: string | null) => void;
  setImproveDirection: (d: "decrease" | "increase") => void;
  setDagEdges: (edges: DagEdge[]) => void;
  setAnalysis: (a: AnalysisBundle | null) => void;
  setIsAnalyzing: (b: boolean) => void;
  setAnalyzeError: (e: string | null) => void;
  reset: () => void;
}

const initialState = {
  dataset: null,
  target: null,
  improveDirection: "decrease" as const,
  dagEdges: [],
  analysis: null,
  isAnalyzing: false,
  analyzeError: null,
};

export const useAppStore = create<AppState>((set) => ({
  ...initialState,

  setDataset: (ds) =>
    set({ dataset: ds, analysis: null, analyzeError: null }),

  setColumnRole: (column, role) =>
    set((s) => {
      if (!s.dataset) return s;
      const columns = s.dataset.columns.map((c) =>
        c.name === column ? { ...c, role } : c,
      );
      return { dataset: { ...s.dataset, columns }, analysis: null };
    }),

  setTarget: (t) =>
    set((s) => {
      if (!s.dataset) return { target: t };
      const columns = s.dataset.columns.map((c) => {
        if (c.name === t) return { ...c, role: "outcome" as ColumnRole };
        if (c.role === "outcome") return { ...c, role: "confounder" as ColumnRole };
        return c;
      });
      return { target: t, dataset: { ...s.dataset, columns }, analysis: null };
    }),

  setImproveDirection: (d) => set({ improveDirection: d }),
  setDagEdges: (edges) => set({ dagEdges: edges, analysis: null }),
  setAnalysis: (a) => set({ analysis: a, analyzeError: null }),
  setIsAnalyzing: (b) => set({ isAnalyzing: b }),
  setAnalyzeError: (e) => set({ analyzeError: e, isAnalyzing: false }),
  reset: () => set(initialState),
}));
