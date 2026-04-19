import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import type { AnalysisBundle, ColumnRole, DagEdge, ParsedDataset } from "./types";

interface AppState {
  dataset: ParsedDataset | null;
  target: string | null;
  improveDirection: "decrease" | "increase";
  dagEdges: DagEdge[];
  analysis: AnalysisBundle | null;
  isAnalyzing: boolean;
  analyzeError: string | null;

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

const initial = {
  dataset: null, target: null,
  improveDirection: "decrease" as const,
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
      // Don't persist the CSV content — it's large and slows storage
      partialize: (s) => ({
        target: s.target,
        improveDirection: s.improveDirection,
        dagEdges: s.dagEdges,
        analysis: s.analysis,
        dataset: s.dataset
          ? { ...s.dataset, csv_content: "" }
          : null,
      }),
    }
  )
);
