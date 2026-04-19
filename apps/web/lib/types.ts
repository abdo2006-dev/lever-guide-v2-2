// TypeScript types — kept in sync with apps/api/app/schemas.py

export type ColumnRole =
  | "outcome"
  | "controllable"
  | "confounder"
  | "mediator"
  | "context"
  | "identifier"
  | "ignore";

export type ColumnKind = "numeric" | "categorical" | "datetime" | "text";
export type Task = "regression" | "classification" | "auto";
export type EvidenceStrength = "strong" | "moderate" | "weak" | "insufficient";
export type EvidenceType = "causal" | "predictive" | "mixed";

export interface TopValue {
  value: string;
  count: number;
}

export interface ColumnMeta {
  name: string;
  kind: ColumnKind;
  role: ColumnRole;
  unique: number;
  missing: number;
  min?: number;
  max?: number;
  mean?: number;
  std?: number;
  median?: number;
  p25?: number;
  p75?: number;
  top_values: TopValue[];
}

export interface DatasetSummary {
  name: string;
  row_count: number;
  columns: ColumnMeta[];
  preview_rows: Record<string, unknown>[];
}

export interface DagEdge {
  source: string;
  target: string;
}

export interface DagValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

// ── Analysis request ──────────────────────────────────────────────────────────

export interface AnalysisRequest {
  dataset_csv: string;
  dataset_name: string;
  target: string;
  task: Task;
  improve_direction: "decrease" | "increase";
  column_roles: Record<string, ColumnRole>;
  dag_edges: DagEdge[];
  random_seed?: number;
}

// ── Predictive ────────────────────────────────────────────────────────────────

export interface FeatureImportance {
  feature: string;
  importance: number;
  importance_norm: number;
}

export interface Coefficient {
  feature: string;
  coef: number;
  std_err: number;
  t_stat: number;
  p_value: number;
  significant: boolean;
}

export interface ModelMetrics {
  r2: number;
  adj_r2?: number;
  rmse: number;
  mae: number;
  cv_r2_mean?: number;
  cv_r2_std?: number;
  n_train: number;
  n_test: number;
}

export interface PredictionPoint {
  actual: number;
  predicted: number;
  residual: number;
}

export type ModelKey = "ols" | "ridge" | "rf" | "xgb" | "lgbm";

export interface PredictiveResult {
  model: ModelKey;
  display_name: string;
  task: Task;
  metrics: ModelMetrics;
  importances: FeatureImportance[];
  predictions: PredictionPoint[];
  coefficients?: Coefficient[];
  is_winner: boolean;
}

// ── Causal ────────────────────────────────────────────────────────────────────

export interface CausalEffect {
  feature: string;
  effect_per_std: number;
  effect_raw: number;
  std_err: number;
  t_stat: number;
  p_value: number;
  conf_int_lo: number;
  conf_int_hi: number;
  adjusted_for: string[];
  controllable: boolean;
  evidence_strength: EvidenceStrength;
  warning?: string;
}

// ── Interventions ─────────────────────────────────────────────────────────────

export interface Intervention {
  rank: number;
  feature: string;
  direction: "increase" | "decrease";
  current_mean: number;
  current_p10: number;
  current_p90: number;
  suggested_value: number;
  delta: number;
  delta_pct: number;
  expected_kpi_change: number;
  expected_kpi_change_pct: number;
  evidence_strength: EvidenceStrength;
  evidence_type: EvidenceType;
  tradeoff: string;
  rationale: string;
  assumptions: string[];
  caveat: string;
}

// ── EDA ───────────────────────────────────────────────────────────────────────

export interface CorrelationPair {
  feature_a: string;
  feature_b: string;
  correlation: number;
  abs_correlation: number;
}

export interface DistributionBucket {
  bin_lo: number;
  bin_hi: number;
  count: number;
  pct: number;
}

export interface FeatureDistribution {
  feature: string;
  kind: ColumnKind;
  distribution: DistributionBucket[];
  categorical_counts: TopValue[];
}

// ── Executive ─────────────────────────────────────────────────────────────────

export interface ExecutiveSummary {
  headline: string;
  sub_headline: string;
  best_model_name: string;
  best_model_r2: number;
  top_levers: string[];
  bullets: string[];
  cautions: string[];
  methodology_note: string;
  disclaimer: string;
}

// ── Bundle ────────────────────────────────────────────────────────────────────

export interface AnalysisBundle {
  request_id: string;
  dataset_name: string;
  target: string;
  task: Task;
  row_count: number;
  feature_count: number;
  controllable_count: number;
  predictive: PredictiveResult[];
  best_model: ModelKey;
  causal: CausalEffect[];
  interventions: Intervention[];
  correlations: CorrelationPair[];
  distributions: FeatureDistribution[];
  executive: ExecutiveSummary;
  dag_validation: DagValidationResult;
  warnings: string[];
  runtime_seconds: number;
}

// ── Local state (no server) ───────────────────────────────────────────────────

export interface ParsedDataset {
  name: string;
  csv_content: string;
  columns: ColumnMeta[];
  preview_rows: Record<string, unknown>[];
  row_count: number;
}
