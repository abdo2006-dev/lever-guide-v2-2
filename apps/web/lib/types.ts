// TypeScript types — kept in sync with apps/api/app/schemas.py

export type ColumnRole =
  | "outcome"
  | "controllable"
  | "planning_lever"
  | "confounder"
  | "mediator"
  | "context"
  | "identifier"
  | "ignore"
  | "unassigned";

export type ColumnKind = "numeric" | "categorical" | "datetime" | "text";
export type Task = "regression";
export type AnalysisMode = "causal" | "descriptive_predictive";
export type EvidenceStrength = "strong" | "moderate" | "weak" | "insufficient";

/**
 * What kind of claim a result is. These three are never interchangeable:
 *  - association: a marginal relationship, no adjustment, no causal claim
 *  - adjusted_effect_estimate: observational effect under an assumed causal graph
 *  - predictive_what_if: model inputs changed and predictions compared
 */
export type ResultType =
  | "association"
  | "adjusted_effect_estimate"
  | "predictive_what_if";

export type IntervalMethod =
  | "ols_analytic_homoskedastic"
  | "row_bootstrap_fixed_model";

export type InterventionStatus =
  | "eligible"
  | "exploratory"
  | "unsupported"
  | "infeasible"
  | "conflicting_evidence";

export type SupportStatus =
  | "within_observed"
  | "outside_observed_within_declared"
  | "outside_declared"
  | "unknown";

export type AdjustmentSupport = "aligned" | "conflicting" | "inconclusive" | "none";

export const RESULT_TYPE_LABEL: Record<ResultType, string> = {
  association: "Association",
  adjusted_effect_estimate: "Adjusted observational effect estimate",
  predictive_what_if: "Predictive what-if simulation",
};

export const RESULT_TYPE_NOTE: Record<ResultType, string> = {
  association:
    "This is a marginal association. It is not adjusted for anything and carries no causal claim.",
  adjusted_effect_estimate:
    "Interpretation depends on the selected causal graph and assumptions, including no important unmeasured confounding.",
  predictive_what_if:
    "This modifies model inputs and compares predictions. It is not automatically a causal intervention estimate.",
};

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
  /** Structurally well-formed (acyclic, known nodes) — not "scientifically proven". */
  valid: boolean;
  errors: string[];
  warnings: string[];
  dag_source: "user_supplied" | "declared_domain_ontology" | "assumed_from_roles";
  graph_assumption?: string | null;
}

// ── Analysis request ──────────────────────────────────────────────────────────

export interface AnalysisRequest {
  dataset_csv: string;
  dataset_name: string;
  target: string;
  task: Task;
  improve_direction: "decrease" | "increase";
  analysis_mode: AnalysisMode;
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
  result_type: ResultType;
  feature: string;
  effect_per_std: number;
  effect_raw: number;
  std_err: number;
  t_stat: number;
  p_value: number;
  conf_int_lo: number;
  conf_int_hi: number;
  interval_method: IntervalMethod;
  adjusted_for: string[];
  adjustment_set_source: "declared_domain_dag" | "derived_from_graph";
  estimand: string;
  causal_role?: string | null;
  controllable: boolean;
  n_observations: number;
  evidence_strength: EvidenceStrength;
  interval_excludes_zero: boolean;
  interpretation_note: string;
  warning?: string;
  notes: string[];
}

// ── Predictive what-if simulations ────────────────────────────────────────────

export interface FeasibilityCheck {
  check: string;
  passed: boolean;
  detail: string;
}

export interface Intervention {
  result_type: ResultType;
  /** 0 means unranked — only `eligible` results are ranked. */
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
  /** Null when no interval could be computed — never a placeholder value. */
  expected_kpi_change_lo?: number | null;
  expected_kpi_change_hi?: number | null;
  interval_method?: IntervalMethod | null;
  uncertainty_status: string;
  status: InterventionStatus;
  status_reason: string;
  support_status: SupportStatus;
  feasibility_checks: FeasibilityCheck[];
  evidence_strength: EvidenceStrength;
  adjustment_support: AdjustmentSupport;
  simulation_model: string;
  simulation_evaluation: string;
  tradeoff: string;
  rationale: string;
  assumptions: string[];
  caveat: string;
  interpretation_note: string;
}

// ── EDA ───────────────────────────────────────────────────────────────────────

export interface CorrelationPair {
  result_type: ResultType;
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

export type ModelRunStatus =
  | "succeeded"
  | "unavailable_dependency"
  | "training_failed"
  | "skipped_by_configuration";

export interface ModelStatus {
  model: ModelKey;
  display_name: string;
  status: ModelRunStatus;
  detail?: string | null;
}

/** A column dropped from a fitted design matrix, and why. Not a silent change. */
export interface ExcludedColumn {
  column: string;
  scope: "treatment" | "adjustment_set";
  lever: string;
  reason: string;
}

export interface AnalysisProvenance {
  analysis_mode: AnalysisMode;
  ontology_id?: string | null;
  ontology_version?: string | null;
  graph_assumption?: string | null;
  dag_source: string;
  adjustment_set_source: string;
  effect_estimator: string;
  effect_interval_method?: string | null;
  simulation_model?: string | null;
  simulation_evaluation?: string | null;
  simulation_interval_method?: string | null;
  n_rows_supplied: number;
  n_rows_analysed: number;
  sampling_note?: string | null;
  train_eval_strategy: string;
  random_seed: number;
  column_roles: Record<string, string>;
  excluded_columns: ExcludedColumn[];
}

export interface AnalysisBundle {
  request_id: string;
  dataset_name: string;
  target: string;
  task: Task;
  analysis_mode: AnalysisMode;
  row_count: number;
  feature_count: number;
  controllable_count: number;
  predictive: PredictiveResult[];
  model_statuses: ModelStatus[];
  best_model: ModelKey;
  causal: CausalEffect[];
  interventions: Intervention[];
  correlations: CorrelationPair[];
  distributions: FeatureDistribution[];
  executive: ExecutiveSummary;
  dag_validation: DagValidationResult;
  provenance?: AnalysisProvenance | null;
  warnings: string[];
  runtime_seconds: number;
}

// ── Copilot / RAG ─────────────────────────────────────────────────────────────

export interface CopilotAskRequest {
  analysis_id: string;
  question: string;
  max_citations?: number;
}

export interface CopilotCitation {
  artifact_id: string;
  title: string;
  kind: "dataset" | "summary" | "dag" | "model" | "causal" | "intervention" | "eda";
  snippet: string;
  score: number;
  metadata: Record<string, unknown>;
}

export interface CopilotAnswerResponse {
  answer: string;
  citations: CopilotCitation[];
  retrieved_artifact_ids: string[];
  model?: string;
  used_llm: boolean;
  warnings: string[];
}

// ── Local state (no server) ───────────────────────────────────────────────────

export interface ParsedDataset {
  name: string;
  csv_content: string;
  columns: ColumnMeta[];
  preview_rows: Record<string, unknown>[];
  row_count: number;
}
