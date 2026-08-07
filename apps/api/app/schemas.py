"""
Pydantic v2 schemas for all API request/response contracts.
These are the single source of truth — the frontend TypeScript types
are generated from / kept in sync with these.
"""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ── Column / Dataset ─────────────────────────────────────────────────────────

ColumnRole = Literal[
    "outcome",
    "controllable",     # a real-time setpoint the user can change
    "planning_lever",   # a scheduling decision, not a per-interval setpoint
    "confounder",
    "mediator",
    "context",
    "identifier",
    "ignore",
    "unassigned",       # the user has not stated a causal role for this column
]
ColumnKind = Literal["numeric", "categorical", "datetime", "text"]
Task = Literal["regression"]

# What kind of claim a result is. These three are not interchangeable and the API
# never lets a consumer guess which one it is holding.
#   association             — a marginal relationship, no adjustment, no claim
#   adjusted_effect_estimate — observational effect under an assumed causal graph
#   predictive_what_if      — model inputs changed, predictions compared
ResultType = Literal["association", "adjusted_effect_estimate", "predictive_what_if"]

# How an interval was produced, or why there is none.
IntervalMethod = Literal[
    "ols_analytic_homoskedastic",   # textbook regression CI
    "row_bootstrap_fixed_model",    # resamples rows; holds the fitted model fixed
]

AnalysisMode = Literal[
    "causal",                 # effect estimation and intervention candidates
    "descriptive_predictive",  # description and prediction only; no causal claims
]

INTERPRETATION_NOTES: dict[str, str] = {
    "association": (
        "This is a marginal association. It is not adjusted for anything and "
        "carries no causal claim."
    ),
    "adjusted_effect_estimate": (
        "Interpretation depends on the selected causal graph and assumptions, "
        "including no important unmeasured confounding."
    ),
    "predictive_what_if": (
        "This modifies model inputs and compares predictions. It is not "
        "automatically a causal intervention estimate."
    ),
}


class TopValue(BaseModel):
    value: str
    count: int


class ColumnMeta(BaseModel):
    name: str
    kind: ColumnKind
    role: ColumnRole
    unique: int
    missing: int
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    median: Optional[float] = None
    p25: Optional[float] = None
    p75: Optional[float] = None
    top_values: list[TopValue] = Field(default_factory=list)


class DatasetSummary(BaseModel):
    name: str
    row_count: int
    columns: list[ColumnMeta]
    preview_rows: list[dict]  # first 10 rows


# ── DAG ───────────────────────────────────────────────────────────────────────

class DagEdge(BaseModel):
    source: str
    target: str

    @field_validator("source", "target")
    @classmethod
    def node_nonempty(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("DAG edge source and target must be non-empty strings")
        return v.strip()


class DagValidationResult(BaseModel):
    # `valid` means structurally well-formed (acyclic, known nodes). It has never
    # meant "scientifically defensible", and `dag_source` is what tells a reader
    # where the graph came from.
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    dag_source: Literal[
        "user_supplied",
        "declared_domain_ontology",
        "assumed_from_roles",
    ] = "assumed_from_roles"
    graph_assumption: Optional[str] = None


# ── Analysis Request ─────────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    dataset_csv: str = Field(description="Raw CSV content as string")
    dataset_name: str = "Uploaded Dataset"
    target: str
    task: Task = "regression"
    improve_direction: Literal["decrease", "increase"] = "decrease"
    analysis_mode: AnalysisMode = "causal"
    column_roles: dict[str, ColumnRole] = Field(default_factory=dict)
    dag_edges: list[DagEdge] = Field(default_factory=list)
    random_seed: int = 42

    @field_validator("target")
    @classmethod
    def target_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("target must not be empty")
        return v


# ── Predictive Results ────────────────────────────────────────────────────────

class FeatureImportance(BaseModel):
    feature: str
    importance: float
    importance_norm: float  # 0–1 normalised


class Coefficient(BaseModel):
    feature: str
    coef: float
    std_err: float
    t_stat: float
    p_value: float
    significant: bool  # p < 0.05


class ModelMetrics(BaseModel):
    r2: float
    adj_r2: Optional[float] = None
    rmse: float
    mae: float
    cv_r2_mean: Optional[float] = None
    cv_r2_std: Optional[float] = None
    n_train: int
    n_test: int


class PredictionPoint(BaseModel):
    actual: float
    predicted: float
    residual: float


ModelKey = Literal["ols", "ridge", "rf", "xgb", "lgbm"]


class PredictiveResult(BaseModel):
    model: ModelKey
    display_name: str
    task: Task
    metrics: ModelMetrics
    importances: list[FeatureImportance]
    predictions: list[PredictionPoint]  # test set, max 600
    coefficients: Optional[list[Coefficient]] = None  # OLS/Ridge only
    is_winner: bool = False


class ModelStatus(BaseModel):
    """
    What actually happened to every configured model.

    A model that could not run appears here with a reason. It is never omitted,
    because omission reads as "not configured" rather than "did not run".
    """
    model: ModelKey
    display_name: str
    status: Literal[
        "succeeded",
        "unavailable_dependency",   # the library could not be imported
        "training_failed",          # the library loaded but fitting raised
        "skipped_by_configuration",
    ]
    detail: Optional[str] = None


# ── Adjusted effect estimates ─────────────────────────────────────────────────

class CausalEffect(BaseModel):
    """
    An observational effect estimate under an assumed causal graph.

    Named `CausalEffect` for backward compatibility of the API surface; every
    field that a reader interprets says `adjusted_effect_estimate`.
    """
    result_type: ResultType = "adjusted_effect_estimate"
    feature: str
    effect_per_std: float          # adjusted β for +1 SD change
    effect_raw: float              # unstandardised β
    std_err: float
    t_stat: float
    p_value: float
    conf_int_lo: float
    conf_int_hi: float
    interval_method: IntervalMethod = "ols_analytic_homoskedastic"
    adjusted_for: list[str]
    adjustment_set_source: Literal["declared_domain_dag", "derived_from_graph"] = (
        "derived_from_graph"
    )
    estimand: str = "total effect on the outcome, under the stated graph"
    causal_role: Optional[str] = None
    controllable: bool
    n_observations: int = 0
    evidence_strength: Literal["strong", "moderate", "weak", "insufficient"] = "insufficient"
    # True when the 95 % interval includes zero — the "do not act on this" signal
    # the source study used, which a p-value threshold alone does not give.
    interval_excludes_zero: bool = False
    interpretation_note: str = INTERPRETATION_NOTES["adjusted_effect_estimate"]
    warning: Optional[str] = None
    notes: list[str] = Field(default_factory=list)


# ── Predictive what-if simulations ────────────────────────────────────────────

EvidenceStrength = Literal["strong", "moderate", "weak"]

# Whether a simulated change may be presented as a candidate action.
#   eligible            — feasible, supported, evidence agrees; may be ranked
#   exploratory         — representable but the mechanism is only partly modelled
#   unsupported         — outside observed support, or the model did not run
#   infeasible          — violates a documented physical constraint
#   conflicting_evidence — the adjusted estimate disagrees with the simulation
InterventionStatus = Literal[
    "eligible", "exploratory", "unsupported", "infeasible", "conflicting_evidence",
]

SupportStatus = Literal[
    "within_observed",
    "outside_observed_within_declared",
    "outside_declared",
    "unknown",
]


class FeasibilityCheck(BaseModel):
    """One named check and what it concluded."""
    check: str
    passed: bool
    detail: str


class Intervention(BaseModel):
    result_type: ResultType = "predictive_what_if"
    # 0 means unranked. Only `eligible` results receive a rank.
    rank: int
    feature: str
    direction: Literal["increase", "decrease"]
    current_mean: float
    current_p10: float
    current_p90: float
    suggested_value: float
    delta: float
    delta_pct: float

    expected_kpi_change: float
    expected_kpi_change_pct: float
    # Null when no interval could be computed. Never filled with a placeholder.
    expected_kpi_change_lo: Optional[float] = None
    expected_kpi_change_hi: Optional[float] = None
    interval_method: Optional[IntervalMethod] = None
    uncertainty_status: str = "not_computed"

    status: InterventionStatus = "exploratory"
    status_reason: str = ""
    support_status: SupportStatus = "unknown"
    feasibility_checks: list[FeasibilityCheck] = Field(default_factory=list)

    evidence_strength: EvidenceStrength
    # Whether the adjusted estimate agrees with the simulated direction. Replaces
    # the old `evidence_type: "causal"`, which asserted a causal reading from a
    # p-value alone and never checked the sign.
    adjustment_support: Literal["aligned", "conflicting", "inconclusive", "none"] = "none"
    simulation_model: str = "gradient_boosting_regressor"
    simulation_evaluation: str = "in-sample: fitted and evaluated on the same rows"
    tradeoff: str
    rationale: str
    assumptions: list[str]
    caveat: str
    interpretation_note: str = INTERPRETATION_NOTES["predictive_what_if"]


# ── EDA ───────────────────────────────────────────────────────────────────────

class CorrelationPair(BaseModel):
    result_type: ResultType = "association"
    feature_a: str
    feature_b: str
    correlation: float
    abs_correlation: float


class DistributionBucket(BaseModel):
    bin_lo: float
    bin_hi: float
    count: int
    pct: float


class FeatureDistribution(BaseModel):
    feature: str
    kind: ColumnKind
    distribution: list[DistributionBucket]  # numeric histogram
    categorical_counts: list[TopValue] = Field(default_factory=list)


# ── Executive Summary ─────────────────────────────────────────────────────────

class ExecutiveSummary(BaseModel):
    headline: str
    sub_headline: str
    best_model_name: str
    best_model_r2: float
    top_levers: list[str]
    bullets: list[str]
    cautions: list[str]
    methodology_note: str
    disclaimer: str


# ── Full Analysis Bundle ──────────────────────────────────────────────────────

class AnalysisProvenance(BaseModel):
    """
    Enough metadata to reconstruct what produced every number in this bundle.

    The UI shows a compact version of this; the API contract keeps all of it.
    """
    analysis_mode: AnalysisMode
    ontology_id: Optional[str] = None
    ontology_version: Optional[str] = None
    graph_assumption: Optional[str] = None
    dag_source: str = "assumed_from_roles"
    adjustment_set_source: str = "derived_from_graph"
    effect_estimator: str = "ordinary least squares on standardised columns"
    effect_interval_method: Optional[str] = None
    simulation_model: Optional[str] = None
    simulation_evaluation: Optional[str] = None
    simulation_interval_method: Optional[str] = None
    n_rows_supplied: int = 0
    n_rows_analysed: int = 0
    sampling_note: Optional[str] = None
    train_eval_strategy: str = ""
    random_seed: int = 42
    column_roles: dict[str, str] = Field(default_factory=dict)
    # Columns dropped from a fitted design matrix — e.g. zero-variance
    # confounders, or a treatment with no observed variation. Never a silent
    # scientific change: the reason travels with the column.
    excluded_columns: list["ExcludedColumn"] = Field(default_factory=list)


class ConfigurationProblem(BaseModel):
    """A specific thing that is missing or wrong, and how to fix it."""
    code: str
    message: str
    remedy: str
    columns: list[str] = Field(default_factory=list)


class ExcludedColumn(BaseModel):
    """
    A column left out of a fitted model, and why.

    Exclusion is not silence: the user's original role assignment for this
    column is unchanged (see `AnalysisProvenance.column_roles`), only the
    fitted design matrix for the named lever's estimate omits it.
    """
    column: str
    scope: Literal["treatment", "adjustment_set"]
    lever: str
    reason: str


class AnalysisBundle(BaseModel):
    request_id: str
    dataset_name: str
    target: str
    task: Task
    analysis_mode: AnalysisMode = "causal"
    row_count: int
    feature_count: int
    controllable_count: int

    predictive: list[PredictiveResult]
    model_statuses: list[ModelStatus] = Field(default_factory=list)
    best_model: str  # model key
    causal: list[CausalEffect]
    # Every candidate, with a status. Only `status == "eligible"` entries carry a
    # rank; the rest are diagnostics and the UI renders them separately.
    interventions: list[Intervention]
    correlations: list[CorrelationPair]
    distributions: list[FeatureDistribution]
    executive: ExecutiveSummary
    dag_validation: DagValidationResult
    provenance: Optional[AnalysisProvenance] = None
    warnings: list[str] = Field(default_factory=list)
    runtime_seconds: float = 0.0

    @property
    def ranked_interventions(self) -> list[Intervention]:
        return [iv for iv in self.interventions if iv.status == "eligible"]


# ── Copilot / RAG ─────────────────────────────────────────────────────────────

class CopilotAskRequest(BaseModel):
    analysis_id: str
    question: str = Field(min_length=2, max_length=1000)
    max_citations: int = Field(default=5, ge=1, le=8)


class CopilotCitation(BaseModel):
    artifact_id: str
    title: str
    kind: Literal["dataset", "summary", "dag", "model", "causal", "intervention", "eda"]
    snippet: str
    score: float
    metadata: dict = Field(default_factory=dict)


class CopilotAnswerResponse(BaseModel):
    answer: str
    citations: list[CopilotCitation]
    retrieved_artifact_ids: list[str]
    model: Optional[str] = None
    used_llm: bool = False
    warnings: list[str] = Field(default_factory=list)
