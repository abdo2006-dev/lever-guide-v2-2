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
    "outcome", "controllable", "confounder",
    "mediator", "context", "identifier", "ignore"
]
ColumnKind = Literal["numeric", "categorical", "datetime", "text"]
Task = Literal["regression"]


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
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ── Analysis Request ─────────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    dataset_csv: str = Field(description="Raw CSV content as string")
    dataset_name: str = "Uploaded Dataset"
    target: str
    task: Task = "regression"
    improve_direction: Literal["decrease", "increase"] = "decrease"
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


class PredictiveResult(BaseModel):
    model: Literal["ols", "ridge", "rf", "xgb", "lgbm"]
    display_name: str
    task: Task
    metrics: ModelMetrics
    importances: list[FeatureImportance]
    predictions: list[PredictionPoint]  # test set, max 600
    coefficients: Optional[list[Coefficient]] = None  # OLS/Ridge only
    is_winner: bool = False


# ── Causal Results ────────────────────────────────────────────────────────────

class CausalEffect(BaseModel):
    feature: str
    effect_per_std: float          # adjusted β for +1 SD change
    effect_raw: float              # unstandardised β
    std_err: float
    t_stat: float
    p_value: float
    conf_int_lo: float
    conf_int_hi: float
    adjusted_for: list[str]
    controllable: bool
    evidence_strength: Literal["strong", "moderate", "weak", "insufficient"]
    warning: Optional[str] = None


# ── Interventions ─────────────────────────────────────────────────────────────

EvidenceStrength = Literal["strong", "moderate", "weak"]


class Intervention(BaseModel):
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
    evidence_strength: EvidenceStrength
    evidence_type: Literal["causal", "predictive", "mixed"]
    tradeoff: str
    rationale: str
    assumptions: list[str]
    caveat: str


# ── EDA ───────────────────────────────────────────────────────────────────────

class CorrelationPair(BaseModel):
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

class AnalysisBundle(BaseModel):
    request_id: str
    dataset_name: str
    target: str
    task: Task
    row_count: int
    feature_count: int
    controllable_count: int

    predictive: list[PredictiveResult]
    best_model: str  # model key
    causal: list[CausalEffect]
    interventions: list[Intervention]
    correlations: list[CorrelationPair]
    distributions: list[FeatureDistribution]
    executive: ExecutiveSummary
    dag_validation: DagValidationResult
    warnings: list[str] = Field(default_factory=list)
    runtime_seconds: float = 0.0


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
