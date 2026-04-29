"""
/api/analyze  —  Main analysis endpoint.
Accepts a JSON request with CSV content and configuration,
runs the full pipeline, and returns an AnalysisBundle.
"""
from __future__ import annotations
import io
import time
import uuid
import logging
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from scipy import stats as scipy_stats

from app.schemas import (
    AnalysisBundle, AnalysisRequest, CorrelationPair,
    DistributionBucket, FeatureDistribution, ExecutiveSummary,
    TopValue, CopilotAskRequest, CopilotAnswerResponse,
)
from app.utils.preprocess import build_feature_matrix, build_column_meta, infer_column_kind
from app.utils.dag import auto_dag, validate_dag
from app.models.pipeline import run_predictive_pipeline
from app.models.causal import run_causal_analysis
from app.models.intervention import run_intervention_engine
from app.rag import answer_with_groq, index_analysis_session, retrieve
from app.utils.wandb_tracking import track_analysis_run

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Demo role defaults for injection moulding dataset ────────────────────────

DEMO_ROLES: dict[str, str] = {
    "scrap_rate_pct":            "outcome",
    # Process-controllable knobs
    "barrel_temperature_c":      "controllable",
    "mold_temperature_c":        "controllable",
    "injection_pressure_bar":    "controllable",
    "hold_pressure_bar":         "controllable",
    "screw_speed_rpm":           "controllable",
    "cooling_time_s":            "controllable",
    "clamp_force_kn":            "controllable",
    "shot_size_g":               "controllable",
    # Environmental confounders — cause both process settings and scrap
    "ambient_temperature_c":     "confounder",
    "ambient_humidity_pct":      "confounder",
    "resin_moisture_pct":        "confounder",
    "resin_batch_quality_index": "confounder",
    "dryer_dewpoint_c":          "confounder",
    # Context: design factors fixed per run
    "cavity_count":              "context",
    "product_variant":           "context",
    "operator_experience_level": "context",
    "operator_shift":            "context",
    # Wear/maintenance indices — context (not directly controllable intra-run)
    "tool_wear_index":           "context",
    "calibration_drift_index":   "context",
    "maintenance_days_since_last": "context",
    # Mediators (on the causal path: cycle_time mediates settings → scrap)
    "cycle_time_s":              "mediator",
    "part_weight_g":             "mediator",
    # Identifiers / derived
    "timestamp":                 "identifier",
    "plant_id":                  "identifier",
    "machine_id":                "identifier",
    "mold_id":                   "identifier",
    "resin_lot_id":              "identifier",
    "defect_type":               "ignore",
    "scrap_count":               "ignore",
    "parts_produced":            "ignore",
    "energy_kwh_interval":       "ignore",
    "pass_fail_flag":            "ignore",
}


def _assign_roles(df: pd.DataFrame, column_roles: dict[str, str], target: str) -> dict[str, str]:
    roles: dict[str, str] = {}
    for col in df.columns:
        if col == target:
            roles[col] = "outcome"
        elif col in column_roles:
            roles[col] = column_roles[col]
        else:
            kind = infer_column_kind(df[col])
            if kind in ("text", "datetime"):
                roles[col] = "ignore"
            elif df[col].nunique() > 50 and kind == "categorical":
                roles[col] = "ignore"
            else:
                roles[col] = "confounder"
    return roles


def _coerce_and_validate_target(df: pd.DataFrame, target: str) -> pd.Series:
    target_numeric = pd.to_numeric(df[target], errors="coerce")
    non_null = target_numeric.dropna()
    if len(non_null) < 30:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Target column '{target}' must contain at least 30 numeric, non-missing rows "
                "for regression analysis."
            ),
        )
    if non_null.nunique() < 2:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Target column '{target}' must vary across rows. "
                "Constant targets are not valid for regression analysis."
            ),
        )
    return target_numeric


def _compute_correlations(df: pd.DataFrame, cols: list[str]) -> list[CorrelationPair]:
    pairs: list[CorrelationPair] = []
    num_df = df[cols].select_dtypes(include="number")
    if num_df.shape[1] < 2:
        return pairs
    corr = num_df.corr()
    names = list(corr.columns)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            v = float(corr.iloc[i, j])
            if not np.isnan(v):
                pairs.append(CorrelationPair(
                    feature_a=names[i],
                    feature_b=names[j],
                    correlation=round(v, 4),
                    abs_correlation=round(abs(v), 4),
                ))
    return sorted(pairs, key=lambda p: -p.abs_correlation)[:100]


def _compute_distributions(df: pd.DataFrame, cols: list[str]) -> list[FeatureDistribution]:
    out: list[FeatureDistribution] = []
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        kind = infer_column_kind(s)
        if kind == "numeric":
            vals = pd.to_numeric(s, errors="coerce").dropna().values
            if len(vals) == 0:
                continue
            counts, edges = np.histogram(vals, bins=min(30, len(np.unique(vals))))
            total = len(vals)
            buckets = [
                DistributionBucket(
                    bin_lo=float(edges[i]),
                    bin_hi=float(edges[i + 1]),
                    count=int(counts[i]),
                    pct=round(100 * int(counts[i]) / total, 2),
                )
                for i in range(len(counts))
            ]
            out.append(FeatureDistribution(feature=col, kind="numeric", distribution=buckets))
        else:
            vc = s.value_counts().head(15)
            out.append(FeatureDistribution(
                feature=col,
                kind="categorical",
                distribution=[],
                categorical_counts=[
                    TopValue(value=str(v), count=int(c)) for v, c in vc.items()
                ],
            ))
    return out


def _build_executive(
    target: str,
    best_model_name: str,
    best_r2: float,
    interventions: list[Any],
    causal_effects: list[Any],
    warnings: list[str],
) -> ExecutiveSummary:
    top_levers = [iv.feature for iv in interventions[:3]]
    bullets = []
    if best_r2 > 0.7:
        bullets.append(
            f"The predictive model explains {best_r2 * 100:.0f}% of variance in {target} — a strong fit."
        )
    elif best_r2 > 0.4:
        bullets.append(
            f"The model captures {best_r2 * 100:.0f}% of variance in {target} — moderate predictive power."
        )
    else:
        bullets.append(
            f"Model fit is modest (R²={best_r2:.2f}). Interpret recommendations conservatively."
        )

    for iv in interventions[:3]:
        direction_word = "reducing" if iv.direction == "decrease" else "increasing"
        bullets.append(
            f"{direction_word.capitalize()} {iv.feature} is estimated to "
            f"{'reduce' if iv.expected_kpi_change < 0 else 'increase'} {target} "
            f"by ~{abs(iv.expected_kpi_change_pct):.1f}% "
            f"({iv.evidence_type} evidence, {iv.evidence_strength} strength)."
        )

    cautions = [
        "All estimates are based on observational data — not randomised experiments.",
        "Causal estimates control for observed confounders but unobserved confounders may exist.",
    ] + warnings[:2]

    return ExecutiveSummary(
        headline=f"Analysis of {target}: {len(interventions)} actionable levers identified",
        sub_headline=(
            f"Best model: {best_model_name} (R²={best_r2:.3f}). "
            f"Top lever: {top_levers[0] if top_levers else 'N/A'}."
        ),
        best_model_name=best_model_name,
        best_model_r2=round(best_r2, 4),
        top_levers=top_levers,
        bullets=bullets,
        cautions=cautions,
        methodology_note=(
            "Predictive importance uses gradient boosted tree feature importance. "
            "Causal estimates use back-door adjusted OLS regression. "
            "Intervention magnitudes are counterfactual simulations from the predictive model."
        ),
        disclaimer=(
            "LeverGuide provides analytical support for decision-making, not guaranteed outcomes. "
            "Validate key recommendations with controlled experiments before large-scale changes."
        ),
    )


@router.post("/analyze", response_model=AnalysisBundle)
async def analyze(req: AnalysisRequest) -> AnalysisBundle:
    start = time.time()
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Analysis started: target={req.target}")

    # ── Parse CSV ─────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(io.StringIO(req.dataset_csv))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"CSV parse error: {exc}")

    if req.target not in df.columns:
        raise HTTPException(
            status_code=422,
            detail=f"Target column '{req.target}' not found in dataset."
        )

    if len(df) < 30:
        raise HTTPException(status_code=422, detail="Dataset must have at least 30 rows.")

    # Cap rows for serverless safety (Render free tier ~512 MB RAM)
    if len(df) > 2_000:
        df = df.sample(2_000, random_state=req.random_seed)

    df[req.target] = _coerce_and_validate_target(df, req.target)

    # ── Assign roles ──────────────────────────────────────────────────────
    roles = _assign_roles(df, req.column_roles, req.target)

    controllable = [c for c, r in roles.items() if r == "controllable"]
    confounders   = [c for c, r in roles.items() if r == "confounder"]
    mediators     = [c for c, r in roles.items() if r == "mediator"]
    context       = [c for c, r in roles.items() if r == "context"]
    identifiers   = [c for c, r in roles.items() if r in ("identifier", "ignore")]

    if not controllable:
        raise HTTPException(
            status_code=422,
            detail="No columns assigned 'controllable' role. Assign at least one lever variable."
        )

    warnings: list[str] = []
    if len(controllable) > 20:
        warnings.append("Many controllable features — consider pruning to the most relevant.")

    # ── DAG ───────────────────────────────────────────────────────────────
    dag_edges = req.dag_edges if req.dag_edges else auto_dag(
        controllable, confounders, context, req.target
    )
    dag_validation = validate_dag(
        dag_edges, list(df.columns), req.target, controllable
    )
    if not dag_validation.valid:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_DAG",
                "message": "The submitted DAG is invalid. Fix the graph and retry.",
                "errors": dag_validation.errors,
                "warnings": dag_validation.warnings,
            },
        )
    warnings.extend(dag_validation.warnings)

    # ── Predictive pipeline ───────────────────────────────────────────────
    pred_features = [
        c for c, r in roles.items()
        if r in ("controllable", "confounder", "context") and c != req.target
    ]
    try:
        X, y, feat_names, _ = build_feature_matrix(
            df, pred_features, req.target, standardize=True
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Feature matrix error: {exc}")

    try:
        predictive_results = run_predictive_pipeline(
            X, y, feat_names,
            task=req.task,
            random_seed=req.random_seed,
            run_cv=(len(df) > 200),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Modelling error: {exc}")

    best = next(r for r in predictive_results if r.is_winner)

    # ── Causal analysis ───────────────────────────────────────────────────
    causal_effects = run_causal_analysis(
        df=df,
        target=req.target,
        controllable=controllable,
        confounders=confounders,
        mediators=mediators,
        context=context,
        dag_edges=dag_edges,
    )

    # ── Intervention engine ───────────────────────────────────────────────
    interventions = run_intervention_engine(
        df=df,
        target=req.target,
        feature_names=pred_features,
        controllable=controllable,
        causal_effects=causal_effects,
        improve_direction=req.improve_direction,
        top_n=8,
        random_seed=req.random_seed,
    )

    # ── EDA ───────────────────────────────────────────────────────────────
    eda_cols = pred_features + [req.target]
    correlations = _compute_correlations(df, eda_cols)
    distributions = _compute_distributions(df, eda_cols[:30])

    # ── Executive summary ─────────────────────────────────────────────────
    executive = _build_executive(
        req.target, best.display_name, best.metrics.r2,
        interventions, causal_effects, warnings,
    )

    runtime = round(time.time() - start, 2)
    logger.info(f"[{request_id}] Done in {runtime}s — best model {best.model} R²={best.metrics.r2:.3f}")

    bundle = AnalysisBundle(
        request_id=request_id,
        dataset_name=req.dataset_name,
        target=req.target,
        task=req.task,
        row_count=len(df),
        feature_count=len(pred_features),
        controllable_count=len(controllable),
        predictive=predictive_results,
        best_model=best.model,
        causal=causal_effects,
        interventions=interventions,
        correlations=correlations,
        distributions=distributions,
        executive=executive,
        dag_validation=dag_validation,
        warnings=warnings,
        runtime_seconds=runtime,
    )

    try:
        index_analysis_session(bundle, df, roles)
    except Exception as exc:
        logger.warning(f"[{request_id}] Copilot index build failed: {exc}")
        bundle.warnings.append("Copilot index could not be built for this analysis.")

    wandb_warning = track_analysis_run(
        request_id=request_id,
        req=req,
        df=df,
        roles=roles,
        predictive_results=predictive_results,
        causal_effects=causal_effects,
        interventions=interventions,
        correlations=correlations,
        executive=executive,
        runtime_seconds=runtime,
    )
    if wandb_warning:
        logger.warning(f"[{request_id}] {wandb_warning}")
        bundle.warnings.append(wandb_warning)

    return bundle


@router.post("/copilot/ask", response_model=CopilotAnswerResponse)
async def ask_copilot(req: CopilotAskRequest) -> CopilotAnswerResponse:
    try:
        citations = retrieve(req.analysis_id, req.question, top_k=req.max_citations)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "ANALYSIS_NOT_INDEXED",
                "message": (
                    "No copilot index exists for this analysis session. "
                    "Re-run the analysis and ask again."
                ),
            },
        )

    try:
        answer, model, used_llm = await answer_with_groq(req.question, citations)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Copilot generation failed")
        raise HTTPException(
            status_code=502,
            detail={
                "code": "COPILOT_GENERATION_FAILED",
                "message": f"Copilot generation failed: {exc}",
            },
        )

    artifact_ids = []
    for citation in citations:
        if citation.artifact_id not in artifact_ids:
            artifact_ids.append(citation.artifact_id)

    warnings = []
    if not used_llm:
        warnings.append("LLM generation was not used; response is based on retrieval status only.")

    return CopilotAnswerResponse(
        answer=answer,
        citations=citations,
        retrieved_artifact_ids=artifact_ids,
        model=model,
        used_llm=used_llm,
        warnings=warnings,
    )
