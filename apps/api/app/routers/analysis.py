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
    AnalysisBundle, AnalysisProvenance, AnalysisRequest, ConfigurationProblem,
    CorrelationPair, DistributionBucket, ExcludedColumn, FeatureDistribution,
    ExecutiveSummary, TopValue, CopilotAskRequest, CopilotAnswerResponse,
)
from app.ontology import (
    INJECTION_MOULDING_ONTOLOGY, SOURCE_DEVIATIONS, resolve_ontology,
)
from app.ontology.schema import DatasetOntology
from app.utils.preprocess import build_feature_matrix, build_column_meta, infer_column_kind
from app.schemas import DagEdge
from app.utils.dag import auto_dag, validate_dag
from app.models.pipeline import run_predictive_pipeline
from app.models.causal import run_causal_analysis
from app.models.intervention import run_intervention_engine
from app.rag import answer_with_groq, index_analysis_session, retrieve
from app.utils.wandb_tracking import track_analysis_run

router = APIRouter()
logger = logging.getLogger(__name__)


# The demo's role assignments now come from the declared ontology, which is also
# what generates the frontend's copy. This alias exists so the historical import
# path keeps working; the ontology is the source of truth.
DEMO_ROLES: dict[str, str] = INJECTION_MOULDING_ONTOLOGY.column_roles()

# Roles that make a column part of the causal argument, as opposed to merely a
# predictor. `unassigned` is deliberately not here.
_ADJUSTER_ROLES = ("confounder", "context")
# Roles that make a column an estimation and intervention target.
_LEVER_ROLES = ("controllable", "planning_lever")
# Roles whose columns may be fed to the predictive model.
#
# Mediators are included here and excluded from adjustment sets, and that
# asymmetry is deliberate: the predictive model's job is to approximate the world
# for simulation, so blocking a real pathway would make it a worse approximator —
# whereas conditioning on a mediator when estimating a total effect turns it into
# a direct effect. It does mean a simulation holds mediators at their observed
# values, which is why levers acting through one are reported as unsupported
# rather than as intervention estimates.
_PREDICTOR_ROLES = (
    "controllable", "planning_lever", "confounder", "context", "mediator",
    "unassigned",
)


def _assign_roles(df: pd.DataFrame, column_roles: dict[str, str], target: str) -> dict[str, str]:
    """
    Resolve a role for every column.

    A column the user did not label is `unassigned`, not `confounder`. Being
    numeric is not evidence that a column confounds anything, and the previous
    default silently put every unlabelled column into every adjustment set.
    """
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
                roles[col] = "unassigned"
    return roles


def _validate_configuration(
    df: pd.DataFrame, roles: dict[str, str], analysis_mode: str, target: str
) -> list[ConfigurationProblem]:
    """
    Independent server-side validation of the analysis configuration.

    The client performs the same checks before submitting, but the API does not
    trust that: this is the authoritative gate.
    """
    problems: list[ConfigurationProblem] = []
    levers = [c for c, r in roles.items() if r in _LEVER_ROLES]
    adjusters = [c for c, r in roles.items() if r in _ADJUSTER_ROLES]
    unassigned = [c for c, r in roles.items() if r == "unassigned"]

    if analysis_mode == "causal" and not levers:
        problems.append(ConfigurationProblem(
            code="NO_TREATMENT_SELECTED",
            message=(
                "Causal estimation needs at least one treatment: no column is "
                "labelled 'controllable' or 'planning_lever'."
            ),
            remedy=(
                "Label the columns you can actually change as 'controllable' "
                "(a setpoint) or 'planning_lever' (a scheduling decision) — or "
                "switch to descriptive/predictive mode, which makes no causal "
                "claims and needs no treatment."
            ),
            columns=unassigned[:20],
        ))

    if analysis_mode == "causal" and levers and not adjusters:
        problems.append(ConfigurationProblem(
            code="NO_ADJUSTERS_SELECTED",
            message=(
                "No column is labelled 'confounder' or 'context', so every "
                "effect estimate would be unadjusted."
            ),
            remedy=(
                "Label the variables that plausibly cause both your treatment "
                "and your outcome as 'confounder'. If there genuinely are none, "
                "the estimates are associations and should be read as such."
            ),
            columns=unassigned[:20],
        ))

    if analysis_mode == "causal" and levers:
        constant_levers = sorted(
            c for c in levers
            if c in df.columns and df[c].dropna().nunique() < 2
        )
        if len(constant_levers) == len(levers):
            problems.append(ConfigurationProblem(
                code="ALL_TREATMENTS_CONSTANT",
                message=(
                    "Every labelled treatment has a single distinct value in "
                    "this data, so no intervention effect can be estimated for "
                    "any of them."
                ),
                remedy=(
                    "Choose a treatment column that varies, or check the data "
                    "for a filtering bug — or switch to descriptive/predictive "
                    "mode, which makes no causal claims and needs no treatment."
                ),
                columns=constant_levers,
            ))

    return problems


def _resolve_graph(
    req: AnalysisRequest,
    df: pd.DataFrame,
    ontology: DatasetOntology | None,
    controllable: list[str],
    confounders: list[str],
    context: list[str],
) -> tuple[list[DagEdge], str, str | None]:
    """Pick the causal graph, and say where it came from."""
    if req.dag_edges:
        return req.dag_edges, "user_supplied", (
            "This graph was supplied with the request. It is an assumption, not "
            "a discovered causal structure."
        )
    if ontology is not None:
        cols = set(df.columns)
        edges = [
            DagEdge(source=s, target=t)
            for s, t in ontology.edges
            if s in cols and t in cols
        ]
        return edges, "declared_domain_ontology", ontology.graph_assumption
    return (
        auto_dag(controllable, confounders, context, req.target),
        "assumed_from_roles",
        (
            "This graph was generated from the roles you assigned: every "
            "confounder points at every lever and at the outcome. It is a "
            "template, not a domain model, and it cannot represent one lever "
            "responding to another."
        ),
    )


def _coerce_and_validate_target(df: pd.DataFrame, target: str) -> pd.Series:
    target_numeric = pd.to_numeric(df[target], errors="coerce")
    non_null = target_numeric.dropna()
    if len(non_null) < 30:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "TARGET_INSUFFICIENT_ROWS",
                "message": (
                    f"Target column '{target}' must contain at least 30 numeric, "
                    "non-missing rows for regression analysis."
                ),
                "problems": [ConfigurationProblem(
                    code="TARGET_INSUFFICIENT_ROWS",
                    message=(
                        f"'{target}' has fewer than 30 numeric, non-missing values."
                    ),
                    remedy="Choose a target column with at least 30 valid numeric rows.",
                    columns=[target],
                ).model_dump()],
            },
        )
    if non_null.nunique() < 2:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "CONSTANT_OUTCOME",
                "message": (
                    f"Target column '{target}' has a single distinct value. An "
                    "outcome with no variation cannot be modelled or used to "
                    "estimate an effect."
                ),
                "problems": [ConfigurationProblem(
                    code="CONSTANT_OUTCOME",
                    message=f"'{target}' does not vary across the analysed rows.",
                    remedy="Choose a target column that varies, or check the data for a filtering bug.",
                    columns=[target],
                ).model_dump()],
            },
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
    analysis_mode: str = "causal",
) -> ExecutiveSummary:
    eligible = [iv for iv in interventions if iv.status == "eligible"]
    demoted = [iv for iv in interventions if iv.status != "eligible"]
    top_levers = [iv.feature for iv in eligible[:3]]

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
            f"Model fit is modest (R²={best_r2:.2f}). Interpret simulated changes conservatively."
        )

    for iv in eligible[:3]:
        direction_word = "reducing" if iv.direction == "decrease" else "increasing"
        interval = ""
        if iv.expected_kpi_change_lo is not None and iv.expected_kpi_change_hi is not None:
            interval = (
                f" (row-resampling 95% interval "
                f"{iv.expected_kpi_change_lo:+.3f} to {iv.expected_kpi_change_hi:+.3f})"
            )
        bullets.append(
            f"Simulation suggests {direction_word} {iv.feature} could "
            f"{'reduce' if iv.expected_kpi_change < 0 else 'increase'} {target} "
            f"by ~{abs(iv.expected_kpi_change_pct):.1f}%{interval}. "
            f"The adjusted estimate is {iv.adjustment_support} with this direction. "
            "This is a predictive what-if, not a measured intervention effect."
        )

    if not eligible and demoted:
        bullets.append(
            f"No candidate cleared feasibility, support and evidence-agreement "
            f"checks. {len(demoted)} candidate(s) were assessed and are listed "
            "with the reason each was set aside."
        )

    for iv in demoted[:3]:
        bullets.append(f"Set aside — {iv.feature} ({iv.status}): {iv.status_reason}")

    cautions = [
        "All estimates come from observational data, not randomised experiments.",
        "Adjusted effect estimates depend on the selected causal graph and on there "
        "being no important unmeasured confounding.",
        "Simulated changes modify model inputs and compare predictions; they are "
        "not automatically causal intervention estimates.",
    ] + warnings[:2]

    if analysis_mode == "descriptive_predictive":
        headline = f"Descriptive and predictive analysis of {target}"
        sub_headline = (
            f"Best model: {best_model_name} (R²={best_r2:.3f}). No causal "
            "question was specified, so no effect estimates or candidate actions "
            "were produced."
        )
    else:
        headline = (
            f"Analysis of {target}: {len(eligible)} candidate lever(s) passed "
            f"screening, {len(demoted)} set aside"
        )
        sub_headline = (
            f"Best model: {best_model_name} (R²={best_r2:.3f}). "
            f"Top candidate: {top_levers[0] if top_levers else 'none'}."
        )

    return ExecutiveSummary(
        headline=headline,
        sub_headline=sub_headline,
        best_model_name=best_model_name,
        best_model_r2=round(best_r2, 4),
        top_levers=top_levers,
        bullets=bullets,
        cautions=cautions,
        methodology_note=(
            "Three different kinds of result appear in this analysis and they are "
            "not interchangeable. (1) Associations: marginal correlations, "
            "unadjusted, no causal claim. (2) Adjusted observational effect "
            "estimates: OLS with a declared or graph-derived adjustment set; "
            "interpretation depends on the selected causal graph and assumptions, "
            "including no important unmeasured confounding. Intervals are "
            "textbook OLS confidence intervals under homoskedasticity, which are "
            "too narrow for clustered panel data. (3) Predictive what-if "
            "simulations: a gradient-boosted model fitted on these same rows, one "
            "input changed, predictions compared — this is not automatically a "
            "causal intervention estimate, and in-sample magnitudes are "
            "optimistic. Feature importance is predictive gain, not a causal "
            "ranking."
        ),
        disclaimer=(
            "LeverGuide provides analytical support for decision-making, not "
            "outcome guarantees. Validate any candidate change with a controlled "
            "test before large-scale rollout."
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

    n_rows_supplied = len(df)
    sampling_note: str | None = None
    # Cap rows for serverless safety (Render free tier ~512 MB RAM)
    if len(df) > 2_000:
        df = df.sample(2_000, random_state=req.random_seed)
        sampling_note = (
            f"{n_rows_supplied:,} rows supplied; a simple random sample of 2,000 "
            "was analysed. The sample is not stratified by machine or time, so it "
            "does not preserve the panel structure."
        )

    df[req.target] = _coerce_and_validate_target(df, req.target)

    # ── Ontology ──────────────────────────────────────────────────────────
    # A curated ontology supplies declared roles, adjustment sets, physical
    # bounds and coupling constraints. It applies only on a close column match;
    # any other dataset takes the generic path and gets no domain claims.
    ontology = resolve_ontology(list(df.columns), req.target)

    # ── Assign roles ──────────────────────────────────────────────────────
    roles = _assign_roles(df, req.column_roles, req.target)

    controllable = [c for c, r in roles.items() if r == "controllable"]
    planning     = [c for c, r in roles.items() if r == "planning_lever"]
    confounders  = [c for c, r in roles.items() if r == "confounder"]
    mediators    = [c for c, r in roles.items() if r == "mediator"]
    context      = [c for c, r in roles.items() if r == "context"]
    unassigned   = [c for c, r in roles.items() if r == "unassigned"]
    levers = controllable + planning

    problems = _validate_configuration(df, roles, req.analysis_mode, req.target)
    if problems:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_ANALYSIS_CONFIGURATION",
                "message": (
                    "The analysis configuration is incomplete. "
                    + " ".join(p.message for p in problems)
                ),
                "problems": [p.model_dump() for p in problems],
            },
        )

    warnings: list[str] = []
    if len(levers) > 20:
        warnings.append("Many lever columns — consider pruning to the most relevant.")
    if unassigned:
        warnings.append(
            f"{len(unassigned)} column(s) have no causal role and were used as "
            "predictors only, never as adjusters: "
            + ", ".join(sorted(unassigned)[:8])
            + ("…" if len(unassigned) > 8 else "")
        )
    if mediators:
        warnings.append(
            "Mediators are predictors of the model but are excluded from every "
            "adjustment set: "
            + ", ".join(sorted(mediators))
            + ". Simulations hold them at their observed values, so a lever whose "
            "effect runs through one is reported as unsupported rather than as an "
            "intervention estimate."
        )

    # ── Causal graph ──────────────────────────────────────────────────────
    dag_edges, dag_source, graph_assumption = _resolve_graph(
        req, df, ontology, controllable, confounders, context
    )
    dag_validation = validate_dag(
        dag_edges, list(df.columns), req.target, levers,
        dag_source=dag_source, graph_assumption=graph_assumption,
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
        if r in _PREDICTOR_ROLES and c != req.target
    ]
    try:
        X, y, feat_names, _ = build_feature_matrix(
            df, pred_features, req.target, standardize=True
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Feature matrix error: {exc}")

    try:
        predictive_results, model_statuses = run_predictive_pipeline(
            X, y, feat_names,
            task=req.task,
            random_seed=req.random_seed,
            run_cv=(len(df) > 200),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Modelling error: {exc}")

    best = next(r for r in predictive_results if r.is_winner)

    for status in model_statuses:
        if status.status == "unavailable_dependency":
            warnings.append(
                f"{status.display_name} did not run: its library is unavailable "
                f"in this environment ({status.detail}). Results below compare "
                f"{len(predictive_results)} model(s), not "
                f"{len(model_statuses)}."
            )
        elif status.status == "training_failed":
            warnings.append(
                f"{status.display_name} failed during training and is not "
                f"included in the comparison ({status.detail})."
            )

    # ── Adjusted effect estimates and what-if simulations ─────────────────
    causal_effects: list[Any] = []
    interventions: list[Any] = []
    excluded_columns: list[ExcludedColumn] = []

    if req.analysis_mode == "causal":
        declared_sets: dict[str, list[str]] | None = None
        causal_roles: dict[str, str] | None = None
        set_notes: dict[str, str] | None = None
        if ontology is not None:
            declared_sets = {
                lever: [c for c in adj if c in df.columns]
                for lever, adj in ontology.adjustment_sets.items()
                if lever in levers
            }
            causal_roles = {v.name: v.causal_role for v in ontology.variables}
            set_notes = dict(SOURCE_DEVIATIONS)

        causal_effects, excluded_columns = run_causal_analysis(
            df=df,
            target=req.target,
            controllable=levers,
            confounders=confounders,
            mediators=mediators,
            context=context,
            dag_edges=dag_edges,
            declared_adjustment_sets=declared_sets,
            causal_roles=causal_roles,
            set_notes=set_notes,
        )
        if excluded_columns:
            warnings.append(
                f"{len(excluded_columns)} column(s) were excluded from a fitted "
                "adjustment set or rejected as a treatment because they have no "
                "observed variation, or produced a non-finite estimate — see "
                "the provenance panel for the column and the reason on each."
            )

        interventions = run_intervention_engine(
            df=df,
            target=req.target,
            feature_names=pred_features,
            controllable=levers,
            causal_effects=causal_effects,
            improve_direction=req.improve_direction,
            top_n=8,
            random_seed=req.random_seed,
            ontology=ontology,
        )

        demoted = [iv for iv in interventions if iv.status != "eligible"]
        if demoted:
            warnings.append(
                f"{len(demoted)} candidate change(s) were assessed but are not "
                "offered as actions — see the diagnostics section for the reason "
                "on each."
            )

    # ── EDA ───────────────────────────────────────────────────────────────
    eda_cols = pred_features + [req.target]
    correlations = _compute_correlations(df, eda_cols)
    distributions = _compute_distributions(df, eda_cols[:30])

    # ── Executive summary ─────────────────────────────────────────────────
    executive = _build_executive(
        req.target, best.display_name, best.metrics.r2,
        interventions, causal_effects, warnings, req.analysis_mode,
    )

    runtime = round(time.time() - start, 2)
    logger.info(f"[{request_id}] Done in {runtime}s — best model {best.model} R²={best.metrics.r2:.3f}")

    provenance = AnalysisProvenance(
        analysis_mode=req.analysis_mode,
        ontology_id=ontology.dataset_id if ontology else None,
        ontology_version=ontology.version if ontology else None,
        graph_assumption=graph_assumption,
        dag_source=dag_source,
        adjustment_set_source=(
            "declared_domain_dag" if ontology is not None else "derived_from_graph"
        ),
        effect_estimator=(
            "ordinary least squares on standardised columns; no fixed effects, "
            "i.i.d. standard errors"
        ),
        effect_interval_method=(
            "ols_analytic_homoskedastic" if causal_effects else None
        ),
        simulation_model=(
            "gradient_boosting_regressor" if interventions else None
        ),
        simulation_evaluation=(
            "fitted and evaluated on the same rows (in-sample)"
            if interventions else None
        ),
        simulation_interval_method=(
            "row_bootstrap_fixed_model" if interventions else None
        ),
        n_rows_supplied=n_rows_supplied,
        n_rows_analysed=len(df),
        sampling_note=sampling_note,
        train_eval_strategy=(
            f"random {100 - round(best.metrics.n_test / max(best.metrics.n_train + best.metrics.n_test, 1) * 100)}"
            f"/{round(best.metrics.n_test / max(best.metrics.n_train + best.metrics.n_test, 1) * 100)} "
            f"train/test split ({best.metrics.n_train} train, {best.metrics.n_test} test), "
            "3-fold CV; not grouped by machine and not time-ordered"
        ),
        random_seed=req.random_seed,
        column_roles=dict(roles),
        excluded_columns=excluded_columns,
    )

    bundle = AnalysisBundle(
        request_id=request_id,
        dataset_name=req.dataset_name,
        target=req.target,
        task=req.task,
        analysis_mode=req.analysis_mode,
        row_count=len(df),
        feature_count=len(pred_features),
        controllable_count=len(levers),
        predictive=predictive_results,
        model_statuses=model_statuses,
        best_model=best.model,
        causal=causal_effects,
        interventions=interventions,
        correlations=correlations,
        distributions=distributions,
        executive=executive,
        dag_validation=dag_validation,
        provenance=provenance,
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
