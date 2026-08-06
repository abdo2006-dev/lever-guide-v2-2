"""
Predictive what-if simulation.

What this produces is a **predictive what-if**: a gradient-boosted regressor is
fitted on the analysed rows, one column is overwritten, and the mean prediction
is compared with the baseline. It is not an identified causal effect, and the
module does not pretend otherwise.

Each candidate is then screened before it may be presented as an action:

  * feasibility — could the row it produces exist? (`feasibility.py`)
  * support     — is the proposed value inside the range the model has seen?
  * agreement   — does the adjusted effect estimate point the same way?
  * mechanism   — is the pathway this lever acts through actually modelled?

Only candidates that clear all four are ranked. The rest keep their numbers and
carry a status saying why they are not actionable, because a rejected candidate
is information, not noise.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from app.models.feasibility import check_intervention
from app.ontology.schema import DatasetOntology
from app.schemas import CausalEffect, Intervention

# Replicates for the row-resampling interval. Cheap: it resamples two existing
# prediction vectors and refits nothing.
BOOTSTRAP_REPLICATES = 400

UNCERTAINTY_ROW_BOOTSTRAP = (
    "Row-resampling interval over the simulated change. It holds the fitted "
    "model fixed, so it captures variation across production intervals but not "
    "uncertainty in the model itself; the true interval is wider."
)

# Why a lever's simulation cannot be read as an intervention estimate, keyed by
# the ontology's declared eligibility.
_ELIGIBILITY_BLOCKERS: dict[str, str] = {
    "derived_constrained": (
        "This value is determined by other columns, so it cannot be changed on "
        "its own."
    ),
    "mediated_unsupported": (
        "This lever acts through a mediator that this simulation holds fixed, so "
        "the simulated change is not a usable estimate of its effect. Mediator "
        "propagation is not implemented."
    ),
    "preliminary": (
        "This lever acts through a mediator that this simulation holds fixed. "
        "The result is preliminary, not an intervention estimate."
    ),
    "not_eligible": "This variable is not something an operator or planner sets.",
}


def _tradeoff(feature: str, direction: str) -> str:
    lookup = {
        ("increase", "injection_pressure_bar"): "Higher pressure may accelerate tooling wear.",
        ("increase", "barrel_temperature_c"):   "Elevated barrel temp risks resin degradation.",
        ("decrease", "cooling_time_s"):          "Faster cooling may reduce dimensional stability.",
        ("increase", "mold_temperature_c"):      "Higher mold temp improves surface finish but slows cycle.",
        ("increase", "cooling_time_s"):          "Longer cooling raises cycle time and reduces throughput.",
    }
    return lookup.get(
        (direction, feature),
        f"Monitor downstream effects of {_gerund(direction)} {feature}. "
        "This trade-off is not quantified.",
    )


def _gerund(direction: str) -> str:
    """'decrease' -> 'decreasing'. The naive f'{direction}ing' gave 'decreaseing'."""
    return direction[:-1] + "ing" if direction.endswith("e") else direction + "ing"


def _evidence_strength(
    causal_row: Optional[CausalEffect],
    pred_delta_magnitude: float,
    kpi_std: float,
) -> str:
    """
    Strength of the *predictive* signal, gated on the adjusted estimate.

    An estimate whose interval includes zero can never make a simulation strong,
    however large the simulated change is — that was the source study's own rule
    for excluding hold pressure from its action package.
    """
    if causal_row is None:
        return "weak"
    if not causal_row.interval_excludes_zero:
        return "weak"
    if causal_row.p_value < 0.01 and pred_delta_magnitude > 0.1 * kpi_std:
        return "strong"
    if causal_row.p_value < 0.05:
        return "moderate"
    return "weak"


def _adjustment_support(
    causal_row: Optional[CausalEffect], direction: str, improve_direction: str
) -> tuple[str, str]:
    """
    Does the adjusted estimate agree with the direction the simulation picked?

    Returns (support, human explanation). The old code awarded a "causal" badge
    on a p-value alone and never compared signs, which shipped recommendations
    that pointed the opposite way from their own estimate.
    """
    if causal_row is None:
        return "none", "No adjusted effect estimate is available for this variable."
    if not causal_row.interval_excludes_zero:
        return "inconclusive", (
            f"The adjusted estimate's 95% interval "
            f"[{causal_row.conf_int_lo:+.3f}, {causal_row.conf_int_hi:+.3f}] "
            "includes zero, so it neither supports nor contradicts this direction."
        )

    beta = causal_row.effect_per_std
    # Moving the lever up moves the outcome in the sign of beta.
    outcome_moves = "increase" if (beta > 0) == (direction == "increase") else "decrease"
    if outcome_moves == improve_direction:
        return "aligned", (
            f"The adjusted estimate (β={beta:+.3f}/SD) points the same way: "
            f"{_gerund(direction)} this lever is estimated to {improve_direction} "
            "the outcome."
        )
    return "conflicting", (
        f"The adjusted estimate (β={beta:+.3f}/SD) points the other way — it "
        f"implies {_gerund(direction)} this lever would {outcome_moves} the "
        "outcome, not "
        f"{improve_direction} it."
    )


def _bootstrap_interval(
    base_pred: np.ndarray, shifted_pred: np.ndarray, rng: np.random.Generator
) -> tuple[Optional[float], Optional[float]]:
    """
    Percentile interval for the mean simulated change, resampling rows.

    Deliberately narrow in scope: the fitted model is held fixed, so this is the
    row-to-row component of the uncertainty only. Returns (None, None) rather
    than a placeholder when there is too little data to resample.
    """
    diff = shifted_pred - base_pred
    n = len(diff)
    if n < 30:
        return None, None
    idx = rng.integers(0, n, size=(BOOTSTRAP_REPLICATES, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def run_intervention_engine(
    df: pd.DataFrame,
    target: str,
    feature_names: list[str],
    controllable: list[str],
    causal_effects: list[CausalEffect],
    improve_direction: str = "decrease",
    top_n: int = 8,
    random_seed: int = 42,
    ontology: Optional[DatasetOntology] = None,
) -> list[Intervention]:
    causal_map = {e.feature: e for e in causal_effects}
    rng = np.random.default_rng(random_seed)

    # Only numeric features can be used in the GBR counterfactual.
    numeric_sim_features = [
        f for f in feature_names
        if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
        and df[f].notna().any()
    ]
    if not numeric_sim_features:
        return []

    df_model = df[numeric_sim_features + [target]].copy()
    for col in numeric_sim_features:
        df_model[col] = pd.to_numeric(df_model[col], errors="coerce")
        df_model[col] = df_model[col].fillna(df_model[col].median())
    df_model[target] = pd.to_numeric(df_model[target], errors="coerce")
    df_model = df_model.dropna(subset=[target])

    if len(df_model) < 50:
        return []

    X = df_model[numeric_sim_features].values.astype(float)
    y = df_model[target].values.astype(float)

    gbr = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.08, max_depth=4,
        min_samples_leaf=10, subsample=0.8,
        random_state=random_seed,
    )
    gbr.fit(X, y)

    kpi_std = float(np.std(y)) or 1.0
    kpi_mean = float(np.mean(y))
    base_pred = gbr.predict(X)
    base_mean = float(np.mean(base_pred))
    feat_idx = {f: i for i, f in enumerate(numeric_sim_features)}

    # Everything the user or the ontology says is a lever, whether or not it can
    # be ranked. A variable that is blocked is reported as blocked, not omitted.
    candidates = [f for f in controllable if f in feat_idx]
    interventions: list[Intervention] = []

    for feat in candidates:
        j = feat_idx[feat]
        col_vals = df_model[feat]
        cur_mean = float(col_vals.mean())
        raw_std = float(col_vals.std())
        is_constant = raw_std == 0.0
        cur_std = raw_std or 1.0
        p10 = float(col_vals.quantile(0.1))
        p90 = float(col_vals.quantile(0.9))

        spec = ontology.spec(feat) if ontology else None
        # Support bounds come from declared physics where we have them, and from
        # the observed range otherwise. The previous p90*1.5 / p10*0.5 heuristic
        # was arbitrary and produced nonsense for negative-valued columns.
        obs_lo, obs_hi = float(col_vals.min()), float(col_vals.max())
        if spec and spec.valid_range:
            lo_bound = max(obs_lo, spec.valid_range[0])
            hi_bound = min(obs_hi, spec.valid_range[1])
        else:
            lo_bound, hi_bound = obs_lo, obs_hi
        if lo_bound > hi_bound:  # declared and observed do not overlap
            lo_bound, hi_bound = obs_lo, obs_hi

        # Simulate both directions unconditionally. A lever that never improves
        # the KPI in either direction is not discarded here — every configured
        # lever gets an explicit status record below, never silence.
        tried: dict[str, tuple[float, float, np.ndarray]] = {}
        for sign, direction in [(+1, "increase"), (-1, "decrease")]:
            target_val = float(np.clip(cur_mean + sign * cur_std, lo_bound, hi_bound))
            shifted = X.copy()
            shifted[:, j] = target_val
            shifted_pred = gbr.predict(shifted)
            sim_impact = float(np.mean(shifted_pred)) - base_mean
            tried[direction] = (target_val, sim_impact, shifted_pred)

        improving = {
            d: v for d, v in tried.items()
            if (improve_direction == "decrease" and v[1] < 0)
            or (improve_direction == "increase" and v[1] > 0)
        }
        no_improving_direction = not improving
        if improving:
            direction = max(improving, key=lambda d: abs(improving[d][1]))
        else:
            # Neither direction moves the KPI the way the user wants. Report it
            # anyway, using the direction whose predicted change is smallest in
            # magnitude as the representative one — the status below makes
            # clear that no action is actually offered.
            direction = min(tried, key=lambda d: abs(tried[d][1]))
        suggested, sim_impact, shifted_pred = tried[direction]

        causal_row = causal_map.get(feat)
        support, support_note = _adjustment_support(causal_row, direction, improve_direction)
        report = check_intervention(
            feature=feat, value=suggested, df=df,
            ontology=ontology, controllable=controllable,
        )

        # ── Status. Order matters: report the most fundamental blocker. ───────
        status: str
        reason: str
        if report.verdict == "not_eligible":
            status, reason = "unsupported", report.reason
        elif report.verdict == "infeasible":
            status, reason = "infeasible", report.reason
        elif spec is not None and spec.intervention_eligibility in _ELIGIBILITY_BLOCKERS:
            blocker = _ELIGIBILITY_BLOCKERS[spec.intervention_eligibility]
            if spec.intervention_eligibility == "derived_constrained":
                status = "infeasible"
            else:
                status = "exploratory"
            reason = blocker
        elif is_constant:
            status = "unsupported"
            reason = (
                f"'{feat}' has no observed variation in the analysed data "
                "(a single distinct value), so no change can be simulated for "
                "it."
            )
        elif spec is not None and spec.evidence_status == "conflicting":
            status = "conflicting_evidence"
            reason = (
                "Evidence for this lever is specification-dependent: changing the "
                "adjustment set changes the conclusion, so no direction is "
                "asserted. " + support_note
            )
        elif support == "conflicting":
            status, reason = "conflicting_evidence", support_note
        elif report.verdict == "unsupported":
            status, reason = "unsupported", report.reason
        elif causal_row is None:
            status = "exploratory"
            reason = (
                "No adjusted effect estimate is available for this variable, so "
                "this is a predictive what-if only."
            )
        elif support == "inconclusive":
            status = "exploratory"
            reason = support_note
        elif no_improving_direction:
            status = "unsupported"
            reason = (
                f"Neither increasing nor decreasing {feat} was estimated to "
                f"{improve_direction} {target} (increase "
                f"Δ={tried['increase'][1]:+.4g}, decrease "
                f"Δ={tried['decrease'][1]:+.4g}). No action is offered."
            )
        else:
            status = "eligible"
            reason = "Feasible, inside observed support, and consistent with the adjusted estimate."

        # ── Uncertainty ──────────────────────────────────────────────────────
        ci_lo, ci_hi = _bootstrap_interval(base_pred, shifted_pred, rng)
        if ci_lo is None:
            interval_method = None
            uncertainty_status = (
                "not_computed: too few rows to resample an interval"
            )
        else:
            interval_method = "row_bootstrap_fixed_model"
            uncertainty_status = "row_bootstrap_fixed_model"

        exp_kpi_pct = (sim_impact / abs(kpi_mean) * 100) if kpi_mean != 0 else 0.0

        # Belt and braces: a GBR prediction on finite inputs cannot produce a
        # non-finite value in practice, but never let one reach the API
        # response if it somehow did. The record is not dropped — that would
        # reintroduce silent disappearance — its simulated numbers are reset to
        # the no-change baseline (which is finite by construction) and the
        # status/reason make clear no simulation result is actually being
        # asserted.
        if not all(math.isfinite(v) for v in (suggested, sim_impact, exp_kpi_pct, cur_mean)):
            status = "unsupported"
            reason = (
                f"The simulation for '{feat}' produced a non-finite value and "
                "was withheld rather than reported."
            )
            suggested, sim_impact, exp_kpi_pct = cur_mean, 0.0, 0.0
            ci_lo = ci_hi = None
            interval_method = None
            uncertainty_status = "not_computed: simulation produced a non-finite value"

        rationale = (
            f"Simulation: setting {feat} to {suggested:.4g} "
            f"({'+' if suggested >= cur_mean else ''}{suggested - cur_mean:.4g} "
            f"from its mean) changes predicted {target} by {sim_impact:+.4f} "
            f"({exp_kpi_pct:+.1f}%). {support_note}"
        )

        assumptions = [
            "Each row keeps its own observed values for every other column; only "
            "this variable is changed, and the change is averaged across rows.",
            "The simulation model was fitted on these same rows, so its "
            "magnitudes are optimistic relative to new data.",
            "Columns physically coupled to this one are not updated, beyond the "
            "coupling constraints checked above.",
            "The training-data distribution is representative of future conditions.",
        ]
        if status != "eligible":
            assumptions.append(f"Not offered as an action: {reason}")

        interventions.append(Intervention(
            rank=0,
            feature=feat,
            direction=direction,
            current_mean=cur_mean,
            current_p10=p10,
            current_p90=p90,
            suggested_value=suggested,
            delta=suggested - cur_mean,
            delta_pct=((suggested - cur_mean) / abs(cur_mean) * 100) if cur_mean != 0 else 0.0,
            expected_kpi_change=sim_impact,
            expected_kpi_change_pct=exp_kpi_pct,
            expected_kpi_change_lo=ci_lo,
            expected_kpi_change_hi=ci_hi,
            interval_method=interval_method,  # type: ignore[arg-type]
            uncertainty_status=uncertainty_status,
            status=status,  # type: ignore[arg-type]
            status_reason=reason,
            support_status=report.support_status,  # type: ignore[arg-type]
            feasibility_checks=report.checks,
            evidence_strength=_evidence_strength(causal_row, abs(sim_impact), kpi_std),
            adjustment_support=support,  # type: ignore[arg-type]
            tradeoff=_tradeoff(feat, direction),
            rationale=rationale,
            assumptions=assumptions,
            caveat=(
                "This is a predictive what-if, not an identified causal effect. "
                "Validate with a controlled test before operational use."
            ),
        ))

    # Only eligible candidates are ranked. Everything else keeps rank 0 and is
    # returned for the diagnostics section rather than being discarded.
    eligible = [iv for iv in interventions if iv.status == "eligible"]
    others = [iv for iv in interventions if iv.status != "eligible"]
    eligible.sort(key=lambda x: abs(x.expected_kpi_change), reverse=True)
    for i, iv in enumerate(eligible[:top_n], 1):
        iv.rank = i
    others.sort(key=lambda x: abs(x.expected_kpi_change), reverse=True)
    return eligible[:top_n] + others
