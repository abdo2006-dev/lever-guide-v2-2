"""
Intervention recommendation engine.

Combines two evidence sources:
  1. Causal: adjusted OLS effect (direction + magnitude, with caveats)
  2. Predictive: GBR counterfactual simulation (what-if) on numeric features only
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from app.schemas import CausalEffect, Intervention


def _evidence_type(has_causal: bool, causal_sig: bool) -> str:
    if has_causal and causal_sig:
        return "causal"
    if has_causal:
        return "mixed"
    return "predictive"


def _evidence_strength(causal_p: float | None, pred_delta_magnitude: float, kpi_std: float) -> str:
    if causal_p is not None and causal_p < 0.01 and pred_delta_magnitude > 0.1 * kpi_std:
        return "strong"
    if causal_p is not None and causal_p < 0.05:
        return "moderate"
    return "weak"


def _tradeoff(feature: str, direction: str) -> str:
    lookup = {
        ("increase", "injection_pressure_bar"): "Higher pressure may accelerate tooling wear.",
        ("increase", "barrel_temperature_c"):   "Elevated barrel temp risks resin degradation.",
        ("decrease", "cooling_time_s"):          "Faster cooling may reduce dimensional stability.",
        ("increase", "mold_temperature_c"):      "Higher mold temp improves surface finish but slows cycle.",
        ("decrease", "cycle_time_s"):            "Shorter cycles may increase defect rate on complex parts.",
    }
    return lookup.get(
        (direction, feature),
        f"Monitor downstream effects of {direction}ing {feature}.",
    )


def run_intervention_engine(
    df: pd.DataFrame,
    target: str,
    feature_names: list[str],
    controllable: list[str],
    causal_effects: list[CausalEffect],
    improve_direction: str = "decrease",
    top_n: int = 8,
    random_seed: int = 42,
) -> list[Intervention]:
    causal_map = {e.feature: e for e in causal_effects}

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

    kpi_std  = float(np.std(y)) or 1.0
    kpi_mean = float(np.mean(y))
    base_mean = float(np.mean(gbr.predict(X)))
    feat_idx  = {f: i for i, f in enumerate(numeric_sim_features)}

    numeric_controllable = [f for f in controllable if f in feat_idx]
    interventions: list[Intervention] = []

    for feat in numeric_controllable:
        j = feat_idx[feat]
        col_vals = df_model[feat]
        cur_mean = float(col_vals.mean())
        cur_std  = float(col_vals.std()) or 1.0
        p10 = float(col_vals.quantile(0.1))
        p90 = float(col_vals.quantile(0.9))

        best_delta, best_direction, best_sim_impact, best_suggested = None, None, None, None

        for sign, direction in [(+1, "increase"), (-1, "decrease")]:
            lo_bound = p10 - abs(p10) * 0.5 if p10 <= 0 else p10 * 0.5
            target_val = float(np.clip(cur_mean + sign * cur_std, lo_bound, p90 * 1.5))
            X_shifted       = X.copy()
            X_shifted[:, j] = target_val
            sim_impact      = float(np.mean(gbr.predict(X_shifted))) - base_mean

            improves = (
                (improve_direction == "decrease" and sim_impact < 0)
                or (improve_direction == "increase" and sim_impact > 0)
            )
            if improves and (best_sim_impact is None or abs(sim_impact) > abs(best_sim_impact)):
                best_delta, best_direction = target_val - cur_mean, direction
                best_sim_impact, best_suggested = sim_impact, target_val

        if best_direction is None or best_sim_impact is None or best_suggested is None:
            continue

        causal_row = causal_map.get(feat)
        has_causal = causal_row is not None
        causal_sig = has_causal and causal_row.p_value < 0.05  # type: ignore[union-attr]
        causal_p   = causal_row.p_value if has_causal else None

        ev_type     = _evidence_type(has_causal, causal_sig)
        ev_str      = _evidence_strength(causal_p, abs(best_sim_impact), kpi_std)
        exp_kpi     = best_sim_impact
        exp_kpi_pct = (exp_kpi / abs(kpi_mean) * 100) if kpi_mean != 0 else 0.0

        causal_note = ""
        if has_causal:
            sign_word = "positively" if causal_row.effect_per_std > 0 else "negatively"  # type: ignore[union-attr]
            causal_note = (
                f" Causal analysis (p={causal_row.p_value:.3f}) indicates "  # type: ignore[union-attr]
                f"this feature {sign_word} affects {target} "
                f"(β={causal_row.effect_per_std:+.3f}/SD)."  # type: ignore[union-attr]
            )

        rationale = (
            f"Shifting {feat} {best_direction} by ~1σ is predicted to "
            f"{'reduce' if improve_direction == 'decrease' else 'increase'} "
            f"{target} by {abs(exp_kpi):.3f} ({abs(exp_kpi_pct):.1f}%).{causal_note}"
        )

        assumptions = [
            "Other variables remain at their current mean values.",
            "The training-data distribution is representative of future conditions.",
        ]
        if not causal_sig:
            assumptions.append("This recommendation is primarily predictive — not confirmed causal.")

        interventions.append(Intervention(
            rank=0,
            feature=feat,
            direction=best_direction,
            current_mean=cur_mean,
            current_p10=p10,
            current_p90=p90,
            suggested_value=best_suggested,
            delta=best_delta,
            delta_pct=(best_delta / abs(cur_mean) * 100) if cur_mean != 0 else 0.0,
            expected_kpi_change=exp_kpi,
            expected_kpi_change_pct=exp_kpi_pct,
            evidence_strength=ev_str,
            evidence_type=ev_type,
            tradeoff=_tradeoff(feat, best_direction),
            rationale=rationale,
            assumptions=assumptions,
            caveat=(
                "This estimate is based on observational data. "
                "Validate with a controlled experiment before operational use."
            ),
        ))

    interventions.sort(key=lambda x: abs(x.expected_kpi_change), reverse=True)
    for i, iv in enumerate(interventions[:top_n], 1):
        iv.rank = i
    return interventions[:top_n]
