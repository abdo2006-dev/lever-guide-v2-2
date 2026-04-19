"""
DAG-aware causal analysis.

For each controllable variable, estimates the adjusted effect on the target
using back-door adjusted OLS (statsmodels). Adjustment set is derived from
the user-supplied DAG via NetworkX.

This is NOT full structural causal modelling. These are adjusted coefficients
from observational data — unobserved confounders may bias estimates.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from app.schemas import CausalEffect, DagEdge
from app.utils.dag import adjustment_set, build_dag


def _evidence_strength(p_value: float, n: int, n_adj: int) -> str:
    if n < 100 or n_adj > n * 0.5:
        return "insufficient"
    if p_value < 0.01:
        return "strong"
    if p_value < 0.05:
        return "moderate"
    if p_value < 0.15:
        return "weak"
    return "insufficient"


def run_causal_analysis(
    df: pd.DataFrame,
    target: str,
    controllable: list[str],
    confounders: list[str],
    mediators: list[str],
    context: list[str],
    dag_edges: list[DagEdge],
) -> list[CausalEffect]:
    """
    For each controllable variable fit:
        y ~ cause + adjustment_set
    and report the adjusted coefficient on 'cause'.
    All numeric features are standardised so β is interpretable as
    "effect of +1 SD change on target (also standardised)".
    """
    G = build_dag(dag_edges)
    effects: list[CausalEffect] = []

    # Standardise numeric columns once
    df_std = df.copy()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    std_map:  dict[str, float] = {}
    mean_map: dict[str, float] = {}
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) > 1 and s.std() > 0:
            mean_map[col] = float(s.mean())
            std_map[col]  = float(s.std())
            df_std[col]   = (df[col] - mean_map[col]) / std_map[col]

    for cause in controllable:
        if cause not in df.columns:
            continue
        # Only handle numeric causes (can't shift a categorical by 1 SD)
        if not pd.api.types.is_numeric_dtype(df[cause]):
            continue

        adj = adjustment_set(
            cause=cause, outcome=target, G=G,
            confounders=confounders, mediators=mediators, context=context,
        )
        adj = {c for c in adj if c in df.columns}

        reg_cols = [cause] + sorted(adj)
        reg_df   = df_std[[target] + reg_cols].dropna()
        n = len(reg_df)
        if n < 30:
            continue

        y = reg_df[target].to_numpy(dtype=float)
        X_raw = reg_df[reg_cols]

        # Encode categoricals; ensure all columns are float
        X_enc = pd.get_dummies(X_raw, drop_first=True).astype(float)
        X_c   = sm.add_constant(X_enc, has_constant="add")

        try:
            fit = sm.OLS(y, X_c).fit()
        except Exception:
            continue

        if cause not in fit.params.index:
            continue

        beta  = float(fit.params[cause])
        se    = float(fit.bse[cause])
        t     = float(fit.tvalues[cause])
        p     = float(fit.pvalues[cause])
        ci    = fit.conf_int()
        ci_lo = float(ci.loc[cause, 0])
        ci_hi = float(ci.loc[cause, 1])

        # Unstandardised effect (original KPI units per 1-unit change)
        sd_cause  = std_map.get(cause,  1.0)
        sd_target = std_map.get(target, 1.0)
        effect_raw = beta * (sd_target / sd_cause) if sd_cause > 0 else beta

        strength = _evidence_strength(p, n, len(adj))
        warning: str | None = None
        if n < 100:
            warning = f"Small sample (n={n}) — treat this estimate cautiously."
        elif p >= 0.15:
            warning = "Effect is not statistically significant at α=0.15."

        effects.append(CausalEffect(
            feature=cause,
            effect_per_std=beta,
            effect_raw=effect_raw,
            std_err=se,
            t_stat=t,
            p_value=p,
            conf_int_lo=ci_lo,
            conf_int_hi=ci_hi,
            adjusted_for=sorted(adj),
            controllable=True,
            evidence_strength=strength,
            warning=warning,
        ))

    return sorted(effects, key=lambda e: abs(e.t_stat), reverse=True)
