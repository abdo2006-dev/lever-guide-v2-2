"""
Adjusted observational effect estimation.

For each lever, estimates the effect on the target using OLS with a back-door
adjustment set. Where a curated ontology declares a per-lever adjustment set, it
is used verbatim; otherwise the set is derived from the graph.

These are **adjusted observational effect estimates**, not proven causes. They
are valid only if the assumed graph is correct and there is no important
unmeasured confounding. This is not full structural causal modelling, and no
identification search is performed beyond the declared or derived set.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from app.schemas import CausalEffect, DagEdge
from app.utils.dag import adjustment_set, build_dag


def _evidence_strength(
    p_value: float, n: int, n_adj: int, interval_excludes_zero: bool
) -> str:
    """
    Strength of the *estimate*, not of a causal claim.

    An interval that includes zero caps the rating at "weak" regardless of the
    p-value. Rating on p alone labelled six of eight demo levers "strong" at
    n = 2,000, including one whose interval crossed zero.
    """
    if n < 100 or n_adj > n * 0.5:
        return "insufficient"
    if not interval_excludes_zero:
        return "weak" if p_value < 0.15 else "insufficient"
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
    declared_adjustment_sets: Optional[dict[str, list[str]]] = None,
    causal_roles: Optional[dict[str, str]] = None,
    set_notes: Optional[dict[str, str]] = None,
) -> list[CausalEffect]:
    """
    For each lever fit:
        y ~ cause + adjustment_set
    and report the adjusted coefficient on 'cause'.

    All numeric columns are standardised, so β is "change in standard deviations
    of the target per +1 SD of the lever".

    `declared_adjustment_sets` — per-lever sets from a curated ontology. When a
    lever has one it is used verbatim and the estimate says so; otherwise the
    graph-derived set is used and the estimate says that instead. A declared set
    is never merged with the derived one: mixing a domain claim with a heuristic
    would make the reported set untraceable.
    """
    G = build_dag(dag_edges)
    declared_adjustment_sets = declared_adjustment_sets or {}
    causal_roles = causal_roles or {}
    set_notes = set_notes or {}
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

        declared = declared_adjustment_sets.get(cause)
        if declared is not None:
            adj = {c for c in declared if c in df.columns}
            adj_source = "declared_domain_dag"
        else:
            adj = adjustment_set(
                cause=cause, outcome=target, G=G,
                confounders=confounders, mediators=mediators, context=context,
            )
            adj = {c for c in adj if c in df.columns}
            adj_source = "derived_from_graph"

        # Belt and braces: whatever the source, a mediator never survives into a
        # total-effect adjustment set.
        dropped_mediators = sorted(adj & set(mediators))
        adj -= set(mediators)

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

        interval_excludes_zero = (ci_lo > 0) or (ci_hi < 0)
        strength = _evidence_strength(p, n, len(adj), interval_excludes_zero)

        warning: str | None = None
        if n < 100:
            warning = f"Small sample (n={n}) — treat this estimate cautiously."
        elif not interval_excludes_zero:
            warning = (
                "The 95% interval includes zero — this estimate does not "
                "establish a direction."
            )

        notes: list[str] = []
        if adj_source == "declared_domain_dag":
            notes.append(
                "Adjustment set declared by the dataset ontology, not derived "
                "from column types."
            )
        else:
            notes.append(
                "Adjustment set derived from the assumed graph and the roles you "
                "assigned. No back-door search was performed."
            )
        if cause in set_notes:
            notes.append(set_notes[cause])
        if dropped_mediators:
            notes.append(
                "Excluded from the adjustment set because they are mediators: "
                + ", ".join(dropped_mediators)
                + ". Conditioning on them would turn a total effect into a "
                "direct effect."
            )

        effects.append(CausalEffect(
            feature=cause,
            effect_per_std=beta,
            effect_raw=effect_raw,
            std_err=se,
            t_stat=t,
            p_value=p,
            conf_int_lo=ci_lo,
            conf_int_hi=ci_hi,
            interval_method="ols_analytic_homoskedastic",
            adjusted_for=sorted(adj),
            adjustment_set_source=adj_source,
            estimand=(
                f"total effect of {cause} on {target}, under the assumed graph"
            ),
            causal_role=causal_roles.get(cause),
            controllable=True,
            n_observations=n,
            evidence_strength=strength,
            interval_excludes_zero=interval_excludes_zero,
            warning=warning,
            notes=notes,
        ))

    return sorted(effects, key=lambda e: abs(e.t_stat), reverse=True)
