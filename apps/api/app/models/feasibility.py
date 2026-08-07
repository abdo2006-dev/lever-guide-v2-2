"""
Intervention feasibility and support checking.

A simulated change is only worth showing as a candidate action if the row it
produces could exist. This module answers that question and returns *why*, so an
intervention is never silently dropped and never silently promoted.

It works with or without a curated ontology. Without one it can still check
observed support; with one it can also check declared physical bounds, declared
categories and documented coupling identities between columns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from app.ontology.schema import DatasetOntology
from app.schemas import FeasibilityCheck

# Statuses this module can conclude. The caller maps them onto the intervention
# status vocabulary; it never invents one of its own.
FeasibilityVerdict = str  # "feasible" | "infeasible" | "unsupported" | "not_eligible"


@dataclass
class FeasibilityReport:
    verdict: FeasibilityVerdict
    reason: str
    support_status: str = "unknown"
    checks: list[FeasibilityCheck] = field(default_factory=list)
    # Share of rows for which the proposed value violates a derived relationship.
    violation_share: float = 0.0
    violating_rows: int = 0

    @property
    def feasible(self) -> bool:
        return self.verdict == "feasible"


def _check(name: str, passed: bool, detail: str) -> FeasibilityCheck:
    return FeasibilityCheck(check=name, passed=passed, detail=detail)


def _support_status(
    value: float,
    observed: Optional[tuple[float, float]],
    declared: Optional[tuple[float, float]],
) -> str:
    if observed is not None and observed[0] <= value <= observed[1]:
        return "within_observed"
    if declared is not None:
        if declared[0] <= value <= declared[1]:
            return "outside_observed_within_declared"
        return "outside_declared"
    if observed is not None:
        # No declared bounds to fall back on, but we know it left the data.
        return "outside_observed_within_declared"
    return "unknown"


def observed_range(df: pd.DataFrame, column: str) -> Optional[tuple[float, float]]:
    if column not in df.columns:
        return None
    s = pd.to_numeric(df[column], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.min()), float(s.max())


def check_intervention(
    feature: str,
    value: float,
    df: pd.DataFrame,
    ontology: Optional[DatasetOntology] = None,
    controllable: Optional[list[str]] = None,
) -> FeasibilityReport:
    """
    Decide whether setting `feature` to `value` produces rows that could exist.

    Checks, in order: eligibility, numeric validity, declared bounds, observed
    support, and any documented derived relationship. Every check is recorded,
    passed or failed, so the UI can show the reasoning rather than a verdict.
    """
    checks: list[FeasibilityCheck] = []
    spec = ontology.spec(feature) if ontology else None

    # ── 1. Is this variable an intervention target at all? ────────────────────
    if spec is not None:
        eligible = spec.intervention_eligibility != "not_eligible"
        checks.append(_check(
            "eligibility",
            eligible,
            f"{spec.label} is declared '{spec.intervention_eligibility}'"
            + ("" if eligible else " — it is not something an operator sets."),
        ))
        if not eligible:
            return FeasibilityReport(
                verdict="not_eligible",
                reason=(
                    f"{spec.label} is not an intervention target: "
                    f"{spec.notes or 'it is not controllable.'}"
                ),
                checks=checks,
            )
    elif controllable is not None:
        eligible = feature in controllable
        checks.append(_check(
            "eligibility", eligible,
            "declared controllable by the user" if eligible
            else "not assigned a controllable role",
        ))
        if not eligible:
            return FeasibilityReport(
                verdict="not_eligible",
                reason=f"'{feature}' was not assigned a controllable role.",
                checks=checks,
            )

    # ── 2. Numeric validity ───────────────────────────────────────────────────
    if not np.isfinite(value):
        checks.append(_check("finite_value", False, f"proposed value is {value}"))
        return FeasibilityReport(
            verdict="infeasible",
            reason="The proposed value is not a finite number.",
            checks=checks,
        )

    # ── 3. Declared bounds and observed support ───────────────────────────────
    declared = spec.valid_range if spec else None
    observed = (spec.observed_range if spec else None) or observed_range(df, feature)
    support = _support_status(value, observed, declared)

    if declared is not None:
        inside = declared[0] <= value <= declared[1]
        checks.append(_check(
            "declared_range", inside,
            f"proposed {value:.4g} against declared {declared[0]:.4g}–{declared[1]:.4g}"
            + ("" if inside else " — outside the declared process limits"),
        ))
        if not inside:
            return FeasibilityReport(
                verdict="infeasible",
                reason=(
                    f"{value:.4g} is outside the declared operating range for "
                    f"{feature} ({declared[0]:.4g}–{declared[1]:.4g})."
                ),
                support_status=support,
                checks=checks,
            )

    if observed is not None:
        inside = observed[0] <= value <= observed[1]
        checks.append(_check(
            "observed_support", inside,
            f"proposed {value:.4g} against observed {observed[0]:.4g}–{observed[1]:.4g}"
            + ("" if inside else " — the model has not seen this region"),
        ))

    # ── 4. Documented derived relationships ───────────────────────────────────
    rel = ontology.derived_for(feature) if ontology else None
    if rel is not None:
        missing = [c for c in rel.inputs if c not in df.columns]
        if missing:
            checks.append(_check(
                "derived_relationship", False,
                f"cannot evaluate: {', '.join(missing)} not present",
            ))
            return FeasibilityReport(
                verdict="unsupported",
                reason=(
                    f"{feature} is a derived quantity, but the columns it depends "
                    f"on ({', '.join(rel.inputs)}) are not all present, so its "
                    "feasibility cannot be established."
                ),
                support_status=support,
                checks=checks,
            )

        basis = pd.Series(np.ones(len(df)), index=df.index, dtype=float)
        for col in rel.inputs:
            basis = basis * pd.to_numeric(df[col], errors="coerce")
        basis = basis.replace(0, np.nan)
        implied = value / basis
        usable = implied.dropna()
        violating = int(((usable < rel.ratio_lo) | (usable > rel.ratio_hi)).sum())
        share = (violating / len(usable)) if len(usable) else 0.0

        product = " x ".join(rel.inputs)
        passed = share <= rel.max_violation_share
        checks.append(_check(
            "derived_relationship", passed,
            f"{violating} of {len(usable)} rows ({share:.1%}) would violate "
            f"{feature} ≈ {product} x k, k in {rel.ratio_lo}–{rel.ratio_hi}",
        ))
        if not passed:
            return FeasibilityReport(
                verdict="infeasible",
                reason=(
                    f"Setting {feature} to {value:.4g} while holding {product} "
                    f"fixed is physically impossible for {violating} of "
                    f"{len(usable)} rows ({share:.1%}). {rel.description}"
                ),
                support_status=support,
                checks=checks,
                violation_share=share,
                violating_rows=violating,
            )

    if support == "outside_declared":
        return FeasibilityReport(
            verdict="infeasible",
            reason=f"{value:.4g} is outside the declared operating range for {feature}.",
            support_status=support,
            checks=checks,
        )
    if support == "outside_observed_within_declared":
        return FeasibilityReport(
            verdict="unsupported",
            reason=(
                f"{value:.4g} lies outside the range observed in this dataset. "
                "The simulation is extrapolating, which is not as reliable as "
                "interpolating inside the observed range."
            ),
            support_status=support,
            checks=checks,
        )

    return FeasibilityReport(
        verdict="feasible",
        reason="Inside observed support and consistent with documented constraints.",
        support_status=support,
        checks=checks,
    )


def check_categorical_value(
    feature: str, value: str, ontology: Optional[DatasetOntology], df: pd.DataFrame
) -> FeasibilityReport:
    """Validate a proposed categorical level against declared or observed levels."""
    spec = ontology.spec(feature) if ontology else None
    declared = list(spec.categories) if (spec and spec.categories) else None
    if declared is None and feature in df.columns:
        declared = sorted(df[feature].dropna().astype(str).unique().tolist())

    if declared is None:
        return FeasibilityReport(
            verdict="unsupported",
            reason=f"No known category list for '{feature}'.",
            checks=[_check("categorical_value", False, "no category list available")],
        )

    ok = str(value) in declared
    shown = ", ".join(declared[:8]) + ("…" if len(declared) > 8 else "")
    return FeasibilityReport(
        verdict="feasible" if ok else "infeasible",
        reason=(
            f"'{value}' is a known level of {feature}." if ok
            else f"'{value}' is not a known level of {feature}. Known levels: {shown}."
        ),
        support_status="within_observed" if ok else "outside_declared",
        checks=[_check(
            "categorical_value", ok,
            f"'{value}' against {len(declared)} known levels",
        )],
    )
