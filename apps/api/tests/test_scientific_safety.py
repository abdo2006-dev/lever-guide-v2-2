"""
Scientific-safety regressions.

Every test here corresponds to a defect the Phase 0 audit measured on the shipped
demo. They are written against behaviour a reader would see, not against
implementation details, so that a future refactor cannot quietly reintroduce the
defect while keeping the internals recognisable.

See docs/audit/SCIENTIFIC_DISCREPANCIES.md and docs/audit/INTERVENTION_AUDIT.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ontology import INJECTION_MOULDING_ONTOLOGY as ONTO


def _effect(bundle, feature):
    return next((e for e in bundle["causal"] if e["feature"] == feature), None)


def _intervention(bundle, feature):
    return next((iv for iv in bundle["interventions"] if iv["feature"] == feature), None)


# ── D-1: the cooling-time adjustment set ─────────────────────────────────────

def test_cooling_time_adjusts_for_mold_temperature(demo_bundle):
    """
    The audit's Critical finding: mold_temperature_c could never enter any
    adjustment set, because the role-template graph has no lever -> lever edges.
    It is the required confounder for cooling time — operators extend cooling in
    response to observed mould temperature.
    """
    e = _effect(demo_bundle, "cooling_time_s")
    assert e is not None, "cooling_time_s was not estimated at all"
    assert "mold_temperature_c" in e["adjusted_for"]
    for required in ("part_weight_g", "shot_size_g", "ambient_humidity_pct",
                     "ambient_temperature_c", "maintenance_days_since_last"):
        assert required in e["adjusted_for"], f"{required} missing"
    assert e["adjustment_set_source"] == "declared_domain_dag"


def test_cooling_time_effect_matches_the_source_analysis(demo_bundle, demo_df):
    """
    Magnitude regression. The shipped app reported -1.171 p.p./SD against the
    source study's -1.743 — a 33 % understatement of the dataset's headline
    lever. beta is in SD/SD here, so it is converted before comparison.
    """
    e = _effect(demo_bundle, "cooling_time_s")
    sigma = demo_df.sample(2000, random_state=42)["scrap_rate_pct"].std()
    beta_pp = e["effect_per_std"] * sigma
    assert -2.0 < beta_pp < -1.6, (
        f"cooling_time_s = {beta_pp:.3f} p.p./SD; the source analysis reports "
        "-1.743 and the pre-fix behaviour was -1.171"
    )


# ── D-3: mediators must never be adjusted for ────────────────────────────────

def test_no_mediator_is_adjusted_for_in_any_estimate(demo_bundle):
    mediators = {"resin_moisture_pct", "calibration_drift_index", "tool_wear_index"}
    for e in demo_bundle["causal"]:
        overlap = mediators & set(e["adjusted_for"])
        assert not overlap, (
            f"{e['feature']} adjusts for mediator(s) {sorted(overlap)} — that "
            "turns a total effect into a direct effect"
        )


def test_adjustment_sets_differ_between_levers(demo_bundle):
    """
    The old behaviour gave every lever the identical set of every confounder
    plus every context column. Per-lever sets are the whole point.
    """
    declared = [
        tuple(sorted(e["adjusted_for"])) for e in demo_bundle["causal"]
        if e["adjustment_set_source"] == "declared_domain_dag"
    ]
    assert len(declared) >= 3
    assert len(set(declared)) > 1, "all levers were given the same adjustment set"


def test_adjustment_sets_are_not_every_numeric_column(demo_bundle, demo_df):
    numeric = set(demo_df.select_dtypes(include="number").columns) - {"scrap_rate_pct"}
    for e in demo_bundle["causal"]:
        if e["adjustment_set_source"] != "declared_domain_dag":
            continue
        assert set(e["adjusted_for"]) < numeric, (
            f"{e['feature']} adjusts for every numeric column"
        )


# ── I-4: physically impossible shot_size_g ───────────────────────────────────

def test_shot_size_cannot_be_ranked(demo_bundle):
    """
    The audit's rank-1 recommendation set shot_size_g to a constant while
    cavity_count and part_weight_g stayed fixed — a short shot for 92.5 % of
    rows. It must not be able to hold a rank.
    """
    iv = _intervention(demo_bundle, "shot_size_g")
    assert iv is not None, (
        "shot_size_g vanished from the response — it must be reported with a "
        "reason, not silently discarded"
    )
    assert iv["status"] == "infeasible"
    assert iv["rank"] == 0
    assert "cavity_count" in iv["status_reason"] or "part_weight" in iv["status_reason"]
    checks = {c["check"]: c for c in iv["feasibility_checks"]}
    assert checks["derived_relationship"]["passed"] is False


def test_shot_size_is_rejected_by_the_feasibility_layer_directly(demo_df):
    """Unit-level version of the same guard, independent of the endpoint."""
    from app.models.feasibility import check_intervention

    report = check_intervention(
        feature="shot_size_g", value=66.344, df=demo_df,
        ontology=ONTO, controllable=["shot_size_g"],
    )
    assert report.verdict == "infeasible"
    assert report.violation_share > 0.5
    assert report.violating_rows > 4000

    # A value consistent with the tooling for a given row is not rejected on
    # coupling grounds, which shows the check is about the identity and not a
    # blanket ban on the column.
    single = demo_df.iloc[[0]]
    consistent = float(single["cavity_count"].iloc[0] * single["part_weight_g"].iloc[0] * 1.08)
    ok = check_intervention(
        feature="shot_size_g", value=consistent, df=single,
        ontology=ONTO, controllable=["shot_size_g"],
    )
    coupling = {c["check"]: c for c in [c.model_dump() for c in ok.checks]}
    assert coupling["derived_relationship"]["passed"] is True


def test_no_eligible_intervention_fails_a_feasibility_check(demo_bundle):
    for iv in demo_bundle["interventions"]:
        if iv["status"] != "eligible":
            continue
        failed = [c["check"] for c in iv["feasibility_checks"] if not c["passed"]]
        assert not failed, f"{iv['feature']} is eligible but failed {failed}"


# ── D-2: hold_pressure_bar ───────────────────────────────────────────────────

def test_hold_pressure_is_never_a_strong_eligible_recommendation(demo_bundle):
    """
    The shipped app badged hold_pressure_bar "strong" and recommended
    decreasing it; the source analysis excluded it because its interval crosses
    zero. Until the estimator work that would settle this, it must not appear as
    a confident action.
    """
    iv = _intervention(demo_bundle, "hold_pressure_bar")
    if iv is not None:
        assert iv["status"] != "eligible", (
            "hold_pressure_bar is offered as an action despite conflicting evidence"
        )
        assert iv["rank"] == 0
        assert not (iv["status"] == "eligible" and iv["evidence_strength"] == "strong")

    e = _effect(demo_bundle, "hold_pressure_bar")
    if e is not None:
        assert e["evidence_strength"] != "strong", (
            "hold_pressure_bar is badged strong; its interval "
            f"[{e['conf_int_lo']:.3f}, {e['conf_int_hi']:.3f}] must gate that"
        )


def test_evidence_strength_is_gated_on_the_interval_not_the_p_value(demo_bundle):
    for e in demo_bundle["causal"]:
        if not e["interval_excludes_zero"]:
            assert e["evidence_strength"] in ("weak", "insufficient"), (
                f"{e['feature']} is rated {e['evidence_strength']} while its "
                "95% interval includes zero"
            )


# ── D-4 / D-5: levers the app could not reach ────────────────────────────────

@pytest.mark.parametrize("feature", ["dryer_dewpoint_c", "maintenance_days_since_last"])
def test_previously_unreachable_levers_are_estimated(demo_bundle, feature):
    """
    Both were mislabelled (confounder / context), so they could never be
    estimated or proposed. They are levers in the source taxonomy.
    """
    e = _effect(demo_bundle, feature)
    assert e is not None, f"{feature} is still not reachable by the estimator"
    assert e["adjusted_for"] == ["ambient_humidity_pct", "ambient_temperature_c"], (
        "the source's adjustment sets for these two are deliberately minimal, so "
        "the mediated path stays open and a total effect is recovered"
    )


@pytest.mark.parametrize("feature", ["dryer_dewpoint_c", "maintenance_days_since_last"])
def test_mediated_levers_are_not_presented_as_intervention_estimates(demo_bundle, feature):
    """
    Restoring them must not mean fabricating a causal result: their pathways run
    through mediators this simulation holds fixed, so the what-if is preliminary.
    """
    iv = _intervention(demo_bundle, feature)
    if iv is None:
        return
    assert iv["status"] == "exploratory"
    assert iv["rank"] == 0
    assert "mediator" in iv["status_reason"].lower()


def test_a_fixed_input_prediction_is_not_called_a_structural_intervention(demo_bundle):
    for iv in demo_bundle["interventions"]:
        assert iv["result_type"] == "predictive_what_if"
        assert "not automatically a causal intervention estimate" in iv["interpretation_note"]


# ── Result-type separation ───────────────────────────────────────────────────

def test_the_three_result_types_are_distinct_and_labelled(demo_bundle):
    assert {c["result_type"] for c in demo_bundle["correlations"]} == {"association"}
    assert {e["result_type"] for e in demo_bundle["causal"]} == {"adjusted_effect_estimate"}
    assert {iv["result_type"] for iv in demo_bundle["interventions"]} == {"predictive_what_if"}


def test_adjusted_estimates_state_the_graph_dependency(demo_bundle):
    for e in demo_bundle["causal"]:
        note = e["interpretation_note"].lower()
        assert "causal graph" in note
        assert "unmeasured confounding" in note


def test_no_unsupported_phrase_appears_anywhere_in_the_payload(demo_bundle):
    import json
    blob = json.dumps(demo_bundle).lower()
    for phrase in (
        "proven cause",
        "true causal impact",
        "guaranteed improvement",
        "ai-discovered causal graph",
        "ai-discovered",
    ):
        assert phrase not in blob, f"unsupported phrase in payload: {phrase!r}"


def test_recommended_intervention_language_is_not_used_for_a_bare_what_if(demo_bundle):
    """
    The word "recommend" must not be attached to a simulation that did not pass
    screening.
    """
    for iv in demo_bundle["interventions"]:
        if iv["status"] == "eligible":
            continue
        text = " ".join([iv["rationale"], iv["status_reason"], *iv["assumptions"]]).lower()
        assert "recommended intervention" not in text


def test_graph_is_labelled_an_assumption_not_a_discovery(demo_bundle):
    dag = demo_bundle["dag_validation"]
    assert dag["dag_source"] == "declared_domain_ontology"
    assert dag["graph_assumption"]
    assert "assumption" in dag["graph_assumption"].lower()


# ── Uncertainty ──────────────────────────────────────────────────────────────

def test_effect_intervals_declare_their_method(demo_bundle):
    for e in demo_bundle["causal"]:
        assert e["interval_method"] == "ols_analytic_homoskedastic"
        assert e["conf_int_lo"] < e["conf_int_hi"]


def test_simulation_intervals_are_distinguished_from_regression_intervals(demo_bundle):
    """A predictive interval and a regression CI are different objects."""
    methods = {iv["interval_method"] for iv in demo_bundle["interventions"]
               if iv["interval_method"] is not None}
    assert methods <= {"row_bootstrap_fixed_model"}
    assert "ols_analytic_homoskedastic" not in methods


def test_missing_intervals_are_null_and_explained_never_invented():
    """
    With too few rows to resample, the bounds must serialise as null with a
    stated reason — not as a placeholder a UI would render as a real interval.
    """
    from app.models.intervention import _bootstrap_interval

    rng = np.random.default_rng(0)
    lo, hi = _bootstrap_interval(np.zeros(10), np.ones(10), rng)
    assert lo is None and hi is None

    lo, hi = _bootstrap_interval(
        rng.normal(size=500), rng.normal(size=500) - 0.5, rng
    )
    assert lo is not None and hi is not None and lo < hi


def test_every_intervention_either_has_both_bounds_or_neither(demo_bundle):
    for iv in demo_bundle["interventions"]:
        lo, hi = iv["expected_kpi_change_lo"], iv["expected_kpi_change_hi"]
        assert (lo is None) == (hi is None), f"{iv['feature']} has a half interval"
        if lo is None:
            assert iv["interval_method"] is None
            assert iv["uncertainty_status"].startswith("not_computed")
        else:
            assert lo < hi
            assert iv["interval_method"] is not None


# ── Support and feasibility, generally ───────────────────────────────────────

def test_out_of_support_value_is_flagged(demo_df):
    from app.models.feasibility import check_intervention

    # Inside the declared range but far outside anything observed.
    report = check_intervention(
        feature="cooling_time_s", value=39.0, df=demo_df,
        ontology=ONTO, controllable=["cooling_time_s"],
    )
    assert report.verdict == "unsupported"
    assert report.support_status == "outside_observed_within_declared"
    assert "extrapolat" in report.reason

    # Outside the declared physical range entirely.
    hard = check_intervention(
        feature="cooling_time_s", value=400.0, df=demo_df,
        ontology=ONTO, controllable=["cooling_time_s"],
    )
    assert hard.verdict == "infeasible"
    assert hard.support_status == "outside_declared"


def test_non_controllable_variable_is_rejected_with_a_reason(demo_df):
    from app.models.feasibility import check_intervention

    report = check_intervention(
        feature="part_weight_g", value=100.0, df=demo_df,
        ontology=ONTO, controllable=[],
    )
    assert report.verdict == "not_eligible"
    assert report.reason


def test_categorical_values_are_validated(demo_df):
    from app.models.feasibility import check_categorical_value

    ok = check_categorical_value("operator_shift", "A_Day", ONTO, demo_df)
    assert ok.verdict == "feasible"

    bad = check_categorical_value("operator_shift", "D_Weekend", ONTO, demo_df)
    assert bad.verdict == "infeasible"
    assert "known levels" in bad.reason.lower()


def test_support_checks_work_without_an_ontology(demo_df):
    """The generic path still gets observed-range checking."""
    from app.models.feasibility import check_intervention

    inside = check_intervention(
        feature="cooling_time_s", value=15.0, df=demo_df,
        ontology=None, controllable=["cooling_time_s"],
    )
    assert inside.verdict == "feasible"
    assert inside.support_status == "within_observed"

    outside = check_intervention(
        feature="cooling_time_s", value=200.0, df=demo_df,
        ontology=None, controllable=["cooling_time_s"],
    )
    assert outside.verdict == "unsupported"


# ── Ranking honesty ──────────────────────────────────────────────────────────

def test_only_eligible_results_are_ranked(demo_bundle):
    for iv in demo_bundle["interventions"]:
        if iv["status"] == "eligible":
            assert iv["rank"] >= 1
        else:
            assert iv["rank"] == 0, (
                f"{iv['feature']} is {iv['status']} but holds rank {iv['rank']}"
            )


def test_ranks_are_contiguous_and_ordered_by_magnitude(demo_bundle):
    eligible = sorted(
        (iv for iv in demo_bundle["interventions"] if iv["status"] == "eligible"),
        key=lambda iv: iv["rank"],
    )
    assert [iv["rank"] for iv in eligible] == list(range(1, len(eligible) + 1))
    magnitudes = [abs(iv["expected_kpi_change"]) for iv in eligible]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_no_eligible_result_conflicts_with_its_own_adjusted_estimate(demo_bundle):
    for iv in demo_bundle["interventions"]:
        if iv["status"] == "eligible":
            assert iv["adjustment_support"] == "aligned", (
                f"{iv['feature']} is ranked while its adjusted estimate is "
                f"{iv['adjustment_support']}"
            )


def test_executive_summary_only_promotes_eligible_levers(demo_bundle):
    eligible = {iv["feature"] for iv in demo_bundle["interventions"]
                if iv["status"] == "eligible"}
    for lever in demo_bundle["executive"]["top_levers"]:
        assert lever in eligible


# ── Traceability ─────────────────────────────────────────────────────────────

def test_provenance_carries_enough_to_reconstruct_the_analysis(demo_bundle):
    p = demo_bundle["provenance"]
    assert p["analysis_mode"] == "causal"
    assert p["ontology_id"] == "injection_molding_demo"
    assert p["ontology_version"] == ONTO.version
    assert p["dag_source"] == "declared_domain_ontology"
    assert p["effect_estimator"]
    assert p["effect_interval_method"] == "ols_analytic_homoskedastic"
    assert p["simulation_model"]
    assert "in-sample" in p["simulation_evaluation"]
    assert p["n_rows_supplied"] == 5000
    assert p["n_rows_analysed"] == 2000
    assert p["sampling_note"]
    assert p["train_eval_strategy"]
    assert p["column_roles"]["cooling_time_s"] == "controllable"


def test_every_result_carries_its_own_traceability(demo_bundle):
    for e in demo_bundle["causal"]:
        assert e["estimand"]
        assert e["n_observations"] > 0
        assert e["adjustment_set_source"] in ("declared_domain_dag", "derived_from_graph")
    for iv in demo_bundle["interventions"]:
        assert iv["status"]
        assert iv["status_reason"]
        assert iv["support_status"]
        assert iv["simulation_model"]


def test_source_deviations_are_surfaced_on_the_affected_estimates(demo_bundle):
    """
    Three adjustment sets had to depart from the source study. Each affected
    estimate must say so on its own record, not only in a source comment.
    """
    for feature in ("mold_temperature_c", "injection_pressure_bar", "hold_pressure_bar"):
        e = _effect(demo_bundle, feature)
        if e is None:
            continue
        joined = " ".join(e["notes"]).lower()
        assert "source" in joined, f"{feature} does not disclose its deviation"


# ── Model availability ───────────────────────────────────────────────────────

def test_model_statuses_cover_every_configured_model(demo_bundle):
    statuses = {s["model"]: s for s in demo_bundle["model_statuses"]}
    assert set(statuses) == {"ols", "ridge", "rf", "xgb", "lgbm"}
    ran = {r["model"] for r in demo_bundle["predictive"]}
    for key, s in statuses.items():
        if s["status"] == "succeeded":
            assert key in ran
        else:
            assert key not in ran
            assert s["detail"], f"{key} failed without saying why"


def test_failed_models_produce_a_visible_warning(demo_bundle):
    failed = [s for s in demo_bundle["model_statuses"] if s["status"] != "succeeded"]
    if not failed:
        pytest.skip("all optional models are importable in this environment")
    blob = " ".join(demo_bundle["warnings"])
    for s in failed:
        assert s["display_name"] in blob, (
            f"{s['display_name']} did not run and no warning says so"
        )
