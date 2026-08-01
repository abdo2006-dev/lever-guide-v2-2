"""
Tests for the curated demo ontology.

These assert the *declaration*, not any dataset: they are what stops a role or an
adjustment set from silently changing meaning, and they are the synchronisation
mechanism between the Python ontology and the frontend's generated JSON copy.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ontology import (
    GENERATED_JSON_PATH,
    INJECTION_MOULDING_ONTOLOGY as ONTO,
    SOURCE_DEVIATIONS,
    build_graph,
    ontology_json,
    resolve_ontology,
    validate_ontology,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


# ── Roles of the variables the audit named ───────────────────────────────────

# (name, causal_role, controllable, derived, mediator, intervention_eligibility)
EXPECTED_ROLES = [
    ("dryer_dewpoint_c",            "process_lever",  True,  False, False, "mediated_unsupported"),
    ("resin_moisture_pct",          "mediator",       False, False, True,  "not_eligible"),
    ("maintenance_days_since_last", "planning_lever", True,  False, False, "preliminary"),
    ("calibration_drift_index",     "mediator",       False, False, True,  "not_eligible"),
    ("tool_wear_index",             "mediator",       False, False, True,  "not_eligible"),
    ("cooling_time_s",              "process_lever",  True,  False, False, "eligible"),
    ("mold_temperature_c",          "process_lever",  True,  False, False, "eligible"),
    ("injection_pressure_bar",      "process_lever",  True,  False, False, "eligible"),
    ("hold_pressure_bar",           "process_lever",  True,  False, False, "eligible"),
    ("shot_size_g",                 "process_lever",  True,  True,  False, "derived_constrained"),
    ("cavity_count",                "context",        False, False, False, "not_eligible"),
    ("part_weight_g",               "context",        False, False, False, "not_eligible"),
    ("scrap_rate_pct",              "outcome",        False, False, False, "not_eligible"),
]


@pytest.mark.parametrize(
    "name,causal_role,controllable,derived,mediator,eligibility", EXPECTED_ROLES
)
def test_demo_ontology_roles(name, causal_role, controllable, derived, mediator, eligibility):
    spec = ONTO.spec(name)
    assert spec is not None, f"{name} is missing from the demo ontology"
    assert spec.causal_role == causal_role
    assert spec.controllable is controllable
    assert spec.derived is derived
    assert spec.mediator is mediator
    assert spec.intervention_eligibility == eligibility


def test_every_named_variable_has_label_unit_and_ranges():
    """Every variable the audit named must be fully described, not just role-tagged."""
    for name, *_ in EXPECTED_ROLES:
        spec = ONTO.spec(name)
        assert spec.label, f"{name} has no human-readable label"
        assert spec.unit, f"{name} has no unit"
        assert spec.valid_range is not None, f"{name} has no declared valid range"
        assert spec.observed_range is not None, f"{name} has no observed range"
        lo, hi = spec.valid_range
        assert lo < hi


def test_roles_are_not_inferred_from_data_type():
    """
    Variables with identical numeric types carry different causal roles.

    `tool_wear_index` and `mold_temperature_c` are both continuous numerics; one
    is a mediator that must never be adjusted for, the other is a lever and a
    required adjuster. Nothing about their dtype distinguishes them.
    """
    assert ONTO.spec("tool_wear_index").mediator is True
    assert ONTO.spec("mold_temperature_c").mediator is False
    assert ONTO.spec("part_weight_g").causal_role == "context"
    assert ONTO.spec("shot_size_g").causal_role == "process_lever"


def test_correcting_the_roles_the_audit_flagged():
    """Regression guard on the specific reassignments Phase 1A made."""
    # clamp_force_kn was offered as a lever; the source calls it context.
    assert ONTO.spec("clamp_force_kn").causal_role == "context"
    assert ONTO.spec("clamp_force_kn").controllable is False
    # cycle_time_s mechanically subsumes cooling_time_s — leakage, not a mediator.
    assert ONTO.spec("cycle_time_s").causal_role == "post_treatment_outcome"
    assert ONTO.spec("cycle_time_s").column_role == "ignore"
    # part_weight_g was a mediator, which dropped it from every adjustment set.
    assert ONTO.spec("part_weight_g").mediator is False


def test_column_role_mapping_is_total_and_coarse():
    roles = ONTO.column_roles()
    assert set(roles) == set(ONTO.names)
    assert roles["maintenance_days_since_last"] == "planning_lever"
    assert roles["dryer_dewpoint_c"] == "controllable"
    assert roles["resin_moisture_pct"] == "mediator"
    assert roles["scrap_rate_pct"] == "outcome"
    # Nothing in a curated ontology may be left unassigned.
    assert "unassigned" not in set(roles.values())


# ── Graph and adjustment sets ────────────────────────────────────────────────

def test_ontology_is_internally_consistent():
    problems = validate_ontology(ONTO)
    assert problems == [], "declared ontology is inconsistent:\n" + "\n".join(problems)


def test_declared_graph_contains_the_reactive_compensation_edge():
    """
    The role-template graph could not express lever -> lever structure at all,
    which is why mold_temperature_c could never adjust for cooling time.
    """
    assert ("mold_temperature_c", "cooling_time_s") in ONTO.edges
    G = build_graph(ONTO)
    assert G.has_edge("mold_temperature_c", "cooling_time_s")
    assert not G.has_edge("cooling_time_s", "mold_temperature_c")


def test_cooling_time_adjustment_set_contains_the_required_confounders():
    adj = set(ONTO.adjustment_sets["cooling_time_s"])
    for required in (
        "mold_temperature_c", "part_weight_g", "shot_size_g",
        "ambient_humidity_pct", "ambient_temperature_c",
        "maintenance_days_since_last",
    ):
        assert required in adj, f"{required} missing from the cooling-time adjustment set"


def test_no_mediator_appears_in_any_total_effect_adjustment_set():
    mediators = set(ONTO.mediators)
    assert mediators == {"resin_moisture_pct", "calibration_drift_index", "tool_wear_index"}
    for cause, adjusters in ONTO.adjustment_sets.items():
        overlap = mediators & set(adjusters)
        assert not overlap, f"{cause} adjusts for mediator(s) {sorted(overlap)}"


def test_adjustment_sets_are_not_every_numeric_column():
    """A per-lever set is a claim about that lever, not a dump of the schema."""
    numeric = {
        v.name for v in ONTO.variables
        if v.valid_range is not None and v.causal_role != "outcome"
    }
    sets = {c: set(a) for c, a in ONTO.adjustment_sets.items()}
    assert len({frozenset(s) for s in sets.values()}) > 1, "all levers share one set"
    for cause, adj in sets.items():
        assert adj < numeric, f"{cause} adjusts for every numeric column"


def test_source_deviations_are_declared_for_every_set_that_departs_from_the_source():
    """
    Where the declared set differs from datathon-CUB-2026/src/utils.py, the
    reason must be recorded — a silent divergence from the source of truth is
    exactly what this phase exists to prevent.
    """
    assert set(SOURCE_DEVIATIONS) == {
        "mold_temperature_c", "injection_pressure_bar", "hold_pressure_bar",
    }
    # cooling_time_s is used verbatim and therefore must have no deviation note.
    assert "cooling_time_s" not in SOURCE_DEVIATIONS
    for cause, reason in SOURCE_DEVIATIONS.items():
        assert cause in ONTO.adjustment_sets
        assert len(reason) > 40


def test_hold_pressure_evidence_is_marked_conflicting():
    spec = ONTO.spec("hold_pressure_bar")
    assert spec.evidence_status == "conflicting"
    assert "specification" in spec.notes.lower()


def test_derived_relationship_for_shot_size():
    rel = ONTO.derived_for("shot_size_g")
    assert rel is not None
    assert set(rel.inputs) == {"cavity_count", "part_weight_g"}
    assert rel.ratio_lo < 1.0 < rel.ratio_hi
    assert ONTO.derived_for("cooling_time_s") is None


def test_graph_assumption_is_stated_and_makes_no_proof_claim():
    text = ONTO.graph_assumption.lower()
    assert "assumption" in text
    assert "not discovered from data" in text or "was not discovered" in text
    # Affirmative proof claims, as opposed to explicit denials of them.
    for banned in ("is proven", "guaranteed", "ai-discovered", "proven cause"):
        assert banned not in text
    assert "is not proven" in text


# ── Resolution ───────────────────────────────────────────────────────────────

def test_resolve_ontology_matches_the_demo_and_nothing_else():
    assert resolve_ontology(ONTO.names, "scrap_rate_pct") is ONTO
    # Wrong target -> no ontology, so no domain claims are applied.
    assert resolve_ontology(ONTO.names, "cycle_time_s") is None
    # A merely similar dataset gets the generic path.
    assert resolve_ontology(["a", "b", "scrap_rate_pct"], "scrap_rate_pct") is None


# ── Frontend synchronisation ─────────────────────────────────────────────────

def test_generated_json_is_in_sync():
    """
    The frontend's copy is generated, not maintained. If this fails, run:

        cd apps/api && ./.venv/bin/python scripts/export_ontology.py
    """
    committed = (REPO_ROOT / GENERATED_JSON_PATH).read_text(encoding="utf-8")
    assert committed == ontology_json(ONTO), (
        "apps/web/lib/generated/demo-ontology.json has drifted from the Python "
        "ontology. Regenerate it with scripts/export_ontology.py."
    )


def test_generated_json_carries_roles_and_ranges():
    payload = json.loads((REPO_ROOT / GENERATED_JSON_PATH).read_text(encoding="utf-8"))
    assert payload["version"] == ONTO.version
    assert payload["column_roles"]["dryer_dewpoint_c"] == "controllable"
    assert payload["column_roles"]["maintenance_days_since_last"] == "planning_lever"
    by_name = {v["name"]: v for v in payload["variables"]}
    assert by_name["shot_size_g"]["derived"] is True
    assert by_name["cooling_time_s"]["observed_range"] == [7.19, 27.29]
