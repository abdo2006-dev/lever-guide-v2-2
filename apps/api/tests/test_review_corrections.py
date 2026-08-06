"""
Independent-review corrections: zero-variance inputs, silently vanishing
levers, and the screw-speed documentation.

Each test reproduces a failure mode an adversarial review found after the rest
of Phase 1A, exercised against the real `/api/analyze` endpoint rather than
only against helper functions, so a future refactor cannot quietly
reintroduce the defect while keeping the internals recognisable.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def isolated_runtime_env(monkeypatch, tmp_path):
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("QDRANT_URL", "")
    monkeypatch.setenv("QDRANT_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY", "")
    yield
    try:
        from app.rag import close_retrieval_store
        close_retrieval_store()
    except Exception:
        pass


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def _payload(csv: str, roles: dict, **over) -> dict:
    base = {
        "dataset_csv": csv,
        "dataset_name": "Uploaded",
        "target": "outcome",
        "task": "regression",
        "improve_direction": "decrease",
        "analysis_mode": "causal",
        "column_roles": roles,
        "dag_edges": [],
        "random_seed": 42,
    }
    base.update(over)
    return base


def _assert_no_non_finite(obj) -> None:
    """Recursively assert no float in `obj` is NaN or +/-inf."""
    if isinstance(obj, float):
        assert math.isfinite(obj), f"non-finite float reached the response: {obj}"
    elif isinstance(obj, dict):
        for v in obj.values():
            _assert_no_non_finite(v)
    elif isinstance(obj, list):
        for v in obj:
            _assert_no_non_finite(v)


def _effect(bundle, feature):
    return next((e for e in bundle["causal"] if e["feature"] == feature), None)


def _intervention(bundle, feature):
    return next((iv for iv in bundle["interventions"] if iv["feature"] == feature), None)


# ── Correction 1: constant-valued inputs must not crash the API ─────────────

def test_constant_outcome_is_a_structured_validation_error(client):
    rng = np.random.default_rng(1)
    n = 60
    df = pd.DataFrame({
        "outcome": np.full(n, 5.0),
        "lever": rng.normal(50, 5, n),
    })
    resp = client.post("/api/analyze", json=_payload(
        df.to_csv(index=False), {"lever": "controllable"},
    ))
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "CONSTANT_OUTCOME"
    problem = detail["problems"][0]
    assert problem["code"] == "CONSTANT_OUTCOME"
    assert problem["message"] and problem["remedy"]
    assert "outcome" in problem["columns"]


def test_constant_treatment_is_excluded_but_descriptive_analysis_survives(client):
    """
    A constant treatment cannot have an identifiable effect. It must not crash,
    must not be silently fit, and must not block the rest of the analysis —
    another lever's estimate and the predictive/EDA results still come back.
    """
    rng = np.random.default_rng(2)
    n = 120
    real_lever = rng.normal(50, 5, n)
    df = pd.DataFrame({
        "outcome": rng.normal(10, 2, n) - 0.1 * real_lever,
        "const_lever": np.full(n, 42.0),
        "real_lever": real_lever,
        "conf": rng.normal(0, 1, n),
    })
    resp = client.post("/api/analyze", json=_payload(
        df.to_csv(index=False),
        {"const_lever": "controllable", "real_lever": "controllable", "conf": "confounder"},
    ))
    assert resp.status_code == 200, resp.text
    bundle = resp.json()
    _assert_no_non_finite(bundle)

    assert _effect(bundle, "const_lever") is None, "a constant treatment must not get a fitted effect"
    assert _effect(bundle, "real_lever") is not None, "the other treatment must still be estimated"
    assert len(bundle["predictive"]) > 0

    excluded = bundle["provenance"]["excluded_columns"]
    entry = next(e for e in excluded if e["column"] == "const_lever")
    assert entry["scope"] == "treatment"
    assert "single observed value" in entry["reason"]

    iv = _intervention(bundle, "const_lever")
    assert iv is not None, "a constant treatment must not silently vanish from interventions"
    assert iv["status"] != "eligible"
    assert "variation" in iv["status_reason"]

    # The user's original role assignment is preserved, not silently changed.
    assert bundle["provenance"]["column_roles"]["const_lever"] == "controllable"


def test_all_constant_treatments_reject_the_whole_request(client):
    rng = np.random.default_rng(3)
    n = 60
    df = pd.DataFrame({
        "outcome": rng.normal(10, 2, n),
        "const_lever": np.full(n, 7.0),
        "conf": rng.normal(0, 1, n),
    })
    resp = client.post("/api/analyze", json=_payload(
        df.to_csv(index=False), {"const_lever": "controllable", "conf": "confounder"},
    ))
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    codes = {p["code"] for p in detail["problems"]}
    assert "ALL_TREATMENTS_CONSTANT" in codes


def test_constant_numeric_confounder_is_excluded_not_fatal(client):
    rng = np.random.default_rng(4)
    n = 150
    lever = rng.normal(50, 5, n)
    df = pd.DataFrame({
        "outcome": rng.normal(10, 2, n) - 0.08 * lever,
        "lever": lever,
        "const_confounder": np.full(n, 3.14),
        "real_confounder": rng.normal(0, 1, n),
    })
    resp = client.post("/api/analyze", json=_payload(
        df.to_csv(index=False),
        {"lever": "controllable", "const_confounder": "confounder", "real_confounder": "confounder"},
    ))
    assert resp.status_code == 200, resp.text
    bundle = resp.json()
    _assert_no_non_finite(bundle)

    e = _effect(bundle, "lever")
    assert e is not None, "the lever must still be estimated once the dead column is dropped"
    assert "const_confounder" not in e["adjusted_for"]
    assert "real_confounder" in e["adjusted_for"]

    excluded = bundle["provenance"]["excluded_columns"]
    entry = next(x for x in excluded if x["column"] == "const_confounder")
    assert entry["scope"] == "adjustment_set"
    assert entry["lever"] == "lever"

    assert any("const_confounder" in w for w in bundle["warnings"]) or any(
        "excluded" in w.lower() for w in bundle["warnings"]
    )
    # The user's original configuration is unchanged.
    assert bundle["provenance"]["column_roles"]["const_confounder"] == "confounder"


def test_constant_categorical_predictor_is_excluded_not_fatal(client):
    """A single-level categorical confounder is zero-variance too, and must not crash."""
    rng = np.random.default_rng(5)
    n = 150
    lever = rng.normal(50, 5, n)
    df = pd.DataFrame({
        "outcome": rng.normal(10, 2, n) - 0.08 * lever,
        "lever": lever,
        "const_category": ["only_level"] * n,
        "real_confounder": rng.normal(0, 1, n),
    })
    resp = client.post("/api/analyze", json=_payload(
        df.to_csv(index=False),
        {"lever": "controllable", "const_category": "confounder", "real_confounder": "confounder"},
    ))
    assert resp.status_code == 200, resp.text
    bundle = resp.json()
    _assert_no_non_finite(bundle)
    e = _effect(bundle, "lever")
    assert e is not None
    excluded_names = {x["column"] for x in bundle["provenance"]["excluded_columns"]}
    assert "const_category" in excluded_names


def test_near_singular_adjustment_set_does_not_crash_the_response(client):
    """
    The independent review's exact reproduction: too many adjuster columns for
    too few rows drives statsmodels' OLS to a non-finite coefficient, which
    previously reached `json.dumps(..., allow_nan=False)` and raised
    `ValueError: Out of range float values are not JSON compliant`.
    """
    rng = np.random.default_rng(6)
    n = 40
    data = {
        "outcome": rng.normal(10, 2, n),
        "lever": rng.normal(100, 10, n),
        "const_confounder": np.full(n, 7.5),
    }
    roles = {"lever": "controllable", "const_confounder": "confounder"}
    for i in range(39):
        data[f"conf_{i}"] = rng.normal(0, 1, n)
        roles[f"conf_{i}"] = "confounder"
    df = pd.DataFrame(data)

    resp = client.post("/api/analyze", json=_payload(df.to_csv(index=False), roles))
    assert resp.status_code == 200, resp.text
    bundle = resp.json()
    _assert_no_non_finite(bundle)
    # json.dumps with allow_nan=False is the exact check that used to crash.
    json.dumps(bundle, allow_nan=False)


def test_demo_analysis_is_unaffected_by_the_zero_variance_guards(demo_bundle):
    """The shipped demo has no zero-variance columns, so nothing is excluded."""
    assert demo_bundle["provenance"]["excluded_columns"] == []


# ── Correction 2: no configured lever may disappear silently ────────────────

def test_every_configured_demo_lever_gets_an_explicit_intervention_status(demo_bundle):
    from app.ontology import INJECTION_MOULDING_ONTOLOGY as onto
    levers = sorted(c for c, r in onto.column_roles().items() if r in ("controllable", "planning_lever"))
    reported = {iv["feature"] for iv in demo_bundle["interventions"]}
    missing = set(levers) - reported
    assert not missing, f"lever(s) vanished from interventions with no status record: {missing}"
    for iv in demo_bundle["interventions"]:
        assert iv["status_reason"], f"{iv['feature']} has a status but no human-readable reason"


def test_screw_speed_rpm_appears_with_an_explicit_non_ranked_status(demo_bundle):
    """
    screw_speed_rpm has a strong adjusted estimate but the simulator found no
    improving direction for it. It must be visible with a reason, not omitted,
    and it must not enter the ranked list on the strength of an estimate this
    branch itself documents as specification-dependent (see
    docs/implementation/PHASE_1A_TRUTH_IN_LABELLING.md §8.1).
    """
    e = _effect(demo_bundle, "screw_speed_rpm")
    assert e is not None, "the adjusted observational estimate must be preserved"

    iv = _intervention(demo_bundle, "screw_speed_rpm")
    assert iv is not None, "screw_speed_rpm must not silently disappear from interventions"
    assert iv["status"] != "eligible"
    assert iv["rank"] == 0
    assert iv["status_reason"]
    assert iv["feature"] not in {r["feature"] for r in demo_bundle["interventions"] if r["status"] == "eligible"}


def test_unsupported_levers_include_a_human_readable_reason(demo_bundle):
    for iv in demo_bundle["interventions"]:
        if iv["status"] != "eligible":
            assert iv["status_reason"] and len(iv["status_reason"]) > 10


def test_no_client_side_filter_would_need_to_hide_a_status_record(demo_bundle):
    """
    Every intervention the API returns has a status the frontend's
    InterventionStatus union already renders (see apps/web/lib/types.ts) — the
    UI never needs to drop a record to avoid an unrecognised value.
    """
    known = {"eligible", "exploratory", "unsupported", "infeasible", "conflicting_evidence"}
    for iv in demo_bundle["interventions"]:
        assert iv["status"] in known


def test_intervention_engine_never_drops_a_lever_that_improves_in_neither_direction():
    """
    Unit-level regression for the root cause: a lever whose predictive
    simulation finds no improving direction in either sign used to be dropped
    via a silent `continue`. It must now come back with an explicit
    'unsupported' status.
    """
    from app.models.intervention import run_intervention_engine

    rng = np.random.default_rng(7)
    n = 200
    # A lever with a strong QUADRATIC relationship to the outcome: both a +1SD
    # and a -1SD shift from the mean move the outcome the same (wrong) way,
    # so neither direction the engine tries can improve a "decrease" goal.
    lever = rng.normal(0, 1, n)
    outcome = lever**2 + rng.normal(0, 0.1, n)
    df = pd.DataFrame({"lever": lever, "outcome": outcome})

    interventions = run_intervention_engine(
        df=df, target="outcome", feature_names=["lever"], controllable=["lever"],
        causal_effects=[], improve_direction="decrease", random_seed=42,
    )
    assert len(interventions) == 1, "the lever must not vanish"
    assert interventions[0].feature == "lever"
    assert interventions[0].status != "eligible"
    assert interventions[0].status_reason


# ── Correction 3: screw-speed documentation ──────────────────────────────────

def test_screw_speed_discrepancy_is_not_explained_by_subsampling():
    """
    Reproduces the independent review's finding directly: the full 5,000-row
    dataset gives essentially the same negative, significant estimate as the
    2,000-row samples, so sub-sampling is not the driver of the discrepancy
    with the source study (docs/implementation/PHASE_1A_TRUTH_IN_LABELLING.md
    §8.1).
    """
    from pathlib import Path
    from app.models.causal import run_causal_analysis
    from app.schemas import DagEdge
    from app.ontology import INJECTION_MOULDING_ONTOLOGY as onto

    demo_csv = Path(__file__).resolve().parents[3] / "apps/web/public/demo/injection_molding_demo.csv"
    if not demo_csv.exists():
        pytest.skip("demo CSV not found")
    df_full = pd.read_csv(demo_csv)

    roles = onto.column_roles()
    confounders = [c for c, r in roles.items() if r == "confounder"]
    context = [c for c, r in roles.items() if r == "context"]
    mediators = [c for c, r in roles.items() if r == "mediator"]
    edges = [DagEdge(source=s, target=t) for s, t in onto.edges if s in df_full.columns and t in df_full.columns]
    declared = {"screw_speed_rpm": list(onto.adjustment_sets["screw_speed_rpm"])}

    def beta_pp(df):
        effects, _ = run_causal_analysis(
            df=df, target=onto.target, controllable=["screw_speed_rpm"],
            confounders=confounders, mediators=mediators, context=context,
            dag_edges=edges, declared_adjustment_sets=declared,
        )
        e = effects[0]
        return e.effect_per_std * df[onto.target].std(), e.interval_excludes_zero

    beta_2k, excl_2k = beta_pp(df_full.sample(2000, random_state=42))
    beta_5k, excl_5k = beta_pp(df_full)

    assert excl_2k and excl_5k, "both should be significant on this declared set"
    assert beta_2k < 0 and beta_5k < 0, "sign must agree between sample and full data"
    # If sub-sampling explained the gap, the full-data estimate would be
    # materially different (closer to the source's near-zero published value).
    # It is not: it stays in the same narrow, negative band.
    assert abs(beta_5k - beta_2k) < 0.03, (
        f"full-data ({beta_5k:.3f}) and 2,000-row ({beta_2k:.3f}) estimates "
        "differ more than sampling noise would explain"
    )


def test_screw_speed_rpm_adjustment_set_was_not_changed_to_add_product_variant():
    """
    The adjustment set must stay a faithful port of the source study's own
    declared backdoor set — product_variant was investigated and deliberately
    not added (see docs/implementation/PHASE_1A_TRUTH_IN_LABELLING.md §8.1 and
    §12.3): it enters the source's number as a fixed effect, not as a
    DAG-justified adjuster for this lever.
    """
    from app.ontology import INJECTION_MOULDING_ONTOLOGY as onto
    assert "product_variant" not in onto.adjustment_sets["screw_speed_rpm"]


def test_screw_speed_rpm_is_declared_conflicting_evidence():
    from app.ontology import INJECTION_MOULDING_ONTOLOGY as onto
    spec = onto.spec("screw_speed_rpm")
    assert spec.evidence_status == "conflicting"
