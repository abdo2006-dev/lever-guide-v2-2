"""
Configuration validation, optional-model failure, and descriptive mode.

Covers the generic-upload path: a fresh CSV must not be able to submit an
implicit causal question, and a user must be able to explore data without
pretending one was specified.
"""
from __future__ import annotations

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


@pytest.fixture
def uploaded_csv() -> str:
    rng = np.random.default_rng(11)
    n = 200
    pressure = rng.normal(100, 8, n)
    humidity = rng.normal(60, 5, n)
    return pd.DataFrame({
        "pressure": pressure,
        "humidity": humidity,
        "batch_id": [f"b-{i}" for i in range(n)],
        "scrap_rate": 0.7 * pressure + 0.3 * humidity + rng.normal(0, 1, n),
    }).to_csv(index=False)


def _payload(csv: str, **over) -> dict:
    base = {
        "dataset_csv": csv,
        "dataset_name": "Uploaded",
        "target": "scrap_rate",
        "task": "regression",
        "improve_direction": "decrease",
        "analysis_mode": "causal",
        "column_roles": {},
        "dag_edges": [],
        "random_seed": 42,
    }
    base.update(over)
    return base


# ── Fresh upload must not submit an implicit causal question ────────────────

def test_fresh_upload_in_causal_mode_is_rejected_with_a_remedy(client, uploaded_csv):
    """
    Previously every unlabelled numeric column silently became a confounder, so
    the app asserted a causal role the user never gave — and then 422'd anyway
    because nothing was controllable. The rejection must now explain what is
    missing and how to fix it.
    """
    resp = client.post("/api/analyze", json=_payload(uploaded_csv))
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "INVALID_ANALYSIS_CONFIGURATION"
    codes = {p["code"] for p in detail["problems"]}
    assert "NO_TREATMENT_SELECTED" in codes
    for p in detail["problems"]:
        assert p["message"] and p["remedy"]
    remedies = " ".join(p["remedy"] for p in detail["problems"]).lower()
    assert "controllable" in remedies
    assert "descriptive" in remedies


def test_unlabelled_columns_are_never_treated_as_confounders(client, uploaded_csv):
    """
    With a treatment named but the rest left alone, the unlabelled column must
    be a predictor and must not appear in any adjustment set.
    """
    resp = client.post("/api/analyze", json=_payload(
        uploaded_csv,
        column_roles={"pressure": "controllable", "batch_id": "identifier"},
    ))
    assert resp.status_code == 422, "no adjuster was named; this should be caught"
    codes = {p["code"] for p in resp.json()["detail"]["problems"]}
    assert "NO_ADJUSTERS_SELECTED" in codes


def test_declaring_roles_makes_the_causal_request_valid(client, uploaded_csv):
    resp = client.post("/api/analyze", json=_payload(
        uploaded_csv,
        column_roles={
            "pressure": "controllable",
            "humidity": "confounder",
            "batch_id": "identifier",
        },
    ))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["causal"]
    assert data["provenance"]["column_roles"]["humidity"] == "confounder"


def test_unassigned_columns_are_predictors_but_never_adjusters(client, uploaded_csv):
    csv = uploaded_csv.replace("humidity", "mystery_metric")
    resp = client.post("/api/analyze", json=_payload(
        csv,
        column_roles={
            "pressure": "controllable",
            "batch_id": "identifier",
            "mystery_metric": "unassigned",
            # A real adjuster so the request is otherwise complete.
            "pressure_dup": "confounder",
        },
    ))
    # No pressure_dup column exists, so there is still no adjuster.
    assert resp.status_code == 422
    codes = {p["code"] for p in resp.json()["detail"]["problems"]}
    assert "NO_ADJUSTERS_SELECTED" in codes


def test_unassigned_column_is_used_as_a_predictor_only(client):
    rng = np.random.default_rng(5)
    n = 200
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    extra = rng.normal(0, 1, n)
    df = pd.DataFrame({
        "lever": a, "adjuster": b, "extra": extra,
        "kpi": a + b + extra + rng.normal(0, 0.2, n),
    })
    resp = client.post("/api/analyze", json=_payload(
        df.to_csv(index=False),
        target="kpi",
        column_roles={"lever": "controllable", "adjuster": "confounder"},
    ))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["provenance"]["column_roles"]["extra"] == "unassigned"
    for e in data["causal"]:
        assert "extra" not in e["adjusted_for"], (
            "an unassigned column entered an adjustment set"
        )
    assert any("no causal role" in w for w in data["warnings"])


# ── Descriptive / predictive mode ────────────────────────────────────────────

def test_descriptive_mode_needs_no_treatment_and_makes_no_causal_claims(client, uploaded_csv):
    resp = client.post("/api/analyze", json=_payload(
        uploaded_csv, analysis_mode="descriptive_predictive",
    ))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["analysis_mode"] == "descriptive_predictive"
    assert data["causal"] == []
    assert data["interventions"] == []
    assert data["predictive"], "predictive results should still be produced"
    assert data["correlations"], "descriptive output should still be produced"
    assert {c["result_type"] for c in data["correlations"]} == {"association"}
    assert "no causal question" in data["executive"]["sub_headline"].lower()


def test_descriptive_mode_provenance_says_so(client, uploaded_csv):
    resp = client.post("/api/analyze", json=_payload(
        uploaded_csv, analysis_mode="descriptive_predictive",
    ))
    p = resp.json()["provenance"]
    assert p["analysis_mode"] == "descriptive_predictive"
    assert p["effect_interval_method"] is None
    assert p["simulation_model"] is None


# ── Optional model failure ───────────────────────────────────────────────────

def test_unimportable_boosters_are_reported_not_swallowed(monkeypatch):
    """
    xgboost and lightgbm need a system OpenMP runtime that is not present
    everywhere. When they cannot be imported the pipeline must say so per model
    rather than silently returning three results while the UI claims five.
    """
    from app.models import pipeline as pipe
    from app.utils.preprocess import build_feature_matrix

    monkeypatch.setattr(pipe, "xgb", None)
    monkeypatch.setattr(pipe, "lgb", None)
    monkeypatch.setattr(pipe, "XGB_IMPORT_ERROR", "libomp.dylib not found")
    monkeypatch.setattr(pipe, "LGBM_IMPORT_ERROR", "libomp.dylib not found")

    rng = np.random.default_rng(3)
    n = 200
    df = pd.DataFrame({
        "a": rng.normal(size=n), "b": rng.normal(size=n),
        "y": rng.normal(size=n),
    })
    X, y, names, _ = build_feature_matrix(df, ["a", "b"], "y")
    results, statuses = pipe.run_predictive_pipeline(X, y, names, run_cv=False)

    by_model = {s.model: s for s in statuses}
    assert set(by_model) == {"ols", "ridge", "rf", "xgb", "lgbm"}
    for key in ("xgb", "lgbm"):
        assert by_model[key].status == "unavailable_dependency"
        assert "libomp" in by_model[key].detail
    assert {r.model for r in results} == {"ols", "ridge", "rf"}
    assert len(results) < len(statuses), "failed models must not be omitted from status"


def test_training_failure_is_distinguished_from_a_missing_dependency(monkeypatch):
    from app.models import pipeline as pipe
    from app.utils.preprocess import build_feature_matrix

    class ExplodingRF:
        def __init__(self, *a, **k):
            pass

        def fit(self, *a, **k):
            raise ValueError("synthetic training failure")

    monkeypatch.setattr(pipe, "RandomForestRegressor", ExplodingRF)

    rng = np.random.default_rng(4)
    n = 150
    df = pd.DataFrame({"a": rng.normal(size=n), "y": rng.normal(size=n)})
    X, y, names, _ = build_feature_matrix(df, ["a"], "y")
    _, statuses = pipe.run_predictive_pipeline(X, y, names, run_cv=False)

    rf = next(s for s in statuses if s.model == "rf")
    assert rf.status == "training_failed"
    assert "synthetic training failure" in rf.detail


def test_optional_boosters_are_not_required_for_the_endpoint(client, uploaded_csv, monkeypatch):
    from app.models import pipeline as pipe
    monkeypatch.setattr(pipe, "xgb", None)
    monkeypatch.setattr(pipe, "lgb", None)

    resp = client.post("/api/analyze", json=_payload(
        uploaded_csv,
        column_roles={
            "pressure": "controllable",
            "humidity": "confounder",
            "batch_id": "identifier",
        },
    ))
    assert resp.status_code == 200, resp.text
    statuses = {s["model"]: s["status"] for s in resp.json()["model_statuses"]}
    assert statuses["xgb"] == "unavailable_dependency"
    assert statuses["lgbm"] == "unavailable_dependency"


# ── Planning levers ──────────────────────────────────────────────────────────

def test_planning_lever_is_a_valid_treatment(client):
    rng = np.random.default_rng(9)
    n = 250
    days = rng.integers(1, 40, n).astype(float)
    ambient = rng.normal(22, 3, n)
    df = pd.DataFrame({
        "days_since_service": days,
        "ambient_temperature_c": ambient,
        "kpi": 0.05 * days + 0.1 * ambient + rng.normal(0, 0.5, n),
    })
    resp = client.post("/api/analyze", json=_payload(
        df.to_csv(index=False),
        target="kpi",
        improve_direction="decrease",
        column_roles={
            "days_since_service": "planning_lever",
            "ambient_temperature_c": "confounder",
        },
    ))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert any(e["feature"] == "days_since_service" for e in data["causal"]), (
        "a planning lever must be estimable, not silently dropped"
    )
    assert data["controllable_count"] == 1
