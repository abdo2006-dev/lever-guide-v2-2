"""
Smoke tests for the LeverGuide API backend.
Run with: pytest apps/api/tests/ -v
"""
import io
import pytest
import pandas as pd
import numpy as np

# ── Unit: DAG utilities ───────────────────────────────────────────────────────

def test_dag_cycle_detection():
    from app.utils.dag import validate_dag
    from app.schemas import DagEdge
    edges = [
        DagEdge(source="A", target="B"),
        DagEdge(source="B", target="C"),
        DagEdge(source="C", target="A"),  # cycle
    ]
    result = validate_dag(edges, ["A", "B", "C", "target"], "target", ["A"])
    assert not result.valid
    assert any("cycle" in e.lower() for e in result.errors)


def test_dag_valid():
    from app.utils.dag import validate_dag
    from app.schemas import DagEdge
    edges = [
        DagEdge(source="A", target="target"),
        DagEdge(source="B", target="A"),
    ]
    result = validate_dag(edges, ["A", "B", "target"], "target", ["A"])
    assert result.valid


def test_adjustment_set_excludes_mediators():
    from app.utils.dag import adjustment_set, build_dag
    from app.schemas import DagEdge
    edges = [
        DagEdge(source="confounder", target="cause"),
        DagEdge(source="cause", target="mediator"),
        DagEdge(source="mediator", target="outcome"),
    ]
    G = build_dag(edges)
    adj = adjustment_set("cause", "outcome", G, ["confounder"], ["mediator"], [])
    assert "mediator" not in adj
    assert "confounder" in adj


def test_auto_dag_structure():
    from app.utils.dag import auto_dag
    edges = auto_dag(["pressure", "temp"], ["humidity"], ["shift"], "scrap_rate")
    sources = {e.source for e in edges}
    targets = {e.target for e in edges}
    assert "pressure" in sources
    assert "scrap_rate" in targets
    assert "humidity" in sources


# ── Unit: Preprocessing ───────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "pressure":    rng.normal(100, 10, n),
        "temperature": rng.normal(200, 15, n),
        "shift":       rng.choice(["A", "B", "C"], n),
        "scrap_rate":  rng.normal(5, 1, n),
    })


def test_build_feature_matrix_shape(sample_df):
    from app.utils.preprocess import build_feature_matrix
    X, y, names, ct = build_feature_matrix(
        sample_df, ["pressure", "temperature", "shift"], "scrap_rate"
    )
    assert X.shape[0] == len(sample_df)
    assert X.shape[1] >= 3  # 2 numeric + 1 ordinal-encoded categorical
    assert len(y) == len(sample_df)


def test_build_feature_matrix_no_nans(sample_df):
    from app.utils.preprocess import build_feature_matrix
    sample_df.loc[0:10, "pressure"] = np.nan
    X, y, names, ct = build_feature_matrix(
        sample_df, ["pressure", "temperature"], "scrap_rate"
    )
    assert not np.isnan(X).any()
    assert not np.isnan(y).any()


# ── Unit: Predictive pipeline ─────────────────────────────────────────────────

def test_pipeline_returns_results(sample_df):
    from app.utils.preprocess import build_feature_matrix
    from app.models.pipeline import run_predictive_pipeline
    X, y, names, _ = build_feature_matrix(
        sample_df, ["pressure", "temperature"], "scrap_rate"
    )
    results = run_predictive_pipeline(X, y, names, task="regression", run_cv=False)
    assert len(results) >= 1
    winner = [r for r in results if r.is_winner]
    assert len(winner) == 1


def test_pipeline_winner_has_best_r2(sample_df):
    from app.utils.preprocess import build_feature_matrix
    from app.models.pipeline import run_predictive_pipeline
    X, y, names, _ = build_feature_matrix(
        sample_df, ["pressure", "temperature"], "scrap_rate"
    )
    results = run_predictive_pipeline(X, y, names, task="regression", run_cv=False)
    winner = next(r for r in results if r.is_winner)
    for r in results:
        assert winner.metrics.r2 >= r.metrics.r2


# ── Unit: Causal analysis ────────────────────────────────────────────────────

def test_causal_analysis_runs(sample_df):
    from app.models.causal import run_causal_analysis
    from app.schemas import DagEdge
    edges = [
        DagEdge(source="pressure", target="scrap_rate"),
        DagEdge(source="temperature", target="scrap_rate"),
    ]
    effects = run_causal_analysis(
        df=sample_df,
        target="scrap_rate",
        controllable=["pressure", "temperature"],
        confounders=[],
        mediators=[],
        context=[],
        dag_edges=edges,
    )
    assert isinstance(effects, list)


def test_causal_analysis_excludes_mediators():
    from app.models.causal import run_causal_analysis
    from app.schemas import DagEdge
    rng = np.random.default_rng(0)
    n = 200
    pressure = rng.normal(100, 10, n)
    mediator = pressure * 0.5 + rng.normal(0, 1, n)
    outcome = mediator * 2 + rng.normal(0, 1, n)
    df = pd.DataFrame({"pressure": pressure, "mediator": mediator, "outcome": outcome})
    edges = [
        DagEdge(source="pressure", target="mediator"),
        DagEdge(source="mediator", target="outcome"),
    ]
    effects = run_causal_analysis(
        df=df,
        target="outcome",
        controllable=["pressure"],
        confounders=[],
        mediators=["mediator"],
        context=[],
        dag_edges=edges,
    )
    for e in effects:
        assert "mediator" not in e.adjusted_for


# ── Integration: Full analysis endpoint ──────────────────────────────────────

@pytest.fixture
def demo_csv_content():
    rng = np.random.default_rng(42)
    n = 300
    pressure = rng.normal(1100, 50, n)
    temperature = rng.normal(240, 10, n)
    humidity = rng.normal(60, 5, n)
    scrap = 0.02 * pressure + 0.01 * temperature + 0.005 * humidity + rng.normal(5, 0.5, n)
    df = pd.DataFrame({
        "injection_pressure_bar": pressure,
        "barrel_temperature_c": temperature,
        "ambient_humidity_pct": humidity,
        "scrap_rate_pct": scrap,
    })
    return df.to_csv(index=False)


def test_full_analysis_endpoint(demo_csv_content):
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    payload = {
        "dataset_csv": demo_csv_content,
        "dataset_name": "Test Dataset",
        "target": "scrap_rate_pct",
        "task": "regression",
        "improve_direction": "decrease",
        "column_roles": {
            "injection_pressure_bar": "controllable",
            "barrel_temperature_c": "controllable",
            "ambient_humidity_pct": "confounder",
        },
        "dag_edges": [],
        "random_seed": 42,
    }
    resp = client.post("/api/analyze", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["target"] == "scrap_rate_pct"
    assert len(data["predictive"]) >= 1
    assert data["best_model"] in ("ols", "ridge", "rf", "xgb", "lgbm")
    assert "executive" in data


def test_health_endpoint():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
