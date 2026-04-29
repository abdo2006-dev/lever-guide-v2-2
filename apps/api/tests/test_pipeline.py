"""
Smoke tests for the LeverGuide API backend.
Run with: pytest apps/api/tests/ -v
"""
import io
import pytest
import pandas as pd
import numpy as np


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


def test_dag_missing_node_detection():
    from app.utils.dag import validate_dag
    from app.schemas import DagEdge
    edges = [DagEdge(source="missing", target="target")]
    result = validate_dag(edges, ["A", "target"], "target", ["A"])
    assert not result.valid
    assert any("unknown column" in e.lower() for e in result.errors)


def test_dag_helpers_handle_missing_nodes():
    from app.utils.dag import adjustment_set, ancestors_of, build_dag, descendants_of, parents_of
    from app.schemas import DagEdge
    graph = build_dag([DagEdge(source="A", target="B")])
    assert parents_of("missing", graph) == set()
    assert ancestors_of("missing", graph) == set()
    assert descendants_of("missing", graph) == set()
    assert adjustment_set("missing", "B", graph, ["A"], [], []) == {"A"}


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


def test_build_feature_matrix_drops_all_missing_columns(sample_df):
    from app.utils.preprocess import build_feature_matrix
    sample_df["all_missing"] = np.nan
    X, y, names, _ = build_feature_matrix(
        sample_df, ["all_missing", "pressure", "shift"], "scrap_rate"
    )
    assert "all_missing" not in names
    assert X.shape[1] >= 2


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


def _analysis_payload(csv_content: str, dag_edges: list[dict] | None = None) -> dict:
    return {
        "dataset_csv": csv_content,
        "dataset_name": "Test Dataset",
        "target": "scrap_rate_pct",
        "task": "regression",
        "improve_direction": "decrease",
        "column_roles": {
            "injection_pressure_bar": "controllable",
            "barrel_temperature_c": "controllable",
            "ambient_humidity_pct": "confounder",
        },
        "dag_edges": dag_edges or [],
        "random_seed": 42,
    }


def test_analyze_rejects_cyclic_dag(demo_csv_content):
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    payload = _analysis_payload(demo_csv_content, [
        {"source": "injection_pressure_bar", "target": "barrel_temperature_c"},
        {"source": "barrel_temperature_c", "target": "injection_pressure_bar"},
    ])
    resp = client.post("/api/analyze", json=payload)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "INVALID_DAG"
    assert any("cycles" in e.lower() for e in detail["errors"])


def test_analyze_rejects_missing_dag_node(demo_csv_content):
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    payload = _analysis_payload(demo_csv_content, [
        {"source": "not_a_column", "target": "scrap_rate_pct"},
    ])
    resp = client.post("/api/analyze", json=payload)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "INVALID_DAG"
    assert any("unknown column" in e.lower() for e in detail["errors"])


def test_analyze_rejects_malformed_dag_edge(demo_csv_content):
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    payload = _analysis_payload(demo_csv_content, [
        {"source": "", "target": "scrap_rate_pct"},
    ])
    resp = client.post("/api/analyze", json=payload)
    assert resp.status_code == 422


def test_analyze_accepts_valid_dag(demo_csv_content):
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    payload = _analysis_payload(demo_csv_content, [
        {"source": "ambient_humidity_pct", "target": "injection_pressure_bar"},
        {"source": "ambient_humidity_pct", "target": "scrap_rate_pct"},
        {"source": "injection_pressure_bar", "target": "scrap_rate_pct"},
        {"source": "barrel_temperature_c", "target": "scrap_rate_pct"},
    ])
    resp = client.post("/api/analyze", json=payload)
    assert resp.status_code == 200, resp.text
    assert resp.json()["dag_validation"]["valid"] is True


def test_analyze_rejects_non_numeric_target():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    df = pd.DataFrame({
        "feature": np.arange(60),
        "target": [f"label_{i}" for i in range(60)],
    })
    resp = client.post("/api/analyze", json={
        "dataset_csv": df.to_csv(index=False),
        "dataset_name": "non-numeric-target",
        "target": "target",
        "task": "regression",
        "improve_direction": "decrease",
        "column_roles": {"feature": "controllable"},
        "dag_edges": [],
        "random_seed": 42,
    })
    assert resp.status_code == 422
    assert "must contain at least 30 numeric" in resp.json()["detail"]


def test_analyze_rejects_constant_target():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    df = pd.DataFrame({
        "feature": np.random.randn(80),
        "target": [5.0] * 80,
    })
    resp = client.post("/api/analyze", json={
        "dataset_csv": df.to_csv(index=False),
        "dataset_name": "constant-target",
        "target": "target",
        "task": "regression",
        "improve_direction": "decrease",
        "column_roles": {"feature": "controllable"},
        "dag_edges": [],
        "random_seed": 42,
    })
    assert resp.status_code == 422
    assert "must vary across rows" in resp.json()["detail"]


def test_random_mixed_dataset_happy_path():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    rng = np.random.default_rng(7)
    n = 180
    pressure = rng.normal(100, 8, n)
    humidity = rng.normal(60, 5, n)
    material = rng.choice(["ABS", "PET", "PP"], n)
    batch_id = [f"batch-{i}" for i in range(n)]
    target = 0.7 * pressure + 0.3 * humidity + rng.normal(0, 1, n)
    df = pd.DataFrame({
        "pressure": pressure,
        "humidity": humidity,
        "material": material,
        "batch_id": batch_id,
        "all_missing": [np.nan] * n,
        "scrap_rate": target,
    })
    resp = client.post("/api/analyze", json={
        "dataset_csv": df.to_csv(index=False),
        "dataset_name": "random-mixed",
        "target": "scrap_rate",
        "task": "regression",
        "improve_direction": "decrease",
        "column_roles": {
            "pressure": "controllable",
            "humidity": "confounder",
            "material": "context",
            "batch_id": "identifier",
            "all_missing": "controllable",
        },
        "dag_edges": [],
        "random_seed": 42,
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["target"] == "scrap_rate"
    assert data["predictive"]


def test_copilot_retrieves_indexed_analysis_without_llm(demo_csv_content):
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    analyze_resp = client.post("/api/analyze", json=_analysis_payload(demo_csv_content))
    assert analyze_resp.status_code == 200, analyze_resp.text
    analysis_id = analyze_resp.json()["request_id"]

    ask_resp = client.post("/api/copilot/ask", json={
        "analysis_id": analysis_id,
        "question": "Which variables are the top intervention recommendations?",
    })
    assert ask_resp.status_code == 200, ask_resp.text
    data = ask_resp.json()
    assert data["used_llm"] is False
    assert len(data["citations"]) >= 1
    assert data["retrieved_artifact_ids"]


def test_copilot_qdrant_persists_after_client_restart(demo_csv_content):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.rag import close_retrieval_store

    client = TestClient(app)
    analyze_resp = client.post("/api/analyze", json=_analysis_payload(demo_csv_content))
    assert analyze_resp.status_code == 200, analyze_resp.text
    analysis_id = analyze_resp.json()["request_id"]

    close_retrieval_store()

    ask_resp = client.post("/api/copilot/ask", json={
        "analysis_id": analysis_id,
        "question": "What is the best model for this run?",
    })
    assert ask_resp.status_code == 200, ask_resp.text
    data = ask_resp.json()
    assert data["citations"]
    assert data["used_llm"] is False


def test_copilot_unknown_analysis_returns_404():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    resp = client.post("/api/copilot/ask", json={
        "analysis_id": "missing",
        "question": "What is the best model?",
    })
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "ANALYSIS_NOT_INDEXED"


def test_health_endpoint():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
