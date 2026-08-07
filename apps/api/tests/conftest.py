"""Shared fixtures for the API test suite."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO_CSV = REPO_ROOT / "apps/web/public/demo/injection_molding_demo.csv"


@pytest.fixture(scope="session")
def demo_df() -> pd.DataFrame:
    """The shipped 5,000-row injection-moulding demo."""
    if not DEMO_CSV.exists():  # pragma: no cover - only if the demo is removed
        pytest.skip(f"demo CSV not found at {DEMO_CSV}")
    return pd.read_csv(DEMO_CSV)


@pytest.fixture(scope="session")
def demo_csv_text() -> str:
    if not DEMO_CSV.exists():  # pragma: no cover
        pytest.skip(f"demo CSV not found at {DEMO_CSV}")
    return DEMO_CSV.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def demo_bundle(demo_csv_text, tmp_path_factory):
    """
    A full analysis of the shipped demo, computed once for the whole session.

    Uses the ontology's own role assignments, which is what the frontend sends
    when a user clicks "Try Demo Dataset". Session-scoped because the analysis
    takes a couple of seconds and every scientific-safety test inspects the same
    bundle; the tests only read it.
    """
    import os

    from fastapi.testclient import TestClient
    from app.main import app
    from app.ontology import INJECTION_MOULDING_ONTOLOGY as onto

    prev = {k: os.environ.get(k) for k in ("QDRANT_PATH", "QDRANT_URL", "GROQ_API_KEY")}
    os.environ["QDRANT_PATH"] = str(tmp_path_factory.mktemp("qdrant-demo"))
    os.environ["QDRANT_URL"] = ""
    os.environ["GROQ_API_KEY"] = ""

    client = TestClient(app)
    resp = client.post("/api/analyze", json={
        "dataset_csv": demo_csv_text,
        "dataset_name": "Injection Molding Demo",
        "target": onto.target,
        "task": "regression",
        "improve_direction": "decrease",
        "analysis_mode": "causal",
        "column_roles": onto.column_roles(),
        "dag_edges": [],
        "random_seed": 42,
    })
    assert resp.status_code == 200, resp.text
    bundle = resp.json()

    try:
        from app.rag import close_retrieval_store
        close_retrieval_store()
    except Exception:
        pass
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

    return bundle
