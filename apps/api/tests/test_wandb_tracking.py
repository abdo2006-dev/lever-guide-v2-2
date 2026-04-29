from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd


def _sample_inputs():
    req = SimpleNamespace(
        dataset_name="demo",
        target="scrap_rate_pct",
        task="regression",
        improve_direction="decrease",
        random_seed=42,
    )
    df = pd.DataFrame(
        {
            "injection_pressure_bar": [1000.0, 1020.0, 1010.0],
            "scrap_rate_pct": [7.1, 6.8, 7.0],
        }
    )
    roles = {
        "injection_pressure_bar": "controllable",
        "scrap_rate_pct": "outcome",
    }
    predictive = [
        SimpleNamespace(
            model="ridge",
            display_name="Ridge",
            is_winner=True,
            metrics={
                "r2": 0.81,
                "adjusted_r2": 0.8,
                "rmse": 0.41,
                "mae": 0.32,
                "cv_r2_mean": 0.78,
                "cv_r2_std": 0.04,
                "train_rows": 2,
                "test_rows": 1,
            },
        )
    ]
    return req, df, roles, predictive


def test_track_analysis_run_disabled_by_default(monkeypatch):
    from app.utils.wandb_tracking import track_analysis_run

    monkeypatch.delenv("WANDB_ENABLED", raising=False)
    req, df, roles, predictive = _sample_inputs()
    warning = track_analysis_run(
        request_id="abc12345",
        req=req,
        df=df,
        roles=roles,
        predictive_results=predictive,
        causal_effects=[],
        interventions=[],
        correlations=[],
        executive={"headline": "demo"},
        runtime_seconds=0.8,
    )
    assert warning is None


def test_track_analysis_run_logs_when_enabled(monkeypatch):
    from app.utils.wandb_tracking import track_analysis_run

    class FakeArtifact:
        def __init__(self, name, type, description):
            self.name = name
            self.type = type
            self.description = description
            self.files: list[str] = []

        def add_file(self, path: str):
            self.files.append(path)

    class FakeRun:
        def __init__(self):
            self.summary = {}
            self.logged = []
            self.artifacts = []
            self.finished = False

        def log(self, payload):
            self.logged.append(payload)

        def log_artifact(self, artifact):
            self.artifacts.append(artifact)

        def finish(self):
            self.finished = True

    class FakeWandb:
        def __init__(self):
            self.run = FakeRun()

        def init(self, **kwargs):
            return self.run

        def Table(self, dataframe):
            return {"rows": len(dataframe)}

        Artifact = FakeArtifact

    fake_wandb = FakeWandb()
    monkeypatch.setenv("WANDB_ENABLED", "true")
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    req, df, roles, predictive = _sample_inputs()
    warning = track_analysis_run(
        request_id="abc12345",
        req=req,
        df=df,
        roles=roles,
        predictive_results=predictive,
        causal_effects=[{"feature": "injection_pressure_bar"}],
        interventions=[{"feature": "injection_pressure_bar"}],
        correlations=[{"feature_a": "a", "feature_b": "b"}],
        executive={"headline": "demo"},
        runtime_seconds=0.8,
    )

    assert warning is None
    assert fake_wandb.run.finished is True
    assert len(fake_wandb.run.logged) >= 1
    assert len(fake_wandb.run.artifacts) == 2
