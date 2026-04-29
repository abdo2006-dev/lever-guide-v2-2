from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from typing import Any

import pandas as pd


def _as_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {k: _as_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_serializable(v) for v in value]
    return value


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def track_analysis_run(
    *,
    request_id: str,
    req: Any,
    df: pd.DataFrame,
    roles: dict[str, str],
    predictive_results: list[Any],
    causal_effects: list[Any],
    interventions: list[Any],
    correlations: list[Any],
    executive: Any,
    runtime_seconds: float,
) -> str | None:
    """Track analysis artifacts in Weights & Biases if enabled.

    Returns a warning string when tracking fails, otherwise None.
    """
    if not _env_flag("WANDB_ENABLED", default=False):
        return None

    try:
        import wandb  # type: ignore
    except Exception:
        return (
            "Weights & Biases tracking enabled but 'wandb' is not installed. "
            "Install dependencies in apps/api and retry."
        )

    try:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "leverguide-v2"),
            entity=os.environ.get("WANDB_ENTITY") or None,
            mode=os.environ.get("WANDB_MODE", "online"),
            job_type="analysis",
            name=f"analysis-{request_id}",
            config={
                "request_id": request_id,
                "dataset_name": req.dataset_name,
                "target": req.target,
                "task": req.task,
                "improve_direction": req.improve_direction,
                "random_seed": req.random_seed,
                "row_count": int(len(df)),
                "column_count": int(df.shape[1]),
                "roles": roles,
            },
            reinit=True,
        )
    except Exception as exc:
        return f"Weights & Biases init failed: {exc}"

    try:
        dataset_profile = {
            "dataset_name": req.dataset_name,
            "shape": {"rows": int(len(df)), "columns": int(df.shape[1])},
            "missing_by_column": {
                col: int(count) for col, count in df.isna().sum().to_dict().items()
            },
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
            "roles": roles,
        }

        model_rows: list[dict[str, Any]] = []
        for result in predictive_results:
            metrics = _as_serializable(result.metrics)
            model_rows.append(
                {
                    "model": result.model,
                    "display_name": result.display_name,
                    "is_winner": result.is_winner,
                    "r2": metrics.get("r2"),
                    "adjusted_r2": metrics.get("adjusted_r2"),
                    "rmse": metrics.get("rmse"),
                    "mae": metrics.get("mae"),
                    "cv_r2_mean": metrics.get("cv_r2_mean"),
                    "cv_r2_std": metrics.get("cv_r2_std"),
                    "train_rows": metrics.get("train_rows"),
                    "test_rows": metrics.get("test_rows"),
                }
            )

        best = next((row for row in model_rows if row["is_winner"]), None)
        if best:
            run.summary["best_model"] = best["model"]
            run.summary["best_r2"] = best["r2"]
            run.summary["best_rmse"] = best["rmse"]
            run.summary["best_mae"] = best["mae"]

        run.log(
            {
                "runtime_seconds": runtime_seconds,
                "dataset/rows": int(len(df)),
                "dataset/columns": int(df.shape[1]),
                "metrics/model_count": len(model_rows),
                "metrics/causal_effect_count": len(causal_effects),
                "metrics/intervention_count": len(interventions),
                "models/metrics_table": wandb.Table(dataframe=pd.DataFrame(model_rows)),
            }
        )

        with tempfile.TemporaryDirectory(prefix="leverguide_wandb_") as tmp_dir:
            dataset_path = os.path.join(tmp_dir, "dataset_profile.json")
            with open(dataset_path, "w", encoding="utf-8") as handle:
                json.dump(dataset_profile, handle, indent=2)

            artifacts_payload = {
                "request_id": request_id,
                "executive": _as_serializable(executive),
                "predictive": [_as_serializable(p) for p in predictive_results],
                "causal": [_as_serializable(c) for c in causal_effects],
                "interventions": [_as_serializable(i) for i in interventions],
                "correlations_top50": [_as_serializable(c) for c in correlations[:50]],
            }
            artifacts_path = os.path.join(tmp_dir, "analysis_artifacts.json")
            with open(artifacts_path, "w", encoding="utf-8") as handle:
                json.dump(artifacts_payload, handle, indent=2)

            dataset_artifact = wandb.Artifact(
                name=f"dataset-{request_id}",
                type="dataset",
                description="Dataset profile used for LeverGuide analysis run.",
            )
            dataset_artifact.add_file(dataset_path)
            run.log_artifact(dataset_artifact)

            analysis_artifact = wandb.Artifact(
                name=f"analysis-{request_id}",
                type="analysis",
                description="Model, causal, intervention, and executive analysis outputs.",
            )
            analysis_artifact.add_file(artifacts_path)
            run.log_artifact(analysis_artifact)
    except Exception as exc:
        return f"Weights & Biases logging failed: {exc}"
    finally:
        try:
            run.finish()
        except Exception:
            pass

    return None
