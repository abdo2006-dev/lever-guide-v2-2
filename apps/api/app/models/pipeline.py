"""
Predictive modelling pipeline — tuned for Render free tier (512 MB RAM, ~30s budget).

Up to five models. XGBoost and LightGBM are **optional**: they need a system
OpenMP runtime that is not present in every environment, and the application
starts and runs without them. What is not optional is saying so — every
configured model reports a status, so "three models ran" is never displayed as
"five models ran".
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
import statsmodels.api as sm
from app.schemas import (
    Coefficient, FeatureImportance, ModelMetrics, ModelStatus, PredictionPoint,
    PredictiveResult,
)

# Import failures are recorded, not swallowed: the reason is what the user needs.
try:
    import xgboost as xgb
    XGB_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - environment dependent
    xgb = None  # type: ignore[assignment]
    XGB_IMPORT_ERROR = str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc)

try:
    import lightgbm as lgb
    LGBM_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - environment dependent
    lgb = None  # type: ignore[assignment]
    LGBM_IMPORT_ERROR = str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc)


DISPLAY_NAMES: dict[str, str] = {
    "ols": "OLS Regression",
    "ridge": "Ridge Regression",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
}


def _truncate(detail: str, limit: int = 300) -> str:
    detail = " ".join(detail.split())
    return detail if len(detail) <= limit else detail[: limit - 1] + "…"


def _require_finite_metrics(m: dict) -> None:
    """
    Raise if a metric came out non-finite (e.g. a near-singular OLS fit).

    Caught by the caller's existing `except Exception` per model, so a
    non-finite result is reported as `training_failed` with a reason rather
    than reaching JSON serialisation.
    """
    bad = [k for k, v in m.items() if isinstance(v, (int, float)) and not math.isfinite(v)]
    if bad:
        raise ValueError(f"non-finite metric(s): {', '.join(bad)}")


class _Skipped(Exception):
    """A model whose status has already been recorded; skip without re-recording."""


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> dict:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) or 1e-9
    r2 = float(1 - ss_res / ss_tot)
    n = len(y_true)
    adj_r2 = float(1 - (1 - r2) * (n - 1) / max(n - n_features - 1, 1))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"r2": r2, "adj_r2": adj_r2, "rmse": rmse, "mae": mae}


def _importance_list(names: list[str], values: np.ndarray) -> list[FeatureImportance]:
    mx = float(max(abs(values))) or 1e-9
    out = [
        FeatureImportance(feature=n, importance=float(v), importance_norm=float(abs(v) / mx))
        for n, v in zip(names, values)
    ]
    return sorted(out, key=lambda x: -x.importance_norm)[:20]


def run_predictive_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    task: str = "regression",
    random_seed: int = 42,
    run_cv: bool = True,
) -> tuple[list[PredictiveResult], list[ModelStatus]]:
    """
    Returns (results, statuses).

    `statuses` has one entry per configured model, including the ones that did
    not run and why. A caller that only reads `results` cannot tell a model that
    was never configured from one that crashed, which is the failure this
    two-value return exists to prevent.
    """
    n, p = X.shape
    test_size = min(0.2, max(0.1, 200 / n))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_seed)
    # 3-fold CV to save time/memory on free tier
    CV_FOLDS = 3

    results: list[PredictiveResult] = []
    statuses: list[ModelStatus] = []

    def record(model: str, status: str, detail: str | None = None) -> None:
        statuses.append(ModelStatus(
            model=model,  # type: ignore[arg-type]
            display_name=DISPLAY_NAMES[model],
            status=status,  # type: ignore[arg-type]
            detail=_truncate(detail) if detail else None,
        ))

    # ── OLS ──────────────────────────────────────────────────────────────────
    try:
        Xc_tr = sm.add_constant(X_tr, has_constant="add")
        Xc_te = sm.add_constant(X_te, has_constant="add")
        ols_fit = sm.OLS(y_tr, Xc_tr).fit()
        y_pred = ols_fit.predict(Xc_te)
        m = _metrics(y_te, y_pred, p)
        if run_cv and n > 100:
            cv = cross_val_score(LinearRegression(), X, y, cv=CV_FOLDS, scoring="r2")
            m["cv_r2_mean"] = float(cv.mean()); m["cv_r2_std"] = float(cv.std())
        _require_finite_metrics(m)
        coefs = [
            Coefficient(feature=nm, coef=float(ols_fit.params[i]),
                        std_err=float(ols_fit.bse[i]), t_stat=float(ols_fit.tvalues[i]),
                        p_value=float(ols_fit.pvalues[i]), significant=bool(ols_fit.pvalues[i] < 0.05))
            for i, nm in enumerate(["(intercept)"] + feature_names)
        ]
        if not all(
            math.isfinite(c.coef) and math.isfinite(c.std_err)
            and math.isfinite(c.t_stat) and math.isfinite(c.p_value)
            for c in coefs
        ):
            raise ValueError("non-finite coefficient(s) — design matrix is degenerate for OLS")
        results.append(PredictiveResult(
            model="ols", display_name="OLS Regression", task=task,
            metrics=ModelMetrics(n_train=len(X_tr), n_test=len(X_te), **m),
            importances=_importance_list(feature_names, np.abs(ols_fit.params[1:])),
            predictions=[PredictionPoint(actual=float(y_te[i]), predicted=float(y_pred[i]),
                         residual=float(y_te[i]-y_pred[i])) for i in range(min(len(y_te),400))],
            coefficients=coefs,
        ))
        record("ols", "succeeded")
    except Exception as exc:
        record("ols", "training_failed", str(exc))

    # ── Ridge ─────────────────────────────────────────────────────────────────
    try:
        ridge = Ridge(alpha=1.0, random_state=random_seed)
        ridge.fit(X_tr, y_tr)
        y_pred = ridge.predict(X_te)
        m = _metrics(y_te, y_pred, p)
        if run_cv and n > 100:
            cv = cross_val_score(ridge, X, y, cv=CV_FOLDS, scoring="r2")
            m["cv_r2_mean"] = float(cv.mean()); m["cv_r2_std"] = float(cv.std())
        _require_finite_metrics(m)
        results.append(PredictiveResult(
            model="ridge", display_name="Ridge Regression", task=task,
            metrics=ModelMetrics(n_train=len(X_tr), n_test=len(X_te), **m),
            importances=_importance_list(feature_names, np.abs(ridge.coef_)),
            predictions=[PredictionPoint(actual=float(y_te[i]), predicted=float(y_pred[i]),
                         residual=float(y_te[i]-y_pred[i])) for i in range(min(len(y_te),400))],
        ))
        record("ridge", "succeeded")
    except Exception as exc:
        record("ridge", "training_failed", str(exc))

    # ── Random Forest ─────────────────────────────────────────────────────────
    try:
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=15,
                                   n_jobs=-1, random_state=random_seed)
        rf.fit(X_tr, y_tr)
        y_pred = rf.predict(X_te)
        m = _metrics(y_te, y_pred, p)
        if run_cv and n > 200:
            cv = cross_val_score(rf, X, y, cv=CV_FOLDS, scoring="r2")
            m["cv_r2_mean"] = float(cv.mean()); m["cv_r2_std"] = float(cv.std())
        _require_finite_metrics(m)
        results.append(PredictiveResult(
            model="rf", display_name="Random Forest", task=task,
            metrics=ModelMetrics(n_train=len(X_tr), n_test=len(X_te), **m),
            importances=_importance_list(feature_names, rf.feature_importances_),
            predictions=[PredictionPoint(actual=float(y_te[i]), predicted=float(y_pred[i]),
                         residual=float(y_te[i]-y_pred[i])) for i in range(min(len(y_te),400))],
        ))
        record("rf", "succeeded")
    except Exception as exc:
        record("rf", "training_failed", str(exc))

    # ── XGBoost (optional) ────────────────────────────────────────────────────
    try:
        if xgb is None:
            record(
                "xgb", "unavailable_dependency",
                XGB_IMPORT_ERROR or "xgboost could not be imported in this environment.",
            )
            raise _Skipped
        xgb_m = xgb.XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=4,
                                   subsample=0.8, colsample_bytree=0.8, min_child_weight=15,
                                   random_state=random_seed, verbosity=0, n_jobs=1)
        # The test set is deliberately not passed to fit(): without early
        # stopping it changes nothing, and it makes held-out data an input to
        # training, which is one flag away from being leakage.
        xgb_m.fit(X_tr, y_tr, verbose=False)
        y_pred = xgb_m.predict(X_te)
        m = _metrics(y_te, y_pred, p)
        _require_finite_metrics(m)
        results.append(PredictiveResult(
            model="xgb", display_name="XGBoost", task=task,
            metrics=ModelMetrics(n_train=len(X_tr), n_test=len(X_te), **m),
            importances=_importance_list(feature_names, xgb_m.feature_importances_),
            predictions=[PredictionPoint(actual=float(y_te[i]), predicted=float(y_pred[i]),
                         residual=float(y_te[i]-y_pred[i])) for i in range(min(len(y_te),400))],
        ))
        record("xgb", "succeeded")
    except _Skipped:
        pass
    except Exception as exc:
        record("xgb", "training_failed", str(exc))

    # ── LightGBM (optional) ───────────────────────────────────────────────────
    try:
        if lgb is None:
            record(
                "lgbm", "unavailable_dependency",
                LGBM_IMPORT_ERROR or "lightgbm could not be imported in this environment.",
            )
            raise _Skipped
        lgbm_m = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.08, max_depth=4,
                                    num_leaves=20, min_child_samples=20,
                                    subsample=0.8, colsample_bytree=0.8,
                                    random_state=random_seed, verbosity=-1, n_jobs=1)
        lgbm_m.fit(X_tr, y_tr)
        y_pred = lgbm_m.predict(X_te)
        m = _metrics(y_te, y_pred, p)
        _require_finite_metrics(m)
        results.append(PredictiveResult(
            model="lgbm", display_name="LightGBM", task=task,
            metrics=ModelMetrics(n_train=len(X_tr), n_test=len(X_te), **m),
            importances=_importance_list(feature_names, lgbm_m.feature_importances_),
            predictions=[PredictionPoint(actual=float(y_te[i]), predicted=float(y_pred[i]),
                         residual=float(y_te[i]-y_pred[i])) for i in range(min(len(y_te),400))],
        ))
        record("lgbm", "succeeded")
    except _Skipped:
        pass
    except Exception as exc:
        record("lgbm", "training_failed", str(exc))

    if not results:
        detail = "; ".join(f"{s.display_name}: {s.status}" for s in statuses)
        raise RuntimeError(f"All models failed — check your data. ({detail})")

    # The winner is selected on the same held-out set its score is reported on,
    # so that score is optimistically biased. Fixing that needs a three-way
    # split or nested CV and is deferred to the analytical-core phase.
    results.sort(key=lambda r: r.metrics.r2, reverse=True)
    results[0].is_winner = True
    statuses.sort(key=lambda s: list(DISPLAY_NAMES).index(s.model))
    return results, statuses
