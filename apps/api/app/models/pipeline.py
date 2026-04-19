"""
Predictive modelling pipeline.
Fits OLS, Ridge, RandomForest, XGBoost, LightGBM; returns metrics and
feature importances. All models use the same train/test split for
fair comparison.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
import statsmodels.api as sm
import xgboost as xgb
import lightgbm as lgb
from app.schemas import (
    PredictiveResult, ModelMetrics, FeatureImportance, PredictionPoint, Coefficient
)


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
        FeatureImportance(
            feature=n,
            importance=float(v),
            importance_norm=float(abs(v) / mx),
        )
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
) -> list[PredictiveResult]:
    """
    Train and evaluate all models. Returns a list of PredictiveResult sorted
    by test R² descending. The best model is flagged is_winner=True.
    """
    n, p = X.shape
    test_size = min(0.2, max(0.1, 200 / n))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )

    results: list[PredictiveResult] = []

    # ── 1. OLS (via statsmodels for inference) ─────────────────────────────
    try:
        Xc_tr = sm.add_constant(X_tr, has_constant="add")
        Xc_te = sm.add_constant(X_te, has_constant="add")
        ols_fit = sm.OLS(y_tr, Xc_tr).fit()
        y_pred_ols = ols_fit.predict(Xc_te)

        m = _metrics(y_te, y_pred_ols, p)
        cv_r2 = None
        if run_cv and n > 100:
            sk_ols = LinearRegression()
            cv_scores = cross_val_score(sk_ols, X, y, cv=5, scoring="r2")
            m["cv_r2_mean"] = float(cv_scores.mean())
            m["cv_r2_std"] = float(cv_scores.std())

        coefs: list[Coefficient] = []
        for i, name in enumerate(["(intercept)"] + feature_names):
            idx = i
            coefs.append(Coefficient(
                feature=name,
                coef=float(ols_fit.params[idx]),
                std_err=float(ols_fit.bse[idx]),
                t_stat=float(ols_fit.tvalues[idx]),
                p_value=float(ols_fit.pvalues[idx]),
                significant=bool(ols_fit.pvalues[idx] < 0.05),
            ))

        imp_vals = np.abs(ols_fit.params[1:])  # skip intercept
        results.append(PredictiveResult(
            model="ols",
            display_name="OLS Regression",
            task=task,
            metrics=ModelMetrics(n_train=len(X_tr), n_test=len(X_te), **m),
            importances=_importance_list(feature_names, imp_vals),
            predictions=[
                PredictionPoint(
                    actual=float(y_te[i]),
                    predicted=float(y_pred_ols[i]),
                    residual=float(y_te[i] - y_pred_ols[i]),
                )
                for i in range(min(len(y_te), 500))
            ],
            coefficients=coefs,
        ))
    except Exception as exc:
        pass  # OLS can fail on rank-deficient matrices

    # ── 2. Ridge ───────────────────────────────────────────────────────────
    try:
        ridge = Ridge(alpha=1.0, random_state=random_seed)
        ridge.fit(X_tr, y_tr)
        y_pred_ridge = ridge.predict(X_te)
        m = _metrics(y_te, y_pred_ridge, p)
        if run_cv and n > 100:
            cv_scores = cross_val_score(ridge, X, y, cv=5, scoring="r2")
            m["cv_r2_mean"] = float(cv_scores.mean())
            m["cv_r2_std"] = float(cv_scores.std())
        results.append(PredictiveResult(
            model="ridge",
            display_name="Ridge Regression",
            task=task,
            metrics=ModelMetrics(n_train=len(X_tr), n_test=len(X_te), **m),
            importances=_importance_list(feature_names, np.abs(ridge.coef_)),
            predictions=[
                PredictionPoint(
                    actual=float(y_te[i]),
                    predicted=float(y_pred_ridge[i]),
                    residual=float(y_te[i] - y_pred_ridge[i]),
                )
                for i in range(min(len(y_te), 500))
            ],
        ))
    except Exception:
        pass

    # ── 3. Random Forest ───────────────────────────────────────────────────
    try:
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=10,
            n_jobs=-1, random_state=random_seed
        )
        rf.fit(X_tr, y_tr)
        y_pred_rf = rf.predict(X_te)
        m = _metrics(y_te, y_pred_rf, p)
        if run_cv and n > 200:
            cv_scores = cross_val_score(rf, X, y, cv=5, scoring="r2")
            m["cv_r2_mean"] = float(cv_scores.mean())
            m["cv_r2_std"] = float(cv_scores.std())
        results.append(PredictiveResult(
            model="rf",
            display_name="Random Forest",
            task=task,
            metrics=ModelMetrics(n_train=len(X_tr), n_test=len(X_te), **m),
            importances=_importance_list(feature_names, rf.feature_importances_),
            predictions=[
                PredictionPoint(
                    actual=float(y_te[i]),
                    predicted=float(y_pred_rf[i]),
                    residual=float(y_te[i] - y_pred_rf[i]),
                )
                for i in range(min(len(y_te), 500))
            ],
        ))
    except Exception:
        pass

    # ── 4. XGBoost ─────────────────────────────────────────────────────────
    try:
        xgb_model = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            random_state=random_seed, verbosity=0, n_jobs=-1
        )
        xgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            verbose=False,
        )
        y_pred_xgb = xgb_model.predict(X_te)
        m = _metrics(y_te, y_pred_xgb, p)
        results.append(PredictiveResult(
            model="xgb",
            display_name="XGBoost",
            task=task,
            metrics=ModelMetrics(n_train=len(X_tr), n_test=len(X_te), **m),
            importances=_importance_list(feature_names, xgb_model.feature_importances_),
            predictions=[
                PredictionPoint(
                    actual=float(y_te[i]),
                    predicted=float(y_pred_xgb[i]),
                    residual=float(y_te[i] - y_pred_xgb[i]),
                )
                for i in range(min(len(y_te), 500))
            ],
        ))
    except Exception:
        pass

    # ── 5. LightGBM ────────────────────────────────────────────────────────
    try:
        lgbm_model = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            num_leaves=31, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            random_state=random_seed, verbosity=-1, n_jobs=-1,
        )
        lgbm_model.fit(X_tr, y_tr)
        y_pred_lgbm = lgbm_model.predict(X_te)
        m = _metrics(y_te, y_pred_lgbm, p)
        results.append(PredictiveResult(
            model="lgbm",
            display_name="LightGBM",
            task=task,
            metrics=ModelMetrics(n_train=len(X_tr), n_test=len(X_te), **m),
            importances=_importance_list(feature_names, lgbm_model.feature_importances_),
            predictions=[
                PredictionPoint(
                    actual=float(y_te[i]),
                    predicted=float(y_pred_lgbm[i]),
                    residual=float(y_te[i] - y_pred_lgbm[i]),
                )
                for i in range(min(len(y_te), 500))
            ],
        ))
    except Exception:
        pass

    if not results:
        raise RuntimeError("All models failed — check your data.")

    # Sort by test R² descending; mark winner
    results.sort(key=lambda r: r.metrics.r2, reverse=True)
    results[0].is_winner = True
    return results
