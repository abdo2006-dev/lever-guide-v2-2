"""
Predictive modelling pipeline — tuned for Render free tier (512 MB RAM, ~30s budget).
Five models, 3-fold CV, capped estimators.
"""
from __future__ import annotations
import numpy as np
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
) -> list[PredictiveResult]:
    n, p = X.shape
    test_size = min(0.2, max(0.1, 200 / n))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_seed)
    # 3-fold CV to save time/memory on free tier
    CV_FOLDS = 3

    results: list[PredictiveResult] = []

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
        coefs = [
            Coefficient(feature=nm, coef=float(ols_fit.params[i]),
                        std_err=float(ols_fit.bse[i]), t_stat=float(ols_fit.tvalues[i]),
                        p_value=float(ols_fit.pvalues[i]), significant=bool(ols_fit.pvalues[i] < 0.05))
            for i, nm in enumerate(["(intercept)"] + feature_names)
        ]
        results.append(PredictiveResult(
            model="ols", display_name="OLS Regression", task=task,
            metrics=ModelMetrics(n_train=len(X_tr), n_test=len(X_te), **m),
            importances=_importance_list(feature_names, np.abs(ols_fit.params[1:])),
            predictions=[PredictionPoint(actual=float(y_te[i]), predicted=float(y_pred[i]),
                         residual=float(y_te[i]-y_pred[i])) for i in range(min(len(y_te),400))],
            coefficients=coefs,
        ))
    except Exception:
        pass

    # ── Ridge ─────────────────────────────────────────────────────────────────
    try:
        ridge = Ridge(alpha=1.0, random_state=random_seed)
        ridge.fit(X_tr, y_tr)
        y_pred = ridge.predict(X_te)
        m = _metrics(y_te, y_pred, p)
        if run_cv and n > 100:
            cv = cross_val_score(ridge, X, y, cv=CV_FOLDS, scoring="r2")
            m["cv_r2_mean"] = float(cv.mean()); m["cv_r2_std"] = float(cv.std())
        results.append(PredictiveResult(
            model="ridge", display_name="Ridge Regression", task=task,
            metrics=ModelMetrics(n_train=len(X_tr), n_test=len(X_te), **m),
            importances=_importance_list(feature_names, np.abs(ridge.coef_)),
            predictions=[PredictionPoint(actual=float(y_te[i]), predicted=float(y_pred[i]),
                         residual=float(y_te[i]-y_pred[i])) for i in range(min(len(y_te),400))],
        ))
    except Exception:
        pass

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
        results.append(PredictiveResult(
            model="rf", display_name="Random Forest", task=task,
            metrics=ModelMetrics(n_train=len(X_tr), n_test=len(X_te), **m),
            importances=_importance_list(feature_names, rf.feature_importances_),
            predictions=[PredictionPoint(actual=float(y_te[i]), predicted=float(y_pred[i]),
                         residual=float(y_te[i]-y_pred[i])) for i in range(min(len(y_te),400))],
        ))
    except Exception:
        pass

    # ── XGBoost ───────────────────────────────────────────────────────────────
    try:
        xgb_m = xgb.XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=4,
                                   subsample=0.8, colsample_bytree=0.8, min_child_weight=15,
                                   random_state=random_seed, verbosity=0, n_jobs=1)
        xgb_m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        y_pred = xgb_m.predict(X_te)
        m = _metrics(y_te, y_pred, p)
        results.append(PredictiveResult(
            model="xgb", display_name="XGBoost", task=task,
            metrics=ModelMetrics(n_train=len(X_tr), n_test=len(X_te), **m),
            importances=_importance_list(feature_names, xgb_m.feature_importances_),
            predictions=[PredictionPoint(actual=float(y_te[i]), predicted=float(y_pred[i]),
                         residual=float(y_te[i]-y_pred[i])) for i in range(min(len(y_te),400))],
        ))
    except Exception:
        pass

    # ── LightGBM ──────────────────────────────────────────────────────────────
    try:
        lgbm_m = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.08, max_depth=4,
                                    num_leaves=20, min_child_samples=20,
                                    subsample=0.8, colsample_bytree=0.8,
                                    random_state=random_seed, verbosity=-1, n_jobs=1)
        lgbm_m.fit(X_tr, y_tr)
        y_pred = lgbm_m.predict(X_te)
        m = _metrics(y_te, y_pred, p)
        results.append(PredictiveResult(
            model="lgbm", display_name="LightGBM", task=task,
            metrics=ModelMetrics(n_train=len(X_tr), n_test=len(X_te), **m),
            importances=_importance_list(feature_names, lgbm_m.feature_importances_),
            predictions=[PredictionPoint(actual=float(y_te[i]), predicted=float(y_pred[i]),
                         residual=float(y_te[i]-y_pred[i])) for i in range(min(len(y_te),400))],
        ))
    except Exception:
        pass

    if not results:
        raise RuntimeError("All models failed — check your data.")

    results.sort(key=lambda r: r.metrics.r2, reverse=True)
    results[0].is_winner = True
    return results
