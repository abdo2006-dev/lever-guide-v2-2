"""
Preprocessing pipeline: robust imputation, encoding, scaling.
Returns sklearn-compatible transformers + transformed DataFrames.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer


MAX_CARDINALITY = 30  # drop categoricals with more unique values than this


def infer_column_kind(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    nunique = series.nunique()
    if nunique <= MAX_CARDINALITY:
        return "categorical"
    return "text"


def build_column_meta(df: pd.DataFrame, roles: dict[str, str]) -> list[dict]:
    """Return per-column statistics for the frontend."""
    metas = []
    for col in df.columns:
        s = df[col]
        kind = infer_column_kind(s)
        role = roles.get(col, "ignore")
        meta: dict = {
            "name": col,
            "kind": kind,
            "role": role,
            "unique": int(s.nunique()),
            "missing": int(s.isna().sum()),
        }
        if kind == "numeric":
            nums = pd.to_numeric(s, errors="coerce")
            meta["min"] = float(nums.min()) if not nums.isna().all() else None
            meta["max"] = float(nums.max()) if not nums.isna().all() else None
            meta["mean"] = float(nums.mean()) if not nums.isna().all() else None
            meta["std"] = float(nums.std()) if not nums.isna().all() else None
            meta["median"] = float(nums.median()) if not nums.isna().all() else None
            meta["p25"] = float(nums.quantile(0.25)) if not nums.isna().all() else None
            meta["p75"] = float(nums.quantile(0.75)) if not nums.isna().all() else None
        else:
            vc = s.value_counts().head(10)
            meta["top_values"] = [
                {"value": str(v), "count": int(c)} for v, c in vc.items()
            ]
        metas.append(meta)
    return metas


def build_feature_matrix(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    standardize: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], ColumnTransformer]:
    """
    Build X, y arrays with proper imputation and encoding.
    Returns (X, y, feature_names_out, transformer).
    """
    df_feat = df[features].copy()
    y = pd.to_numeric(df[target], errors="coerce").values

    # Drop rows where target is missing
    valid_mask = ~np.isnan(y)
    df_feat = df_feat[valid_mask]
    y = y[valid_mask]

    # Separate numeric and categorical columns
    numeric_cols = [
        c for c in features
        if pd.api.types.is_numeric_dtype(df_feat[c])
    ]
    cat_cols = [
        c for c in features
        if not pd.api.types.is_numeric_dtype(df_feat[c])
        and df_feat[c].nunique() <= MAX_CARDINALITY
    ]
    # Drop high-cardinality text features silently
    used_cols = numeric_cols + cat_cols

    if not used_cols:
        raise ValueError("No usable feature columns after preprocessing.")

    df_feat = df_feat[used_cols]

    # Build sklearn transformers
    transformers = []
    if numeric_cols:
        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            *([("scale", StandardScaler())] if standardize else []),
        ])
        transformers.append(("num", num_pipe, numeric_cols))
    if cat_cols:
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
            *([("scale", StandardScaler())] if standardize else []),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    ct = ColumnTransformer(transformers=transformers, remainder="drop")
    X = ct.fit_transform(df_feat)

    # Recover output feature names
    feat_names_out: list[str] = []
    if numeric_cols:
        feat_names_out.extend(numeric_cols)
    if cat_cols:
        feat_names_out.extend(cat_cols)

    return X, y, feat_names_out, ct
