"""SHAP explanations for the final operational XGBoost configuration.

Uses XGBoost's built-in TreeSHAP (``Booster.predict(pred_contribs=True)``),
which computes exact Shapley values for tree ensembles (Lundberg et al.,
2020) - the same algorithm as ``shap.TreeExplainer`` - without adding a
dependency.

What is explained: the model fitted for the test period (all data up to the
first test origin, operational scenario), evaluated on a random sample of
its own recent TRAINING rows (the last 60 days before the test period). Test
rows are not used, so the explanation describes what the model learned, not
how it behaved on the evaluation days.

Interpretation limits: SHAP attributes a prediction to the model's inputs.
It does not identify causes of energy use, and correlated inputs (e.g.
rolling means and recent lags) share credit in ways that depend on the tree
structure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_LABELS = {
    "y_t-0": "Energy at origin hour (t0)",
    "y_t-1": "Energy t0 - 1 h",
    "y_t-2": "Energy t0 - 2 h",
    "y_t-23": "Energy t0 - 23 h",
    "y_t-167": "Energy t0 - 167 h",
    "roll_mean_3": "Mean, last 3 h",
    "roll_std_3": "Std dev, last 3 h",
    "roll_mean_24": "Mean, last 24 h",
    "roll_std_24": "Std dev, last 24 h",
    "roll_mean_168": "Mean, last 7 days",
    "roll_std_168": "Std dev, last 7 days",
    "y_same_hour_1d": "Same hour yesterday",
    "y_same_hour_1w": "Same hour last week",
    "hour": "Hour of day",
    "hour_sin": "Hour of day (sin)",
    "hour_cos": "Hour of day (cos)",
    "dow": "Day of week",
    "dow_sin": "Day of week (sin)",
    "dow_cos": "Day of week (cos)",
    "is_weekend": "Weekend flag",
    "Appliances_lag1": "Energy 1 h before",
    "Appliances_lag2": "Energy 2 h before",
    "Appliances_lag3": "Energy 3 h before",
    "Appliances_lag24": "Energy 24 h before",
    "Appliances_lag48": "Energy 48 h before",
    "Appliances_lag168": "Energy 168 h before",
    "Appliances_rollmean3": "Mean, previous 3 h",
    "Appliances_rollstd3": "Std dev, previous 3 h",
    "Appliances_rollmean6": "Mean, previous 6 h",
    "Appliances_rollstd6": "Std dev, previous 6 h",
    "Appliances_rollmean24": "Mean, previous 24 h",
    "Appliances_rollstd24": "Std dev, previous 24 h",
}


def label(f):
    return FEATURE_LABELS.get(f, f)


def tree_shap(model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (shap values n x p, base values n). Checks additivity."""
    import xgboost as xgb
    booster = model.get_booster()
    dm = xgb.DMatrix(X, feature_names=list(X.columns))
    contrib = booster.predict(dm, pred_contribs=True)
    shap, base = contrib[:, :-1], contrib[:, -1]
    margin = booster.predict(dm, output_margin=True)
    if not np.allclose(shap.sum(1) + base, margin, atol=1e-2, rtol=1e-4):
        raise AssertionError("TreeSHAP additivity check failed")
    return shap, base


def _sample(rows: pd.DataFrame, cutoff, recent_days, n, seed):
    recent = rows[rows.index > cutoff - pd.Timedelta(days=recent_days)]
    return recent.sample(min(n, len(recent)), random_state=seed).sort_index()


def explain_direct(model, history: pd.DataFrame, n_sample=400, recent_days=60,
                   seed=42) -> dict:
    """Per-horizon mean |SHAP| for a fitted ``XGBoostDirect``."""
    from . import config
    y = history[config.TARGET].astype(float)
    cutoff = history.index[-1]
    imp, samples = {}, {}
    for h, m in model.models.items():
        rows = model._rows(y, h, None).dropna()
        X = _sample(rows, cutoff, recent_days, n_sample, seed)[model.feature_cols]
        shap, _ = tree_shap(m, X)
        imp[h] = np.abs(shap).mean(0)
        if h in (1, 12, 24):
            samples[h] = (X, shap)
    by_h = pd.DataFrame(imp, index=model.feature_cols)       # feature x horizon
    overall = by_h.mean(1).sort_values(ascending=False)
    return {"overall": overall, "by_horizon": by_h, "samples": samples,
            "n_sample": n_sample, "recent_days": recent_days, "kind": "direct"}


def explain_recursive(model, history: pd.DataFrame, n_sample=400, recent_days=60,
                      seed=42) -> dict:
    rows = model._training_frame(history)
    X = _sample(rows, history.index[-1], recent_days, n_sample, seed)[model.feature_cols]
    shap, _ = tree_shap(model.model, X)
    overall = pd.Series(np.abs(shap).mean(0), index=model.feature_cols)
    return {"overall": overall.sort_values(ascending=False), "by_horizon": None,
            "samples": {1: (X, shap)}, "n_sample": n_sample,
            "recent_days": recent_days, "kind": "recursive"}
