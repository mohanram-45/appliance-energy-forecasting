"""Feature-based gradient-boosting model (XGBoost), legacy single-window version.

The model predicts the hourly target from the engineered features. It is
evaluated over the final 14 days (rolling one-step), and a genuine 24-hour
recursive forecast is produced for the last day so it can be compared with
the benchmarks, SARIMAX and the foundation model on the same horizon.

Recursive forecasting: target lag / rolling features for the forecast window
are rebuilt from the model's own predictions, so the target's future values
never leak in. Calendar features are deterministic and weather/sensor
covariates are taken from the test window (a *conditional* forecast - see the
data-leakage discussion in the report).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from . import config
from . import features as F
from .utils import save_fig, score, split_days


def _model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=600, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        random_state=config.RANDOM_STATE, n_jobs=4,
    )


def _recursive_forecast(model, feat_cols, history: pd.DataFrame,
                        exog_future: pd.DataFrame, horizon: int) -> pd.Series:
    """Step through the horizon, feeding predictions back as lag features."""
    hist = history.copy()
    preds = []
    for ts in exog_future.index[:horizon]:
        # append the covariate row for this step, target still unknown
        row = exog_future.loc[[ts]].copy()
        row[config.TARGET] = np.nan
        hist = pd.concat([hist, row])
        featured = F.build_feature_frame(hist, dropna=False)
        x = featured.loc[[ts], feat_cols]
        yhat = float(model.predict(x)[0])
        preds.append(yhat)
        hist.loc[ts, config.TARGET] = yhat  # feed back for the next lag
    return pd.Series(preds, index=exog_future.index[:horizon])


def run(df, horizon=config.HORIZON, test_days=config.ML_TEST_DAYS):
    """Legacy end-to-end XGBoost run. Returns forecast + metrics + feature importances."""
    config.assert_legacy_write_allowed("ml_model.run")
    frame = F.build_feature_frame(df)
    feat_cols = F.feature_columns(frame)

    train, test = split_days(frame, test_days)
    X_train, y_train = train[feat_cols], train[config.TARGET]
    X_test, y_test = test[feat_cols], test[config.TARGET]

    model = _model()
    model.fit(X_train, y_train)

    # (a) rolling one-step performance across the whole 14-day test period
    test_pred = pd.Series(model.predict(X_test), index=y_test.index)
    metrics_14d = score(y_test.values, test_pred.values)

    # (b) recursive 24-hour forecast for the final day (comparable horizon)
    cov_cols = [c for c in df.columns if c != config.TARGET]
    horizon_idx = df.index[-horizon:]
    history = df.loc[df.index < horizon_idx[0]].copy()
    exog_future = df.loc[horizon_idx, cov_cols]
    fc24 = _recursive_forecast(model, feat_cols, history, exog_future, horizon)
    y_true24 = df.loc[horizon_idx, config.TARGET]
    metrics_24h = score(y_true24.values, fc24.values)

    # feature importances by group
    imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values(
        ascending=False)
    imp.to_csv(config.METRIC_DIR / "xgb_feature_importance.csv")
    _plot_importance(imp)
    group_imp = _group_importance(imp, feat_cols)

    return {
        "name": "XGBoost",
        "model": model,
        "forecast": fc24, "test": y_true24,
        "train_tail": df[config.TARGET].loc[df.index < horizon_idx[0]],
        "metrics": metrics_24h, "metrics_14d": metrics_14d,
        "importance": imp, "group_importance": group_imp,
    }


def _plot_importance(imp: pd.Series, top=20):
    fig, ax = plt.subplots(figsize=(8, 7))
    imp.head(top)[::-1].plot.barh(ax=ax, color="tab:green")
    ax.set_title("XGBoost feature importance (top 20)")
    save_fig(fig, "07_xgb_importance.png")


def _group_importance(imp: pd.Series, feat_cols) -> pd.Series:
    groups = F.feature_groups(feat_cols)
    return pd.Series(
        {g: imp[cols].sum() for g, cols in groups.items() if cols}
    ).sort_values(ascending=False)
