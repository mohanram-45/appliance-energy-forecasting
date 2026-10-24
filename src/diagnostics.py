"""Model diagnostics that run on the VALIDATION partition only.

``xgb_recursion_diagnostic`` measures how much error the recursive XGBoost
strategy accumulates over the 24-step horizon. It compares, hour by hour:

* recursive forecast: the real forecast (lags 1-3 and rolling windows are
  filled with the model's own earlier predictions for steps > 1);
* teacher-forced one-step prediction: the same model, but every lag/rolling
  feature uses the TRUE previous hour. This uses future ground truth BY
  DESIGN, is not a forecast, and is never written to any evaluation table.
  It is the "no accumulated error" reference.

Which inputs can be predictions (for H = 24, origin t0, step h):
  lag1/2/3 come from predictions once h > 1/2/3; lag24/48/168 always refer to
  hours <= t0 and are therefore always observed; rolling windows of length w
  contain min(h - 1, w) predicted values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, features as F
from .protocol import CALENDAR_COLS, future_view, history_view

TARGET = config.TARGET


def predicted_input_share(horizon=config.HORIZON, lags=F.LAGS,
                          windows=F.ROLL_WINDOWS) -> pd.DataFrame:
    """Per step: how many of the target-history inputs are predictions."""
    rows = []
    for h in range(1, horizon + 1):
        lag_pred = sum(1 for k in lags if k < h)
        roll_pred = {w: min(h - 1, w) for w in windows}
        rows.append({"step": h, "lags_from_predictions": lag_pred,
                     **{f"roll{w}_predicted_values": v for w, v in roll_pred.items()}})
    return pd.DataFrame(rows)


def xgb_recursion_diagnostic(df, parts, scenario="operational", params=None):
    """Recursive vs teacher-forced error by step on validation origins."""
    from .forecasters import XGBoostRecursive
    origins = parts.validation
    model = XGBoostRecursive(scenario, params=params).fit(history_view(df, origins[0]))
    rows = []
    for o in origins:
        hist, fut = history_view(df, o), future_view(df, o, scenario=scenario)
        rec = model.predict(hist, fut)["yhat"]
        y_all = df.loc[:fut.index[-1], TARGET]            # truth (diagnostic only)
        for h, ts in enumerate(fut.index, 1):
            row = F.target_history_features(y_all.loc[:ts - pd.Timedelta(hours=1)], ts)
            for c in CALENDAR_COLS + model.exog_cols:
                row[c] = fut.at[ts, c]
            x = pd.DataFrame([row], index=[ts])[model.feature_cols]
            tf = float(model.model.predict(x)[0])
            rows.append({"origin": o, "step": h, "y_true": df.at[ts, TARGET],
                         "recursive": float(rec.loc[ts]), "teacher_forced": tf})
    d = pd.DataFrame(rows)
    out = (d.groupby("step")
           .apply(lambda g: pd.Series({
               "RMSE_recursive": np.sqrt(np.mean((g.recursive - g.y_true) ** 2)),
               "RMSE_teacher_forced": np.sqrt(np.mean((g.teacher_forced - g.y_true) ** 2)),
               "mean_recursive": g.recursive.mean(),
               "mean_true": g.y_true.mean(),
               "std_recursive_across_origins": g.recursive.std(ddof=1),
               "std_true_across_origins": g.y_true.std(ddof=1)}))
           .reset_index())
    out = out.merge(predicted_input_share(), on="step")
    return out, d
