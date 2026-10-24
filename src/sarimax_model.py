"""SARIMAX with an AIC grid search (original single-window implementation).

Steps:
  1. Assess stationarity (done in eda.py) and let the grid choose d/D.
  2. Loop over p in [0,6], d in [0,2], q in [0,6] and pick the model with
     the lowest AIC. A daily (m=24) seasonal component is included because
     the EDA shows a strong 24-hour cycle. A small set of outdoor-weather
     exogenous variables is allowed, which turns SARIMA into SARIMAX.
  3. Inspect the model-residual ACF and their distribution.
  4. Forecast the next 24 hours with 95% confidence intervals; report RMSE.

Because a seasonal period of 24 is expensive, the (p,d,q) grid is searched
with a fixed seasonal order and, for the default run, on a capped window of
recent history (config.SARIMAX_TRAIN_DAYS). Both choices are documented in
the report - they trade a little optimality for a pipeline that actually
finishes.
"""
from __future__ import annotations

import itertools
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

from . import config
from .utils import save_fig, score

warnings.filterwarnings("ignore")

# Exogenous variables that could plausibly be known from a weather forecast.
EXOG_COLS = ["T_out", "RH_out"]


def _fit(endog, order, seasonal_order, exog=None, maxiter=50):
    model = SARIMAX(
        endog, exog=exog, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=maxiter, method="lbfgs")


def param_grid(p_range=config.P_RANGE, d_range=config.D_RANGE,
               q_range=config.Q_RANGE):
    """Every (p, d, q) combination the grid search will attempt.

    With the original grid ranges (p 0-6, d 0-2, q 0-6) this is exactly
    7 * 3 * 7 = 147 combinations.
    """
    return list(itertools.product(p_range, d_range, q_range))


def grid_search(endog, exog=None,
                seasonal_order=(1, 0, 1, config.DAILY_SEASON),
                p_range=config.P_RANGE, d_range=config.D_RANGE,
                q_range=config.Q_RANGE, verbose=True):
    """Loop over every (p, d, q), selecting the lowest-AIC model.

    Returns (best_order, best_aic, results_table). Every attempt is recorded,
    including failures, so the saved CSV proves all 147 combinations were
    tried. Columns: p, d, q, seasonal_order, aic, status, error.
    """
    combos = param_grid(p_range, d_range, q_range)
    rows, best = [], (None, np.inf)
    for p, d, q in combos:
        row = {"p": p, "d": d, "q": q,
               "seasonal_order": str(seasonal_order),
               "aic": np.nan, "status": "ok", "error": ""}
        try:
            res = _fit(endog, (p, d, q), seasonal_order, exog)
            aic = float(res.aic)
            row["aic"] = aic
            if not np.isfinite(aic):
                row["status"] = "non_finite_aic"
            elif aic < best[1]:
                best = ((p, d, q), aic)
            if verbose:
                print(f"  ARIMA({p},{d},{q}) AIC={aic:.1f}")
        except Exception as e:
            row["status"] = "failed"
            row["error"] = str(e).replace("\n", " ")[:300]
            if verbose:
                print(f"  ARIMA({p},{d},{q}) failed: {e}")
        rows.append(row)

    table = pd.DataFrame(rows)
    n_attempted = len(table)
    n_ok = int((table["status"] == "ok").sum())
    print(f"[sarimax] attempted {n_attempted} combinations "
          f"({n_ok} succeeded, {n_attempted - n_ok} failed/non-finite)")
    # keep natural (attempt) order in the file, best model floats to the notes
    table = table.sort_values("aic", na_position="last").reset_index(drop=True)
    if best[0] is None:
        raise RuntimeError("SARIMAX grid search: no model converged.")
    return best[0], best[1], table


def residual_diagnostics(res, name="sarimax"):
    """ACF of residuals, histogram + QQ plot, and a Ljung-Box test."""
    resid = pd.Series(res.resid).iloc[max(1, config.DAILY_SEASON):]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_acf(resid, lags=48, ax=axes[0])
    axes[0].set_title("Residual ACF")
    axes[1].hist(resid, bins=40, color="tab:blue", alpha=0.8)
    axes[1].set_title("Residual distribution")
    stats.probplot(resid, dist="norm", plot=axes[2])
    axes[2].set_title("Residual QQ plot")
    save_fig(fig, f"06_{name}_residuals.png")

    lb = acorr_ljungbox(resid, lags=[24], return_df=True)
    return {"ljung_box_stat": float(lb["lb_stat"].iloc[0]),
            "ljung_box_pvalue": float(lb["lb_pvalue"].iloc[0])}


def forecast(res, horizon=config.HORIZON, exog_future=None):
    """Return (mean, lower95, upper95) for the next `horizon` steps."""
    fc = res.get_forecast(steps=horizon, exog=exog_future)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)
    return mean, ci.iloc[:, 0], ci.iloc[:, 1]


def run(df, horizon=config.HORIZON, use_exog=True, fast=False, verbose=False):
    """Legacy end-to-end SARIMAX run. Returns a dict with forecast, CI, metrics, order."""
    config.assert_legacy_write_allowed("sarimax_model.run")
    y = df[config.TARGET].astype(float)
    train_full, test = y.iloc[:-horizon], y.iloc[-horizon:]

    # cap history so the m=24 grid finishes in reasonable time
    train = train_full.iloc[-config.SARIMAX_TRAIN_DAYS * 24:]

    exog = exog_future = None
    if use_exog and all(c in df.columns for c in EXOG_COLS):
        exog = df.loc[train.index, EXOG_COLS]
        exog_future = df.loc[test.index, EXOG_COLS]

    if fast:  # smaller grid for smoke tests
        p_range, d_range, q_range = range(0, 3), range(0, 2), range(0, 3)
        seasonal = (1, 0, 0, config.DAILY_SEASON)
    else:
        p_range, d_range, q_range = config.P_RANGE, config.D_RANGE, config.Q_RANGE
        seasonal = (1, 0, 1, config.DAILY_SEASON)

    order, aic, table = grid_search(
        train, exog, seasonal, p_range, d_range, q_range, verbose=verbose)
    table.to_csv(config.METRIC_DIR / "sarimax_grid.csv", index=False)

    # refit the chosen model on the (capped) training data
    res = _fit(train, order, seasonal, exog, maxiter=200)
    diag = residual_diagnostics(res)
    mean, lo, hi = forecast(res, horizon, exog_future)
    mean = np.clip(mean.values, 0, None)  # appliance use cannot be negative

    # persist the selected model so it is reproducible and easy to cite
    summary = {
        "order": list(order),
        "seasonal_order": list(seasonal),
        "aic": float(aic),
        "n_combinations_attempted": int(len(table)),
        "n_combinations_succeeded": int((table["status"] == "ok").sum()),
        "ljung_box_stat": diag["ljung_box_stat"],
        "ljung_box_pvalue": diag["ljung_box_pvalue"],
        "train_days": config.SARIMAX_TRAIN_DAYS,
        "used_exog": bool(exog is not None),
        "exog_cols": EXOG_COLS if exog is not None else [],
        "conditional_forecast": bool(exog is not None),
    }
    (config.METRIC_DIR / "sarimax_summary.json").write_text(
        json.dumps(summary, indent=2))
    pd.DataFrame([summary]).to_csv(
        config.METRIC_DIR / "sarimax_summary.csv", index=False)

    return {
        "name": "SARIMAX",
        "order": order, "seasonal_order": seasonal, "aic": aic,
        "forecast": pd.Series(mean, index=test.index),
        "lower": pd.Series(np.clip(lo.values, 0, None), index=test.index),
        "upper": pd.Series(hi.values, index=test.index),
        "metrics": score(test.values, mean),
        "diagnostics": diag,
        "summary": summary,
        "train_tail": train_full, "test": test,
    }
