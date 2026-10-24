"""Leakage-free SARIMAX order selection (Phase 2B).

Why not the legacy procedure
----------------------------
The legacy grid (``sarimax_model.grid_search``) ranked all 147 (p,d,q) models
by AIC. That has three problems:

1. AIC values for different differencing orders d are not comparable: the
   likelihood is evaluated on a differently transformed series with a
   different number of burn-in observations (audit B2).
2. Fits were capped at ``maxiter=50`` with warnings suppressed, so
   non-converged fits were labelled "ok" and ranked (audit B4). On the
   training window, (2,1,2), (3,0,6) and (6,2,6) do not converge in 50
   iterations.
3. The legacy window overlapped what is now the test period.

Procedure used here
-------------------
Stage 1 - within-d AIC, training data only.
    The original grid (p 0-6, d 0-2, q 0-6, seasonal (1,0,1,24), 45-day
    window) is fitted on the 45 days ending at the last TRAINING hour. Every
    fit records convergence, warnings, errors and runtime. For each d
    separately, the ``top_k`` converged models with the lowest AIC become
    candidates. AIC is never compared across different d.

Stage 2 - validation forecasting accuracy, across d.
    Each candidate is fitted on the training partition and evaluated with the
    shared rolling-origin protocol on the VALIDATION origins (never test).
    The candidate with the lowest pooled validation RMSE wins. Candidates
    whose stage-2 fit did not converge are only eligible if none converged.

The test partition is not read at any point. ``train_end`` is the first
validation origin, so the stage-1 window ends before any validation target.
"""
from __future__ import annotations

import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import config

TARGET = config.TARGET


def fit_with_diagnostics(y, order, seasonal_order, exog=None, maxiter=200):
    """Fit one SARIMAX and record what happened instead of hiding it.

    Returns (results_or_None, info). ``info`` has: converged, n_warnings,
    warnings (first 3 messages), error, aic, seconds.
    """
    from . import sarimax_model
    info = {"order": tuple(order), "seasonal_order": tuple(seasonal_order),
            "maxiter": maxiter, "converged": False, "n_warnings": 0,
            "warnings": "", "error": "", "aic": np.nan, "seconds": np.nan}
    t0 = time.perf_counter()
    res = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")   # overrides the legacy global ignore
        try:
            res = sarimax_model._fit(y, tuple(order), tuple(seasonal_order),
                                     exog, maxiter=maxiter)
            rv = res.mle_retvals if isinstance(res.mle_retvals, dict) else {}
            info["converged"] = bool(rv.get("converged", False))
            info["aic"] = float(res.aic)
            if not np.isfinite(info["aic"]) or not np.all(np.isfinite(res.params)):
                info["converged"] = False
                info["error"] = "non-finite AIC or parameters"
        except Exception as e:  # recorded, never silently dropped
            info["error"] = f"{type(e).__name__}: {e}"[:300]
    info["seconds"] = round(time.perf_counter() - t0, 2)
    msgs = [f"{w.category.__name__}: {w.message}" for w in caught]
    info["n_warnings"] = len(msgs)
    info["warnings"] = " | ".join(m[:150] for m in msgs[:3])
    return res, info


def _status(info):
    if info["error"]:
        return "failed"
    return "converged" if info["converged"] else "not_converged"


def _grid_worker(args):
    y, exog, order, seasonal, maxiter = args
    _res, info = fit_with_diagnostics(y, order, seasonal, exog, maxiter)
    info["status"] = _status(info)
    return info


def training_window(df, train_end, exog_cols=None,
                    days=config.SARIMAX_TRAIN_DAYS):
    hist = df.loc[:pd.Timestamp(train_end)]
    w = hist.iloc[-days * 24:]
    y = w[TARGET].astype(float).asfreq(config.FREQ)
    x = w[exog_cols].asfreq(config.FREQ) if exog_cols else None
    return y, x


def stage1_grid(y, exog=None, seasonal_order=(1, 0, 1, config.DAILY_SEASON),
                p_range=config.P_RANGE, d_range=config.D_RANGE,
                q_range=config.Q_RANGE, maxiter=200, n_jobs=1,
                progress=True) -> pd.DataFrame:
    """Fit every (p,d,q); one row per attempt with full diagnostics."""
    combos = [(p, d, q) for p in p_range for d in d_range for q in q_range]
    jobs = [(y, exog, c, seasonal_order, maxiter) for c in combos]
    rows = []
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for i, info in enumerate(ex.map(_grid_worker, jobs), 1):
                rows.append(info)
                if progress and i % 10 == 0:
                    print(f"[sarimax-select] {i}/{len(jobs)} fits", flush=True)
    else:
        for i, job in enumerate(jobs, 1):
            rows.append(_grid_worker(job))
            if progress and i % 10 == 0:
                print(f"[sarimax-select] {i}/{len(jobs)} fits", flush=True)
    t = pd.DataFrame(rows)
    t[["p", "d", "q"]] = pd.DataFrame(t["order"].tolist(), index=t.index)
    t["seasonal_order"] = t["seasonal_order"].astype(str)
    return t.drop(columns="order")[["p", "d", "q", "seasonal_order", "status",
                                    "converged", "aic", "n_warnings",
                                    "warnings", "error", "seconds", "maxiter"]]


def stage1_candidates(grid: pd.DataFrame, top_k: int = 2) -> pd.DataFrame:
    """Per d, the top_k converged fits by AIC. AIC never compared across d."""
    ok = grid[grid["status"] == "converged"].copy()
    ok["aic_rank_within_d"] = ok.groupby("d")["aic"].rank(method="first")
    cand = ok[ok["aic_rank_within_d"] <= top_k].sort_values(["d", "aic"])
    return cand.reset_index(drop=True)


def stage2_validation(df, candidates: pd.DataFrame, parts, scenario="operational",
                      seasonal_order=(1, 0, 1, config.DAILY_SEASON),
                      maxiter=200) -> pd.DataFrame:
    """Score each candidate on the VALIDATION origins with the shared protocol."""
    from . import backtest, metrics
    from .forecasters import SARIMAXForecaster
    rows = []
    for _, c in candidates.iterrows():
        order = (int(c.p), int(c.d), int(c.q))
        model = SARIMAXForecaster(order, seasonal_order, scenario=scenario,
                                  maxiter=maxiter,
                                  order_source="stage-1 candidate")
        row = {"p": order[0], "d": order[1], "q": order[2],
               "stage1_aic": float(c.aic)}
        try:
            res = backtest.run_backtest(df, {"sarimax": model}, parts.validation,
                                        scenario=scenario, verbose=False)
            f = res["forecasts"]
            s = metrics.score(f["y_true"], f["yhat"])
            fl = res["fit_log"].iloc[0]
            row.update({"val_RMSE": s["RMSE"], "val_MAE": s["MAE"],
                        "val_sMAPE": s["sMAPE"],
                        "fit_converged": bool(fl["converged"]),
                        "fit_warnings": int(fl.get("n_warnings", 0) or 0),
                        "fit_seconds": float(fl["seconds"]), "error": ""})
        except Exception as e:
            row.update({"val_RMSE": np.nan, "fit_converged": False,
                        "error": f"{type(e).__name__}: {e}"[:300]})
        rows.append(row)
    return pd.DataFrame(rows)


def choose(stage2: pd.DataFrame) -> pd.Series:
    ok = stage2[np.isfinite(stage2["val_RMSE"])]
    if ok.empty:
        raise RuntimeError("no SARIMAX candidate produced validation forecasts")
    conv = ok[ok["fit_converged"]]
    pool = conv if not conv.empty else ok
    return pool.sort_values(["val_RMSE", "stage1_aic"]).iloc[0]


def select_order(df, parts, scenario="operational", fast=False, top_k=2,
                 maxiter=200, n_jobs=1, progress=True) -> dict:
    """Full two-stage selection. Never touches test origins."""
    from . import sarimax_model
    from .protocol import SCENARIOS
    allowed = SCENARIOS[scenario]["future_columns"]
    exog_cols = [c for c in sarimax_model.EXOG_COLS if c in allowed] or None
    if fast:   # smoke-test grid only
        pr, dr, qr = range(0, 2), range(0, 2), range(0, 2)
        seasonal = (1, 0, 0, config.DAILY_SEASON)
    else:      # the original grid and seasonal order
        pr, dr, qr = config.P_RANGE, config.D_RANGE, config.Q_RANGE
        seasonal = (1, 0, 1, config.DAILY_SEASON)
    y, x = training_window(df, parts.train_end, exog_cols)
    assert y.index[-1] <= parts.train_end < parts.test[0]
    grid = stage1_grid(y, x, seasonal, pr, dr, qr, maxiter, n_jobs, progress)
    cand = stage1_candidates(grid, top_k)
    if cand.empty:
        raise RuntimeError("no SARIMAX fit converged in stage 1; raise maxiter")
    s2 = stage2_validation(df, cand, parts, scenario, seasonal, maxiter)
    best = choose(s2)
    order = [int(best.p), int(best.d), int(best.q)]
    boundary = (order[0] == max(pr) or order[2] == max(qr))
    return {
        "order": order, "seasonal_order": list(seasonal),
        "scenario": scenario, "exog_cols": exog_cols or [],
        "selected_on": "stage 1: within-d AIC on training window; "
                       "stage 2: pooled RMSE on validation origins",
        "selection_val_RMSE": float(best.val_RMSE),
        "selection_fit_converged": bool(best.fit_converged),
        "on_grid_boundary": bool(boundary),
        "stage1_window": [str(y.index[0]), str(y.index[-1])],
        "validation_origins": [str(o) for o in parts.validation],
        "test_origins_used": False,
        "grid_size": int(len(grid)),
        "stage1_status_counts": grid["status"].value_counts().to_dict(),
        "top_k_per_d": top_k, "maxiter": maxiter, "fast_grid": fast,
        "tables": {"grid": grid, "candidates": cand, "validation": s2},
    }
