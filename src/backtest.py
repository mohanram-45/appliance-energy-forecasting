"""Rolling-origin backtest: run every model on identical origins and targets.

Retraining policy (``refit_every``)
-----------------------------------
* ``0`` (default, "fixed"): each trainable model is fitted ONCE on all data up
  to the first origin of the evaluated partition, then kept fixed. At later
  origins it still conditions on the newly observed history (lags for
  XGBoost, Kalman state for SARIMAX) but its parameters do not change.
* ``k >= 1``: refit at origin 0, k, 2k, ... on all data up to that origin
  (expanding window). ``1`` = refit at every origin.

Untrained models (benchmarks, zero-shot Chronos) have nothing to fit; they
are recomputed from the history at every origin regardless of the policy.

For the TEST partition, "data up to the first origin" includes the
validation period (standard: select on validation, refit on train+val,
evaluate on test). For the VALIDATION partition it is the training period
only, so validation scores never see validation or test targets.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, metrics
from .forecasters import Forecaster
from .protocol import (future_view, history_view, target_index,
                       assert_no_forbidden_future_columns)

TARGET = config.TARGET


def _should_fit(i: int, refit_every: int) -> bool:
    return i == 0 or (refit_every > 0 and i % refit_every == 0)


def run_backtest(df: pd.DataFrame, models: dict[str, Forecaster],
                 origins, horizon: int = config.HORIZON,
                 scenario: str = "operational", refit_every: int = 0,
                 verbose: bool = True) -> dict:
    """Return {'forecasts': long DataFrame, 'fit_log': DataFrame}.

    The true target values are looked up only AFTER a model has returned its
    forecast and are never passed to the model.
    """
    origins = pd.DatetimeIndex(origins)
    rows, fit_log = [], []
    for name, model in models.items():
        t_model = time.perf_counter()
        for i, origin in enumerate(origins):
            hist = history_view(df, origin)
            fut = future_view(df, origin, horizon, scenario)
            if model.trainable and _should_fit(i, refit_every):
                t0 = time.perf_counter()
                model.fit(hist)
                fit_log.append({"model": name, "origin": origin,
                                "fit_data_end": hist.index[-1],
                                "history_rows": len(hist),
                                "train_rows": getattr(model, "n_train_rows_", None),
                                "seconds": round(time.perf_counter() - t0, 3),
                                "converged": getattr(model, "converged_", None),
                                "n_warnings": getattr(model, "n_warnings_", None),
                                "warnings": getattr(model, "warnings_", None)})
            pred = model.predict(hist, fut)
            _validate_prediction(name, origin, pred, horizon)

            y_true = df.loc[pred.index, TARGET].to_numpy(dtype=float)
            block = pd.DataFrame({
                "model": name, "origin": origin,
                "target_time": pred.index,
                "step": np.arange(1, horizon + 1),
                "y_true": y_true, "yhat": pred["yhat"].to_numpy(),
            })
            for c in ("lower", "upper", "interval_level"):
                block[c] = pred[c].to_numpy() if c in pred else np.nan
            rows.append(block)
        if verbose:
            print(f"[backtest] {name:20s} {len(origins)} origins "
                  f"in {time.perf_counter() - t_model:.1f}s", file=sys.stderr)
    forecasts = pd.concat(rows, ignore_index=True)
    check_identical_targets(forecasts)
    return {"forecasts": forecasts, "fit_log": pd.DataFrame(fit_log)}


def _validate_prediction(name, origin, pred: pd.DataFrame, horizon: int):
    expected = target_index(origin, horizon)
    if not isinstance(pred, pd.DataFrame) or "yhat" not in pred:
        raise TypeError(f"{name}: predict() must return a DataFrame with 'yhat'")
    if not pred.index.equals(expected):
        raise AssertionError(f"{name} @ {origin}: forecast index != "
                             f"{expected[0]}..{expected[-1]}")
    if not np.isfinite(pred["yhat"].to_numpy(dtype=float)).all():
        raise ValueError(f"{name} @ {origin}: non-finite forecast")


def check_identical_targets(forecasts: pd.DataFrame):
    """Every model must be scored on exactly the same (origin, target) pairs."""
    keys = {m: set(zip(g["origin"], g["target_time"]))
            for m, g in forecasts.groupby("model")}
    ref = next(iter(keys.values()))
    for m, k in keys.items():
        if k != ref:
            raise AssertionError(f"model {m} evaluated on different targets")


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def window_metrics(forecasts: pd.DataFrame, df: pd.DataFrame,
                   season: int = config.DAILY_SEASON) -> pd.DataFrame:
    """One row per (model, origin): metrics over that 24-hour window."""
    out = []
    for (m, origin), g in forecasts.groupby(["model", "origin"], sort=False):
        hist = df.loc[:origin, TARGET].to_numpy(dtype=float)
        s = metrics.score(g["y_true"], g["yhat"], y_history=hist, season=season)
        out.append({"model": m, "origin": origin, **s})
    return pd.DataFrame(out)


def aggregate_metrics(forecasts: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Per model: distribution of window metrics + metrics pooled over all points."""
    rows = []
    for m, w in windows.groupby("model", sort=False):
        f = forecasts[forecasts["model"] == m]
        pooled = metrics.score(f["y_true"], f["yhat"])
        row = {"model": m, "n_windows": int(len(w)),
               "n_points": int(pooled["n_points"])}
        for k in ("RMSE", "MAE", "MAPE", "sMAPE"):
            row[f"pooled_{k}"] = pooled[k]
        for k in ("RMSE", "MAE", "MAPE", "sMAPE", "MASE"):
            if k not in w:
                continue
            v = w[k].astype(float)
            row.update({f"window_{k}_mean": v.mean(),
                        f"window_{k}_median": v.median(),
                        f"window_{k}_std": v.std(ddof=1),
                        f"window_{k}_q25": v.quantile(0.25),
                        f"window_{k}_q75": v.quantile(0.75),
                        f"window_{k}_min": v.min(),
                        f"window_{k}_max": v.max()})
        row["zero_actuals_excluded_from_MAPE"] = pooled["n_zero_actuals_excluded_from_MAPE"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pooled_RMSE").reset_index(drop=True)


def horizon_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    """RMSE / MAE by forecast step (1..H) across all origins."""
    g = forecasts.assign(err=forecasts["yhat"] - forecasts["y_true"])
    return (g.groupby(["model", "step"])["err"]
            .agg(RMSE=lambda e: float(np.sqrt(np.mean(e ** 2))),
                 MAE=lambda e: float(np.mean(np.abs(e))),
                 n="size")
            .reset_index())


# --------------------------------------------------------------------------
# Provenance and output
# --------------------------------------------------------------------------
def run_metadata() -> dict:
    meta = {"python": sys.version.split()[0], "platform": platform.platform(),
            "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    for pkg in ("pandas", "numpy", "statsmodels", "xgboost", "torch", "chronos"):
        try:
            mod = __import__(pkg)
            meta[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except Exception:
            meta[f"{pkg}_version"] = None
    try:
        meta["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=config.ROOT, capture_output=True,
            text=True, timeout=10).stdout.strip() or None
        meta["git_dirty"] = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=config.ROOT,
            capture_output=True, text=True, timeout=10).stdout.strip())
    except Exception:
        meta["git_commit"], meta["git_dirty"] = None, None
    return meta


def unique_run_name(prefix: str) -> str:
    """``<prefix>_<YYYYmmdd-HHMMSS>`` (UTC) - unique per second."""
    return f"{prefix}_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d-%H%M%S')}"


def prepare_run_dir(run_name: str | None, scenario: str,
                    base: Path | None = None, prefix: str = "run") -> Path:
    """Create ``<base>/<scenario>/<run_name>/``; never reuse an existing one.

    Scenarios live in separate sub-folders so operational and weather_oracle
    results cannot be mixed up. Legacy folders are refused outright.
    """
    base = Path(base) if base else config.BACKTEST_DIR
    protected = {config.LEGACY_METRIC_DIR.resolve(), config.LEGACY_FIG_DIR.resolve(),
                 config.METRIC_DIR.resolve(), config.FIG_DIR.resolve()}
    name = run_name or unique_run_name(prefix)
    if any(ch in name for ch in "/\\") or name in {"", ".", ".."}:
        raise ValueError(f"invalid run name {name!r}")
    out = (base / scenario / name).resolve()
    if out in protected or any(p in out.parents for p in protected):
        raise ValueError("refusing to write into the legacy outputs folders")
    if out.exists():
        raise FileExistsError(f"{out} already exists; results are never "
                              "overwritten - choose another --run-name")
    out.mkdir(parents=True)
    return out


ORACLE_NOTICE = (
    "WEATHER-ORACLE SCENARIO - NOT A REALISTIC FORECAST RESULT\n\n"
    "These forecasts used OBSERVED outdoor weather for the target hours as a\n"
    "stand-in for a perfect weather forecast. That information is not\n"
    "available at the forecast origin. Treat every number in this folder as\n"
    "an upper bound on what weather information could add, never as\n"
    "operational accuracy, and never mix it with operational results.\n")


def label_scenario(frame: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Add scenario columns so a table cannot lose its provenance."""
    out = frame.copy()
    out.insert(0, "scenario", scenario)
    out.insert(1, "operationally_valid", scenario == "operational")
    return out


def save_results(out_dir: Path, result: dict, windows, aggregate, horizon,
                 run_config: dict):
    sc = run_config["scenario"]
    label_scenario(result["forecasts"], sc).to_csv(out_dir / "forecasts.csv", index=False)
    result["fit_log"].to_csv(out_dir / "fit_log.csv", index=False)
    label_scenario(windows, sc).to_csv(out_dir / "window_metrics.csv", index=False)
    label_scenario(aggregate, sc).to_csv(out_dir / "aggregate_metrics.csv", index=False)
    label_scenario(horizon, sc).to_csv(out_dir / "horizon_metrics.csv", index=False)
    (out_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, default=str))
    if sc != "operational":
        (out_dir / "SCENARIO_NOTICE.txt").write_text(ORACLE_NOTICE)
