"""Point-forecast error metrics used by the rolling-origin backtest.

Conventions (documented because they change results when targets hit zero):

* RMSE, MAE: ordinary definitions over all evaluated points.
* MAPE: undefined when the actual value is 0, so those points are EXCLUDED
  and the number excluded is reported. If every actual is 0, MAPE is NaN.
* sMAPE: 200 * |y - yhat| / (|y| + |yhat|). When both y and yhat are 0 the
  forecast is perfect, so that point contributes 0 (not dropped, not NaN).
  Range: 0-200.
* MASE (extra, scale-free): MAE divided by the in-sample MAE of the
  seasonal-naive forecast on the training history available at the origin.

Hourly appliance energy in the UCI data never reaches 0 (minimum 170 Wh),
but the functions are written so that synthetic or other data cannot
silently produce inf/NaN.
"""
from __future__ import annotations

import numpy as np

METRIC_NAMES = ("RMSE", "MAE", "MAPE", "sMAPE")


def _arrays(y_true, y_pred):
    y = np.asarray(y_true, dtype=float).ravel()
    f = np.asarray(y_pred, dtype=float).ravel()
    if y.shape != f.shape:
        raise ValueError(f"shape mismatch: y_true {y.shape} vs y_pred {f.shape}")
    if np.isnan(y).any() or np.isnan(f).any():
        raise ValueError("NaN in y_true or y_pred; evaluation windows must "
                         "be complete and forecasts finite.")
    return y, f


def rmse(y_true, y_pred) -> float:
    y, f = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y - f) ** 2)))


def mae(y_true, y_pred) -> float:
    y, f = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(y - f)))


def mape(y_true, y_pred) -> tuple[float, int]:
    """Return (MAPE %, number of zero-actual points excluded)."""
    y, f = _arrays(y_true, y_pred)
    nonzero = y != 0
    n_excluded = int((~nonzero).sum())
    if not nonzero.any():
        return float("nan"), n_excluded
    return float(np.mean(np.abs((y[nonzero] - f[nonzero]) / y[nonzero])) * 100), n_excluded


def smape(y_true, y_pred) -> float:
    y, f = _arrays(y_true, y_pred)
    denom = np.abs(y) + np.abs(f)
    terms = np.zeros_like(y)
    ok = denom != 0
    terms[ok] = 2 * np.abs(f[ok] - y[ok]) / denom[ok]
    return float(np.mean(terms) * 100)


def mase(y_true, y_pred, y_history, season: int = 24) -> float:
    """MAE scaled by the in-sample seasonal-naive MAE of ``y_history``."""
    y, f = _arrays(y_true, y_pred)
    h = np.asarray(y_history, dtype=float)
    if len(h) <= season:
        return float("nan")
    d = np.abs(h[season:] - h[:-season])  # NaN where either hour is masked
    if np.isnan(d).all():
        return float("nan")
    scale = np.nanmean(d)
    if scale == 0:
        return float("nan")
    return float(np.mean(np.abs(y - f)) / scale)


def score(y_true, y_pred, y_history=None, season: int = 24) -> dict:
    """All metrics for one forecast window (or one pooled set of points)."""
    m, n_zero = mape(y_true, y_pred)
    out = {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": m,
        "sMAPE": smape(y_true, y_pred),
        "n_points": int(np.asarray(y_true).size),
        "n_zero_actuals_excluded_from_MAPE": n_zero,
    }
    if y_history is not None:
        out["MASE"] = mase(y_true, y_pred, y_history, season)
    return out
