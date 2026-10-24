"""Shared helpers: error metrics, train/test splitting and plotting utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: write figures to disk without a display
import matplotlib.pyplot as plt

from . import config


# --------------------------------------------------------------------------
# Error metrics
# --------------------------------------------------------------------------
def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error, ignoring zero-valued actuals."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE - more stable when actuals get close to zero."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom != 0
    return float(np.mean(2 * np.abs(y_pred - y_true)[mask] / denom[mask]) * 100)


def score(y_true, y_pred) -> dict:
    """Bundle the four metrics used throughout the study."""
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def train_test_split(series: pd.Series, horizon: int = config.HORIZON):
    """Hold out the final `horizon` observations as the test set."""
    return series.iloc[:-horizon], series.iloc[-horizon:]


def split_days(df: pd.DataFrame, test_days: int = config.ML_TEST_DAYS):
    """Hold out the final `test_days` days (used by the legacy XGBoost run)."""
    cutoff = df.index.max() - pd.Timedelta(days=test_days)
    return df[df.index <= cutoff], df[df.index > cutoff]


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def save_fig(fig, name: str):
    path = config.FIG_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_forecast(train_tail, y_true, y_pred, title, name,
                  lower=None, upper=None, tail=72):
    """Overlay a forecast on recent history, optionally with a CI band."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(train_tail.index[-tail:], train_tail.values[-tail:],
            color="0.4", label="history")
    ax.plot(y_true.index, y_true.values, color="black", lw=2, label="actual")
    ax.plot(y_true.index, np.asarray(y_pred), color="tab:red", lw=2,
            label="forecast")
    if lower is not None and upper is not None:
        ax.fill_between(y_true.index, lower, upper, color="tab:red",
                        alpha=0.18, label="95% CI")
    ax.set_title(title)
    ax.set_ylabel("Appliances (Wh)")
    ax.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    return save_fig(fig, name)
