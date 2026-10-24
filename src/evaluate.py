"""Legacy single-window evaluation: comparison table and diagnostic plots."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .utils import save_fig


def comparison_table(results: dict) -> pd.DataFrame:
    """results: {name: metrics_dict}. Sorted by RMSE (lower is better)."""
    config.assert_legacy_write_allowed("evaluate.comparison_table")
    table = pd.DataFrame(results).T
    table = table.sort_values("RMSE")
    table.index.name = "model"
    table.to_csv(config.METRIC_DIR / "model_comparison.csv")
    return table


def all_forecasts_plot(actual: pd.Series, forecasts: dict,
                       train_tail: pd.Series, tail=72):
    """One figure overlaying every model's 24h forecast on the actuals."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(train_tail.index[-tail:], train_tail.values[-tail:],
            color="0.6", lw=1, label="history")
    ax.plot(actual.index, actual.values, color="black", lw=2.5, label="actual")
    for name, fc in forecasts.items():
        ax.plot(fc.index, fc.values, lw=1.4, alpha=0.9, label=name)
    ax.set_title("24-hour forecasts - all models")
    ax.set_ylabel("Appliances (Wh)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.autofmt_xdate()
    return save_fig(fig, "08_all_forecasts.png")


def error_diagnostics(actual: pd.Series, forecasts: dict,
                      benchmark_name: str):
    """Per-hour error curves and a bar chart of RMSE vs the best benchmark."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    for name, fc in forecasts.items():
        axes[0].plot(range(len(actual)), fc.values - actual.values,
                     marker=".", lw=1, label=name)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_title("Forecast error by hour ahead")
    axes[0].set_xlabel("hours into the forecast")
    axes[0].set_ylabel("error (Wh)")
    axes[0].legend(fontsize=7, ncol=2)

    rmses = {n: float(np.sqrt(np.mean((fc.values - actual.values) ** 2)))
             for n, fc in forecasts.items()}
    s = pd.Series(rmses).sort_values()
    colors = ["tab:green" if n != benchmark_name else "tab:orange"
              for n in s.index]
    s.plot.barh(ax=axes[1], color=colors)
    axes[1].set_title(f"RMSE (orange = best benchmark: {benchmark_name})")
    axes[1].set_xlabel("RMSE (Wh)")
    save_fig(fig, "09_error_diagnostics.png")
    return s


def skill_scores(comparison: pd.DataFrame, benchmark_name: str) -> pd.DataFrame:
    """Percentage RMSE improvement of each model over the best benchmark."""
    base = comparison.loc[benchmark_name, "RMSE"]
    out = comparison.copy()
    out["RMSE_skill_%_vs_benchmark"] = (1 - out["RMSE"] / base) * 100
    out.to_csv(config.METRIC_DIR / "model_comparison_with_skill.csv")
    return out
