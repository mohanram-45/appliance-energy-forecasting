"""Exploratory analysis and stationarity testing.

Produces the initial time-series plots, a seasonal decomposition, ACF/PACF
plots and the ADF + KPSS stationarity tests. All figures are written to
outputs/figures and the numeric test results are returned for the report.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

from . import config
from .utils import save_fig


def overview_plots(df: pd.DataFrame):
    """Full series, a representative week and the hourly/daily profiles."""
    y = df[config.TARGET]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y.index, y.values, lw=0.6)
    ax.set_title("Hourly appliance energy use - full period")
    ax.set_ylabel("Appliances (Wh)")
    save_fig(fig, "01_full_series.png")

    week = y.iloc[: 24 * 7]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(week.index, week.values, marker=".", lw=1)
    ax.set_title("First week - daily rhythm visible")
    ax.set_ylabel("Appliances (Wh)")
    save_fig(fig, "02_first_week.png")

    # average profile by hour-of-day and day-of-week
    prof = df.copy()
    prof["hour"] = prof.index.hour
    prof["dow"] = prof.index.dayofweek
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    prof.groupby("hour")[config.TARGET].mean().plot(ax=axes[0], marker="o")
    axes[0].set_title("Mean use by hour of day"); axes[0].set_xlabel("hour")
    prof.groupby("dow")[config.TARGET].mean().plot(ax=axes[1], marker="o")
    axes[1].set_title("Mean use by day of week"); axes[1].set_xlabel("0=Mon")
    save_fig(fig, "03_seasonal_profiles.png")


def decomposition(df: pd.DataFrame, period: int = config.DAILY_SEASON):
    y = df[config.TARGET]
    result = seasonal_decompose(y, model="additive", period=period)
    fig = result.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle("Additive decomposition (daily period)")
    save_fig(fig, "04_decomposition.png")
    return result


def correlograms(df: pd.DataFrame, lags: int = 72):
    y = df[config.TARGET]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    plot_acf(y, lags=lags, ax=axes[0])
    axes[0].set_title("ACF - raw series")
    plot_pacf(y, lags=lags, ax=axes[1], method="ywm")
    axes[1].set_title("PACF - raw series")
    save_fig(fig, "05_acf_pacf.png")


def differencing_plots(df: pd.DataFrame, lags: int = 72):
    """Make differencing explicit: plot the level, the first difference and the
    seasonal (24h) difference, then show ACF/PACF of the differenced series.

    Differencing is how we remove the trend/seasonality that the ADF/KPSS tests
    flag; the ACF/PACF of the differenced series is what justifies the SARIMA
    (p, q) and seasonal orders.
    """
    y = df[config.TARGET]
    d1 = y.diff().dropna()
    ds = y.diff(config.DAILY_SEASON).dropna()

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(y.index, y.values, lw=0.6)
    axes[0].set_title(f"Level (ADF p={adfuller(y.dropna())[1]:.3f})")
    axes[1].plot(d1.index, d1.values, lw=0.6, color="tab:orange")
    axes[1].set_title(f"1st difference (ADF p={adfuller(d1)[1]:.3f})")
    axes[2].plot(ds.index, ds.values, lw=0.6, color="tab:green")
    axes[2].set_title(f"Seasonal (24h) difference (ADF p={adfuller(ds)[1]:.3f})")
    fig.suptitle("Effect of differencing on stationarity")
    save_fig(fig, "05b_differencing.png")

    # ACF/PACF after first differencing -> reads off candidate SARIMA orders
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    plot_acf(d1, lags=lags, ax=axes[0])
    axes[0].set_title("ACF - 1st-differenced series")
    plot_pacf(d1, lags=lags, ax=axes[1], method="ywm")
    axes[1].set_title("PACF - 1st-differenced series")
    save_fig(fig, "05c_acf_pacf_differenced.png")


def stationarity_tests(series: pd.Series, label: str = "series") -> dict:
    """ADF (null = unit root) and KPSS (null = stationary)."""
    adf = adfuller(series.dropna(), autolag="AIC")
    try:
        kp = kpss(series.dropna(), regression="c", nlags="auto")
    except Exception:
        kp = (float("nan"), float("nan"), None, {})
    out = {
        "label": label,
        "adf_stat": adf[0], "adf_pvalue": adf[1],
        "adf_stationary": adf[1] < 0.05,
        "kpss_stat": kp[0], "kpss_pvalue": kp[1],
        "kpss_stationary": (kp[1] > 0.05) if kp[1] == kp[1] else None,
    }
    return out


def run(df: pd.DataFrame) -> pd.DataFrame:
    """Run every EDA step and return a tidy stationarity results table."""
    config.assert_legacy_write_allowed("eda.run")
    overview_plots(df)
    decomposition(df)
    correlograms(df)
    differencing_plots(df)

    y = df[config.TARGET]
    rows = [
        stationarity_tests(y, "level"),
        stationarity_tests(y.diff().dropna(), "1st difference"),
        stationarity_tests(y.diff(config.DAILY_SEASON).dropna(),
                            "seasonal (24h) difference"),
    ]
    table = pd.DataFrame(rows)
    table.to_csv(config.METRIC_DIR / "stationarity_tests.csv", index=False)
    return table
