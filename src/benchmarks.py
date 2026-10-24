"""Benchmark forecasts: mean, naive, seasonal naive (24 h / 168 h) and drift.

Every model forecasts the next `horizon` hours from the training history.
These simple methods set the bar every more complex model must beat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .utils import score


def _future_index(train: pd.Series, horizon: int) -> pd.DatetimeIndex:
    step = train.index[-1] - train.index[-2]
    return pd.date_range(train.index[-1] + step, periods=horizon, freq=step)


def mean_forecast(train, horizon=config.HORIZON):
    idx = _future_index(train, horizon)
    return pd.Series(np.repeat(train.mean(), horizon), index=idx)


def naive_forecast(train, horizon=config.HORIZON):
    """Carry the last observed value forward."""
    idx = _future_index(train, horizon)
    return pd.Series(np.repeat(train.iloc[-1], horizon), index=idx)


def seasonal_naive_forecast(train, horizon=config.HORIZON, season=config.DAILY_SEASON):
    """Repeat the value from one season ago (24h = daily, 168h = weekly)."""
    idx = _future_index(train, horizon)
    last_season = train.iloc[-season:].values
    reps = int(np.ceil(horizon / season))
    vals = np.tile(last_season, reps)[:horizon]
    return pd.Series(vals, index=idx)


def drift_forecast(train, horizon=config.HORIZON):
    """Extrapolate the average slope between the first and last point."""
    idx = _future_index(train, horizon)
    slope = (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)
    steps = np.arange(1, horizon + 1)
    return pd.Series(train.iloc[-1] + slope * steps, index=idx)


def run_all(train, test, horizon=config.HORIZON) -> dict:
    """Fit every benchmark and return {name: (forecast, metrics)}."""
    forecasts = {
        "Mean": mean_forecast(train, horizon),
        "Naive": naive_forecast(train, horizon),
        "SeasonalNaive-daily": seasonal_naive_forecast(
            train, horizon, config.DAILY_SEASON),
        "SeasonalNaive-weekly": seasonal_naive_forecast(
            train, horizon, config.WEEKLY_SEASON),
        "Drift": drift_forecast(train, horizon),
    }
    return {name: (fc, score(test.values, fc.values))
            for name, fc in forecasts.items()}
