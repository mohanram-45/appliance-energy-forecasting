"""Covariate / feature engineering.

Turns the hourly dataframe into a supervised-learning matrix. Feature groups:

  * sensor    - indoor temperatures (T1..T9) and humidities (RH_1..RH_9)
  * weather   - outdoor temperature, humidity, pressure, wind, dewpoint...
  * time      - hour-of-day and day-of-week, encoded cyclically, weekend flag
  * lag       - past values of the target (t-1, t-2, t-3, t-24, t-48, t-168)
  * rolling   - rolling mean / std of the target over 3, 6 and 24 hours

All lag and rolling features are shifted so a row only ever sees the past;
this prevents target leakage during model training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

TARGET = config.TARGET
LAGS = [1, 2, 3, 24, 48, 168]
ROLL_WINDOWS = [3, 6, 24]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = df.index
    df["hour"] = idx.hour
    df["dow"] = idx.dayofweek
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    # cyclical encodings so 23:00 and 00:00 sit next to each other
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    return df


def add_lag_features(df: pd.DataFrame, target=TARGET, lags=LAGS) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"{target}_lag{lag}"] = df[target].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target=TARGET,
                         windows=ROLL_WINDOWS) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        # shift(1) => window ends at t-1, never uses the current value
        roll = df[target].shift(1).rolling(w)
        df[f"{target}_rollmean{w}"] = roll.mean()
        df[f"{target}_rollstd{w}"] = roll.std()
    return df


def build_feature_frame(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """Full featured frame; keeps the target column named as config.TARGET."""
    out = add_time_features(df)
    out = add_lag_features(out)
    out = add_rolling_features(out)
    if dropna:
        out = out.dropna()
    return out


def target_history_features(y_past: pd.Series, ts, target=TARGET,
                            lags=LAGS, windows=ROLL_WINDOWS) -> dict:
    """Lag / rolling features for ONE timestamp from strictly earlier values.

    Produces exactly the values ``add_lag_features`` and
    ``add_rolling_features`` give for row ``ts`` of a regular hourly frame,
    but refuses to run if ``y_past`` contains ``ts`` or anything later. The
    recursive forecaster calls this with history + its own earlier
    predictions, so ground truth after the origin cannot reach a feature.
    """
    ts = pd.Timestamp(ts)
    if len(y_past) and y_past.index.max() >= ts:
        raise AssertionError(
            f"leakage guard: series passed for {ts} contains values at/after it")
    hour = pd.Timedelta(hours=1)
    out = {}
    for lag in lags:
        out[f"{target}_lag{lag}"] = y_past.get(ts - lag * hour, np.nan)
    for w in windows:
        vals = y_past.reindex(pd.date_range(ts - w * hour, periods=w,
                                            freq="h")).to_numpy(dtype=float)
        if np.isnan(vals).any():
            out[f"{target}_rollmean{w}"] = np.nan
            out[f"{target}_rollstd{w}"] = np.nan
        else:
            out[f"{target}_rollmean{w}"] = float(vals.mean())
            out[f"{target}_rollstd{w}"] = float(vals.std(ddof=1))
    return out


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Everything except the target and the lights energy channel."""
    drop = {TARGET, "lights"}
    return [c for c in frame.columns if c not in drop]


def feature_groups(cols: list[str]) -> dict[str, list[str]]:
    """Label each feature by group - used to discuss which groups help most."""
    groups = {"sensor": [], "weather": [], "time": [], "lag": [], "rolling": []}
    weather = {"T_out", "RH_out", "Press_mm_hg", "Windspeed", "Visibility",
               "Tdewpoint"}
    for c in cols:
        if "lag" in c:
            groups["lag"].append(c)
        elif "rollmean" in c or "rollstd" in c:
            groups["rolling"].append(c)
        elif c in {"hour", "dow", "is_weekend", "hour_sin", "hour_cos",
                   "dow_sin", "dow_cos"}:
            groups["time"].append(c)
        elif c in weather:
            groups["weather"].append(c)
        else:
            groups["sensor"].append(c)
    return groups
