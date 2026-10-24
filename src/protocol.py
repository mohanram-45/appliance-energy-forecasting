"""Leakage-safe forecasting protocol: origins, partitions and feature availability.

Definitions used everywhere in the backtest
-------------------------------------------
* **Forecast origin** ``t0``: the timestamp of the LAST hour whose value is
  observed. With ``ORIGIN_HOUR = 23`` the origin is 23:00 and the forecast
  covers the next calendar day.
* **Target window**: ``t0 + 1h, ..., t0 + H h`` (H = 24). Step ``h`` is the
  forecast for ``t0 + h``.
* **History view**: every column of the hourly frame up to and including
  ``t0``. This is the only data a model may see from the past.
* **Future view**: what is allowed to be known about the target hours *at the
  origin*. It always contains calendar features (derived from timestamps
  alone). Other columns are added only if the scenario explicitly allows them.

Feature availability classes
----------------------------
=====================  ==========================================  ==============
Class                  Columns                                     Future use?
=====================  ==========================================  ==============
known in advance       calendar: hour, day of week, weekend flag   always
historical observed    target (Appliances), lights, indoor T1..T9  never
                       / RH_1..RH_9, outdoor weather               (history only)
oracle (labelled)      outdoor weather at the target hour          only in the
                                                                   ``weather_oracle``
                                                                   scenario
=====================  ==========================================  ==============

``operational`` is the default and the only scenario that matches what a real
system would know. ``weather_oracle`` substitutes OBSERVED outdoor weather for
a weather forecast; it is an upper bound on what perfect weather forecasts
could add and must always be reported as such. Indoor sensors and the target
are never available for future hours in any scenario.

Partitions
----------
Eligible origins (complete look-back and complete target window) are split
chronologically: the last ``n_test`` form the TEST partition, the
``n_val`` before them the VALIDATION partition, and all data before the first
validation origin is the TRAINING partition. Because origins are 24 h apart
and windows do not overlap, the last validation target hour is the first test
origin: no hour is ever both a validation target and a test target.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

TARGET = config.TARGET
CALENDAR_COLS = ["hour", "dow", "is_weekend",
                 "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
OUTDOOR_WEATHER_COLS = ["T_out", "RH_out", "Press_mm_hg", "Windspeed",
                        "Visibility", "Tdewpoint"]
INDOOR_SENSOR_COLS = ([f"T{i}" for i in range(1, 10)]
                      + [f"RH_{i}" for i in range(1, 10)])
NEVER_FUTURE_COLS = [TARGET, "lights"] + INDOOR_SENSOR_COLS

SCENARIOS = {
    "operational": {
        "future_columns": [],
        "description": "Only calendar features are known for target hours. "
                       "All measured series (target, sensors, weather) are "
                       "used only up to the origin.",
    },
    "weather_oracle": {
        "future_columns": OUTDOOR_WEATHER_COLS,
        "description": "As operational, plus OBSERVED outdoor weather at the "
                       "target hours as a stand-in for a perfect weather "
                       "forecast. Upper bound - not an operational result.",
    },
}


# --------------------------------------------------------------------------
# Calendar features: functions of the timestamp only
# --------------------------------------------------------------------------
def calendar_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    idx = pd.DatetimeIndex(index)
    return pd.DataFrame({
        "hour": idx.hour,
        "dow": idx.dayofweek,
        "is_weekend": (idx.dayofweek >= 5).astype(int),
        "hour_sin": np.sin(2 * np.pi * idx.hour / 24),
        "hour_cos": np.cos(2 * np.pi * idx.hour / 24),
        "dow_sin": np.sin(2 * np.pi * idx.dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * idx.dayofweek / 7),
    }, index=idx)


# --------------------------------------------------------------------------
# Windows and views
# --------------------------------------------------------------------------
def target_index(origin, horizon: int = config.HORIZON) -> pd.DatetimeIndex:
    origin = pd.Timestamp(origin)
    return pd.date_range(origin + pd.Timedelta(hours=1), periods=horizon,
                         freq=config.FREQ)


def history_view(df: pd.DataFrame, origin) -> pd.DataFrame:
    """All columns up to and including the origin (a copy, so models cannot
    mutate the master frame)."""
    hist = df.loc[:pd.Timestamp(origin)].copy()
    assert hist.index.max() == pd.Timestamp(origin), "origin not in data"
    return hist


def future_view(df: pd.DataFrame, origin, horizon: int = config.HORIZON,
                scenario: str = "operational") -> pd.DataFrame:
    """What may be known at the origin about the target hours."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; "
                         f"choose from {sorted(SCENARIOS)}")
    idx = target_index(origin, horizon)
    fut = calendar_frame(idx)
    allowed = SCENARIOS[scenario]["future_columns"]
    if allowed:
        missing = [c for c in allowed if c not in df.columns]
        if missing:
            raise KeyError(f"scenario {scenario} needs columns {missing}")
        fut = fut.join(df.loc[idx, allowed])
    assert_no_forbidden_future_columns(fut)
    return fut


def assert_no_forbidden_future_columns(future: pd.DataFrame):
    bad = [c for c in future.columns if c in NEVER_FUTURE_COLS]
    if bad:
        raise AssertionError(f"leakage: future view contains {bad}")


# --------------------------------------------------------------------------
# Origins and partitions
# --------------------------------------------------------------------------
def eligible_origins(df: pd.DataFrame, horizon: int = config.HORIZON,
                     origin_hour: int = config.ORIGIN_HOUR,
                     lookback: int = config.MAX_LOOKBACK_HOURS,
                     step_hours: int = config.ORIGIN_STEP_HOURS
                     ) -> pd.DatetimeIndex:
    """Origins whose look-back and target window are fully observed.

    * The hourly index must be regular (no missing timestamps).
    * The target must be non-missing for the ``lookback`` hours ending at the
      origin (lags up to 168 h) and for all ``horizon`` target hours.
    """
    idx = df.index
    if idx.freq is None:
        inferred = pd.infer_freq(idx)
        if inferred is None or pd.Timedelta(pd.tseries.frequencies.to_offset(
                inferred)) != pd.Timedelta(hours=1):
            raise ValueError("hourly index must be regular; use asfreq('h')")
    y = df[TARGET]
    ok = y.notna().astype(int)
    # complete look-back: count of observed values in the trailing window
    back_ok = ok.rolling(lookback, min_periods=lookback).sum() == lookback
    # complete target window: next `horizon` hours all observed
    fwd_ok = (ok[::-1].rolling(horizon, min_periods=horizon).sum()[::-1]
              .shift(-1) == horizon)
    if step_hours != 24:
        raise NotImplementedError("only daily origins (step 24 h) are "
                                  "supported; windows must not overlap")
    cand = (idx.hour == origin_hour) & back_ok.values & fwd_ok.fillna(False).values
    return idx[cand]


@dataclass
class Partitions:
    train_end: pd.Timestamp            # last hour of the training partition
    validation: pd.DatetimeIndex       # validation origins
    test: pd.DatetimeIndex             # test origins
    data_start: pd.Timestamp
    data_end: pd.Timestamp
    notes: list = field(default_factory=list)

    def origins(self, name: str) -> pd.DatetimeIndex:
        if name == "validation":
            return self.validation
        if name == "test":
            return self.test
        raise ValueError("partition must be 'validation' or 'test'")

    def summary(self) -> dict:
        def span(o):
            return {"n_origins": int(len(o)),
                    "first_origin": str(o[0]) if len(o) else None,
                    "last_origin": str(o[-1]) if len(o) else None,
                    "first_target": str(o[0] + pd.Timedelta(hours=1)) if len(o) else None,
                    "last_target": str(o[-1] + pd.Timedelta(hours=config.HORIZON)) if len(o) else None}
        return {"data_start": str(self.data_start),
                "data_end": str(self.data_end),
                "train_end": str(self.train_end),
                "validation": span(self.validation),
                "test": span(self.test),
                "notes": self.notes}


def make_partitions(df: pd.DataFrame, n_test: int = config.N_TEST_ORIGINS,
                    n_val: int = config.N_VAL_ORIGINS,
                    horizon: int = config.HORIZON,
                    origin_hour: int = config.ORIGIN_HOUR,
                    min_train_hours: int = config.MIN_TRAIN_HOURS
                    ) -> Partitions:
    """Chronological train / validation / test split of forecast origins."""
    if n_test < 1:
        raise ValueError("n_test must be >= 1")
    origins = eligible_origins(df, horizon, origin_hour)
    need = n_test + n_val
    if len(origins) < need:
        raise ValueError(f"only {len(origins)} eligible origins; "
                         f"{need} requested (n_test={n_test}, n_val={n_val})")
    test = origins[-n_test:]
    val = origins[-need:-n_test] if n_val else origins[:0]
    first_eval = val[0] if n_val else test[0]
    train_end = first_eval
    train_hours = int((train_end - df.index[0]) / pd.Timedelta(hours=1)) + 1
    if train_hours < min_train_hours:
        raise ValueError(f"training partition has {train_hours} h; at least "
                         f"{min_train_hours} h required")
    notes = []
    gaps = (np.diff(origins[-need:].asi8) // 3_600_000_000_000)
    if len(gaps) and (gaps != 24).any():
        notes.append("some consecutive origins are more than 24 h apart "
                     "because ineligible days were skipped")
    return Partitions(train_end=train_end, validation=val, test=test,
                      data_start=df.index[0], data_end=df.index[-1],
                      notes=notes)
