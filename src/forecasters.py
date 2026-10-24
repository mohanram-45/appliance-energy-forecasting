"""Model adapters for the shared rolling-origin protocol.

Every forecaster exposes the same two calls:

    fit(history)                -> self          (no-op for untrained models)
    predict(history, future)    -> DataFrame indexed exactly like ``future``
                                   with column ``yhat`` and optional
                                   ``lower`` / ``upper`` / ``interval_level``

``history`` is ``protocol.history_view(df, origin)``: data up to and including
the origin. ``future`` is ``protocol.future_view(...)``: calendar features for
the target hours plus any scenario-approved columns. Neither ever contains the
target or indoor sensors for target hours, so an adapter cannot leak future
ground truth even by mistake.

The adapters wrap - they do not replace - the original implementations in
``benchmarks.py``, ``sarimax_model.py``, ``ml_model.py`` and
``foundation.py``. Original model configurations are kept; every deviation
needed for leakage safety is listed in ``deviations`` and written to the run
metadata.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import benchmarks, config
from . import features as F
from .protocol import SCENARIOS, CALENDAR_COLS

TARGET = config.TARGET


class Forecaster:
    name = "base"
    trainable = False          # does fit() estimate anything?
    deviations: list[str] = []  # differences from the original configuration

    def fit(self, history: pd.DataFrame) -> "Forecaster":
        return self

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name, "class": type(self).__name__,
                "trainable": self.trainable, "deviations": list(self.deviations)}

    @staticmethod
    def _frame(future, yhat, lower=None, upper=None, level=None):
        out = pd.DataFrame({"yhat": np.asarray(yhat, dtype=float)},
                           index=future.index)
        if lower is not None:
            out["lower"] = np.asarray(lower, dtype=float)
            out["upper"] = np.asarray(upper, dtype=float)
            out["interval_level"] = level
        return out


# --------------------------------------------------------------------------
# Benchmarks: stateless, recomputed from the history at every origin
# --------------------------------------------------------------------------
class _BenchmarkForecaster(Forecaster):
    def __init__(self, name, fn, **kw):
        self.name, self._fn, self._kw = name, fn, kw
        self.deviations = []

    def predict(self, history, future):
        # pandas .mean() skips NaN (masked incomplete hours); eligibility
        # guarantees the last 168 h - all the naive variants need - are complete
        fc = self._fn(history[TARGET], len(future), **self._kw)
        if not fc.index.equals(future.index):
            raise AssertionError(f"{self.name}: forecast index misaligned")
        return self._frame(future, fc.values)


def mean_model():
    return _BenchmarkForecaster("mean", benchmarks.mean_forecast)


def naive_model():
    return _BenchmarkForecaster("naive", benchmarks.naive_forecast)


def drift_model():
    return _BenchmarkForecaster("drift", benchmarks.drift_forecast)


def seasonal_naive_model(season: int):
    return _BenchmarkForecaster(f"seasonal_naive_{season}",
                                benchmarks.seasonal_naive_forecast,
                                season=season)


# --------------------------------------------------------------------------
# XGBoost - recursive multi-step, original hyper-parameters
# --------------------------------------------------------------------------
class XGBoostRecursive(Forecaster):
    """Recursive 24-step XGBoost using the original ``ml_model._model()``.

    Features per target hour: target lags (1,2,3,24,48,168), rolling mean/std
    (3,6,24) computed from history + earlier predictions, calendar features,
    and - only in the ``weather_oracle`` scenario - outdoor weather at that
    hour. Indoor sensors are not used because their values for target hours
    are unknowable at the origin.
    """
    name = "xgboost"
    trainable = True

    def __init__(self, scenario: str = "operational", params: dict | None = None):
        from . import ml_model  # lazy: xgboost only needed if used
        self.scenario = scenario
        self.model = ml_model._model()
        if params:
            self.model.set_params(**params)
        self.exog_cols = list(SCENARIOS[scenario]["future_columns"])
        self.feature_cols = None
        self.deviations = [
            "Indoor sensors (T1-T9, RH_1-RH_9) and lights removed: their "
            "target-hour values are unknown at the origin.",
            "Outdoor weather used only in the weather_oracle scenario."
            if scenario == "weather_oracle" else
            "Outdoor weather removed (operational scenario).",
            "Training data = all history up to the fit origin (original used "
            "data up to 14 days before the end).",
        ]
        if params:
            self.deviations.append(f"Hyper-parameters overridden: {params}")

    def _training_frame(self, history):
        cal = F.add_time_features(history[[TARGET]])
        cal = cal[[TARGET] + CALENDAR_COLS]
        frame = F.add_rolling_features(F.add_lag_features(cal))
        if self.exog_cols:
            frame = frame.join(history[self.exog_cols])
        return frame.dropna()

    def fit(self, history):
        frame = self._training_frame(history)
        self.feature_cols = [c for c in frame.columns if c != TARGET]
        self.model.fit(frame[self.feature_cols], frame[TARGET])
        self.n_train_rows_ = int(len(frame))
        return self

    def predict(self, history, future):
        if self.feature_cols is None:
            raise RuntimeError("fit() before predict()")
        y = history[TARGET].copy()      # observed up to the origin only
        preds = []
        for ts in future.index:
            row = F.target_history_features(y, ts)
            for c in CALENDAR_COLS + self.exog_cols:
                row[c] = future.at[ts, c]
            x = pd.DataFrame([row], index=[ts])[self.feature_cols]
            yhat = float(self.model.predict(x)[0])
            preds.append(yhat)
            y.loc[ts] = yhat            # feed back the PREDICTION, not truth
        # no clipping: the original XGBoost forecast was not clipped either
        return self._frame(future, preds)


# --------------------------------------------------------------------------
# XGBoost - direct multi-horizon (one model per forecast step)
# --------------------------------------------------------------------------
ORIGIN_LAGS = (0, 1, 2, 23, 167)       # y[t0-k]: last hours, same hour 1 d / 1 w ago
ORIGIN_ROLL = (3, 24, 168)             # rolling mean/std ending at the origin


def direct_origin_features(y: pd.Series) -> pd.DataFrame:
    """Features observable at origin t (row t uses y up to and including t)."""
    out = pd.DataFrame(index=y.index)
    for k in ORIGIN_LAGS:
        out[f"y_t-{k}"] = y.shift(k)
    for w in ORIGIN_ROLL:
        r = y.rolling(w, min_periods=w)
        out[f"roll_mean_{w}"] = r.mean()
        out[f"roll_std_{w}"] = r.std()
    return out


class XGBoostDirect(Forecaster):
    """Direct multi-horizon XGBoost: a separate model for each step h = 1..24.

    Model h maps information observable at the origin t0 to y[t0 + h]:

    * origin features: y[t0], y[t0-1], y[t0-2], y[t0-23], y[t0-167] and
      rolling mean/std over 3, 24 and 168 h ending at t0;
    * seasonal lags of the TARGET hour that are still in the past:
      y[t0+h-24] (same hour yesterday) and y[t0+h-168] (same hour last week)
      - both <= t0 for every h <= 24;
    * calendar features of the target hour (+ outdoor weather at the target
      hour only in the weather_oracle scenario).

    No prediction is ever fed back, so errors cannot accumulate across steps.
    Training rows use every hour t of the history as an origin, but only
    rows whose TARGET t + h is <= the fit cutoff (checked by a test).
    Hyper-parameters are the original ``ml_model._model()`` values.
    """
    name = "xgboost_direct"
    trainable = True

    def __init__(self, scenario: str = "operational", params: dict | None = None,
                 horizon: int = config.HORIZON):
        from . import ml_model
        self.scenario, self.horizon, self.params = scenario, horizon, params
        self._new_model = ml_model._model
        self.exog_cols = list(SCENARIOS[scenario]["future_columns"])
        self.models, self.feature_cols = {}, None
        self.deviations = [
            "Direct strategy: 24 horizon-specific models with the original "
            "XGBoost hyper-parameters; no recursion.",
            "Inputs: target history observable at the origin + calendar of the "
            "target hour" + (" + observed outdoor weather (weather_oracle)."
                             if self.exog_cols else "."),
        ]

    def _rows(self, y: pd.Series, h: int, exog: pd.DataFrame | None) -> pd.DataFrame:
        """One row per origin t: features at t, target y[t+h] (NaN if unknown)."""
        X = direct_origin_features(y)
        X["y_same_hour_1d"] = y.shift(24 - h)      # y[t+h-24]
        X["y_same_hour_1w"] = y.shift(168 - h)     # y[t+h-168]
        tgt_idx = y.index + pd.Timedelta(hours=h)
        cal = protocol_calendar(tgt_idx)
        cal.index = y.index
        X = X.join(cal)
        if self.exog_cols and exog is not None:
            w = exog.reindex(tgt_idx)
            w.index = y.index
            X = X.join(w)
        X["target"] = y.shift(-h)                  # NaN beyond the history
        return X

    def fit(self, history):
        y = history[TARGET].astype(float)
        exog = history[self.exog_cols] if self.exog_cols else None
        self.models, n_rows = {}, []
        for h in range(1, self.horizon + 1):
            rows = self._rows(y, h, exog).dropna()   # drops targets after cutoff
            last_target = rows.index.max() + pd.Timedelta(hours=h)
            assert last_target <= history.index[-1], "leakage: target after cutoff"
            self.feature_cols = [c for c in rows.columns if c != "target"]
            m = self._new_model()
            if self.params:
                m.set_params(**self.params)
            m.fit(rows[self.feature_cols], rows["target"])
            self.models[h] = m
            n_rows.append(len(rows))
        self.n_train_rows_ = int(np.mean(n_rows))
        self.fit_cutoff_ = history.index[-1]
        return self

    def predict(self, history, future):
        if not self.models:
            raise RuntimeError("fit() before predict()")
        y = history[TARGET].astype(float)
        t0 = y.index[-1]
        preds = []
        for h, ts in enumerate(future.index, 1):
            feats = direct_origin_features(y.iloc[-200:]).iloc[[-1]]
            feats["y_same_hour_1d"] = y.get(ts - pd.Timedelta(hours=24), np.nan)
            feats["y_same_hour_1w"] = y.get(ts - pd.Timedelta(hours=168), np.nan)
            for c in CALENDAR_COLS + self.exog_cols:
                feats[c] = future.at[ts, c]
            assert feats.index[0] == t0
            preds.append(float(self.models[h].predict(feats[self.feature_cols])[0]))
        return self._frame(future, preds)


def protocol_calendar(index):
    from .protocol import calendar_frame
    return calendar_frame(index)


# --------------------------------------------------------------------------
# SARIMAX - fixed order, parameters estimated at fit(), state updated at
# every origin by filtering the observed history (no re-estimation)
# --------------------------------------------------------------------------
class SARIMAXForecaster(Forecaster):
    name = "sarimax"
    trainable = True

    def __init__(self, order, seasonal_order=(1, 0, 1, config.DAILY_SEASON),
                 scenario: str = "operational",
                 train_days: int = config.SARIMAX_TRAIN_DAYS,
                 maxiter: int = 200, order_source: str = "unspecified"):
        from . import sarimax_model  # lazy: statsmodels only needed if used
        self._sm = sarimax_model
        self.order, self.seasonal_order = tuple(order), tuple(seasonal_order)
        self.scenario, self.train_days, self.maxiter = scenario, train_days, maxiter
        self.order_source = order_source
        allowed = SCENARIOS[scenario]["future_columns"]
        self.exog_cols = ([c for c in sarimax_model.EXOG_COLS if c in allowed]
                          or None)
        self.res = None
        self.deviations = [
            f"Order {self.order}x{self.seasonal_order} from: {order_source}.",
            "Exogenous T_out/RH_out kept (original config) - weather_oracle."
            if self.exog_cols else
            "Exogenous T_out/RH_out removed: future weather is not known in "
            "the operational scenario.",
            "Parameters estimated once per fit; at each origin the state is "
            "updated by Kalman filtering the observed history with fixed "
            "parameters (statsmodels results.apply(refit=False)).",
        ]

    def _window(self, history):
        w = history.iloc[-self.train_days * 24:]
        y = w[TARGET].astype(float).asfreq(config.FREQ)
        x = None
        if self.exog_cols:
            x = w[self.exog_cols].asfreq(config.FREQ)
            if x.isna().any().any():
                raise ValueError("SARIMAX exogenous history contains NaN")
        return y, x

    def fit(self, history):
        from .sarimax_selection import fit_with_diagnostics
        y, x = self._window(history)
        res, info = fit_with_diagnostics(y, self.order, self.seasonal_order, x,
                                         maxiter=self.maxiter)
        if res is None:
            raise RuntimeError(f"SARIMAX{self.order} fit failed: {info['error']}")
        self.res = res
        self.n_train_rows_ = int(len(y))
        self.converged_ = info["converged"]
        self.n_warnings_ = info["n_warnings"]
        self.warnings_ = info["warnings"]
        if not self.converged_:
            print(f"[sarimax] WARNING: SARIMAX{self.order}x{self.seasonal_order} "
                  f"did not converge (maxiter={self.maxiter}); flagged in "
                  "fit_log.csv", file=sys.stderr)
        return self

    def predict(self, history, future):
        if self.res is None:
            raise RuntimeError("fit() before predict()")
        y, x = self._window(history)
        res = self.res.apply(endog=y, exog=x, refit=False)
        x_future = future[self.exog_cols] if self.exog_cols else None
        fc = res.get_forecast(steps=len(future), exog=x_future)
        mean = np.asarray(fc.predicted_mean, dtype=float)
        ci = np.asarray(fc.conf_int(alpha=0.05), dtype=float)
        return self._frame(future, np.clip(mean, 0, None),
                           np.clip(ci[:, 0], 0, None), ci[:, 1], 0.95)


# --------------------------------------------------------------------------
# Chronos - zero-shot, context = observed history up to the origin
# --------------------------------------------------------------------------
class ChronosForecaster(Forecaster):
    name = "chronos"
    trainable = False

    def __init__(self, model_name: str | None = None, num_samples: int = 20):
        from . import foundation
        self._fd = foundation
        self.model_name = model_name or foundation.MODEL_NAME
        self.num_samples = num_samples
        self.deviations = [
            "Context = full observed history up to each origin; chronos-t5 "
            "uses at most its last 512 values internally (unchanged).",
            "Interval is the 10-90% sample quantile band (80%), labelled as such.",
        ]
        if num_samples != 20:
            self.deviations.append(f"num_samples={num_samples} (original 20)")

    def predict(self, history, future):
        ctx = history[TARGET].to_numpy(dtype=float)
        med, lo, hi, device = self._fd._chronos_forecast(
            ctx, len(future), self.model_name, num_samples=self.num_samples)
        self.device_ = device
        return self._frame(future, np.clip(med, 0, None),
                           np.clip(lo, 0, None), np.clip(hi, 0, None), 0.80)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
BENCHMARK_NAMES = ["mean", "naive", "drift",
                   "seasonal_naive_24", "seasonal_naive_168"]
ALL_MODEL_NAMES = BENCHMARK_NAMES + ["sarimax", "xgboost", "xgboost_direct", "chronos"]


def build(name: str, scenario: str = "operational", **kw) -> Forecaster:
    if name == "mean":
        return mean_model()
    if name == "naive":
        return naive_model()
    if name == "drift":
        return drift_model()
    if name == "seasonal_naive_24":
        return seasonal_naive_model(config.DAILY_SEASON)
    if name == "seasonal_naive_168":
        return seasonal_naive_model(config.WEEKLY_SEASON)
    if name == "xgboost":
        return XGBoostRecursive(scenario, params=kw.get("xgb_params"))
    if name == "xgboost_direct":
        return XGBoostDirect(scenario, params=kw.get("xgb_params"))
    if name == "sarimax":
        if kw.get("sarimax_order") is None:
            raise ValueError(
                "SARIMAX needs an explicit order. Select one first "
                "(run_backtest.py --select-sarimax-order, which uses training "
                "and validation data only) and pass --sarimax-order-file, or "
                "pass --sarimax-order p,d,q.")
        return SARIMAXForecaster(
            kw["sarimax_order"],
            kw.get("sarimax_seasonal_order", (1, 0, 1, config.DAILY_SEASON)),
            scenario=scenario, maxiter=kw.get("sarimax_maxiter", 200),
            order_source=kw.get("sarimax_order_source", "command line"))
    if name == "chronos":
        return ChronosForecaster(kw.get("chronos_model"),
                                 kw.get("chronos_num_samples", 20))
    raise ValueError(f"unknown model {name!r}; choose from {ALL_MODEL_NAMES}")
