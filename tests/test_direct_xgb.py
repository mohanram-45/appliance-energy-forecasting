"""Direct multi-horizon XGBoost: training cutoff, feature alignment, leakage."""
import importlib.util

import numpy as np
import pandas as pd
import pytest

from src import backtest, config, data, forecasters, protocol

pytestmark = pytest.mark.skipif(importlib.util.find_spec("xgboost") is None,
                                reason="xgboost not installed")
FAST = {"n_estimators": 20, "max_depth": 3, "n_jobs": 1}


@pytest.fixture(scope="module")
def df():
    return data.make_synthetic(n_days=45).asfreq("h")


@pytest.fixture(scope="module")
def parts(df):
    return protocol.make_partitions(df, n_test=3, n_val=2)


def test_training_targets_never_after_cutoff(df, parts, monkeypatch):
    m = forecasters.build("xgboost_direct", xgb_params=FAST)
    seen = {}
    real = m._rows

    def spy(y, h, exog):
        rows = real(y, h, exog)
        kept = rows.dropna()
        seen[h] = kept.index.max() + pd.Timedelta(hours=h)
        return rows
    monkeypatch.setattr(m, "_rows", spy)
    origin = parts.test[0]
    m.fit(protocol.history_view(df, origin))
    assert len(seen) == 24 and len(m.models) == 24
    assert all(last <= origin for last in seen.values())
    assert max(seen.values()) == origin            # uses data right up to it


def test_prediction_features_match_training_rows(df, parts):
    """The row the model sees at forecast time equals the training row it would
    have had for the same origin - so train/predict features are aligned."""
    m = forecasters.build("xgboost_direct", xgb_params=FAST)
    origin = parts.test[1]
    m.fit(protocol.history_view(df, parts.test[0]))
    y_full = df[config.TARGET].astype(float)
    hist = protocol.history_view(df, origin)
    fut = protocol.future_view(df, origin)
    for h in (1, 7, 24):
        train_row = m._rows(y_full, h, None).loc[[origin], m.feature_cols]
        ts = fut.index[h - 1]
        y = hist[config.TARGET].astype(float)
        feats = forecasters.direct_origin_features(y.iloc[-200:]).iloc[[-1]]
        feats["y_same_hour_1d"] = y.get(ts - pd.Timedelta(hours=24))
        feats["y_same_hour_1w"] = y.get(ts - pd.Timedelta(hours=168))
        for c in protocol.CALENDAR_COLS:
            feats[c] = fut.at[ts, c]
        np.testing.assert_allclose(feats[m.feature_cols].to_numpy(float),
                                   train_row.to_numpy(float))
        # seasonal lags of the target hour are always in the observed past
        assert ts - pd.Timedelta(hours=24) <= origin


@pytest.mark.parametrize("scenario", ["operational", "weather_oracle"])
def test_direct_forecast_unchanged_by_future_data(df, parts, scenario):
    origin = parts.test[0]
    keep = protocol.SCENARIOS[scenario]["future_columns"]
    d2 = df.copy()
    after = d2.index > origin
    rng = np.random.default_rng(3)
    for c in d2.columns:
        if c not in keep:
            d2.loc[after, c] = rng.normal(9e3, 1e3, after.sum())

    def run(d):
        m = {"xgboost_direct": forecasters.build("xgboost_direct", scenario,
                                                 xgb_params=FAST)}
        return backtest.run_backtest(d, m, [origin], scenario=scenario,
                                     verbose=False)["forecasts"]["yhat"]
    pd.testing.assert_series_equal(run(df), run(d2))


def test_direct_alignment_and_identical_targets(df, parts):
    m = {"xgboost": forecasters.build("xgboost", xgb_params=FAST),
         "xgboost_direct": forecasters.build("xgboost_direct", xgb_params=FAST),
         "seasonal_naive_24": forecasters.build("seasonal_naive_24")}
    f = backtest.run_backtest(df, m, parts.test, verbose=False)["forecasts"]
    for name, g in f.groupby("model"):
        assert len(g) == 24 * len(parts.test), name
    backtest.check_identical_targets(f)
