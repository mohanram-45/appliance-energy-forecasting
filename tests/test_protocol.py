"""Forecast origins, partitions, timestamp alignment and feature availability."""
import numpy as np
import pandas as pd
import pytest

from src import config, data, features as F, protocol


@pytest.fixture(scope="module")
def df():
    d = data.make_synthetic(n_days=70)
    d["Visibility"] = 40.0
    return d.asfreq("h")


def test_target_index_alignment():
    idx = protocol.target_index("2016-03-01 23:00", 24)
    assert len(idx) == 24
    assert idx[0] == pd.Timestamp("2016-03-02 00:00")
    assert idx[-1] == pd.Timestamp("2016-03-02 23:00")
    assert (np.diff(idx.asi8) == 3_600_000_000_000).all()


def test_history_view_stops_at_origin(df):
    origin = df.index[500]
    h = protocol.history_view(df, origin)
    assert h.index[-1] == origin
    h.iloc[-1, 0] = -999                       # a copy: master frame untouched
    assert df.iloc[500, 0] != -999


def test_future_view_operational_is_calendar_only(df):
    fut = protocol.future_view(df, df.index[500], 24, "operational")
    assert list(fut.columns) == protocol.CALENDAR_COLS
    assert fut.index.equals(protocol.target_index(df.index[500], 24))
    # identical to the training-time calendar features
    ref = F.add_time_features(df.loc[fut.index, [config.TARGET]])
    pd.testing.assert_frame_equal(fut, ref[protocol.CALENDAR_COLS],
                                  check_dtype=False, check_freq=False)


def test_future_view_never_contains_target_or_indoor(df):
    for sc in protocol.SCENARIOS:
        fut = protocol.future_view(df, df.index[500], 24, sc)
        assert not set(fut.columns) & set(protocol.NEVER_FUTURE_COLS)
    oracle = protocol.future_view(df, df.index[500], 24, "weather_oracle")
    assert set(protocol.OUTDOOR_WEATHER_COLS) <= set(oracle.columns)
    with pytest.raises(AssertionError, match="leakage"):
        protocol.assert_no_forbidden_future_columns(
            pd.DataFrame({config.TARGET: [1.0]}))
    with pytest.raises(ValueError):
        protocol.future_view(df, df.index[500], 24, "nonsense")


def test_eligible_origins_respect_gaps(df):
    d = df.copy()
    origins = protocol.eligible_origins(d)
    assert (origins.hour == config.ORIGIN_HOUR).all()
    assert origins[0] - d.index[0] >= pd.Timedelta(hours=config.MAX_LOOKBACK_HOURS - 1)
    assert origins[-1] + pd.Timedelta(hours=24) <= d.index[-1]
    # mask one hour -> the origin whose window contains it, and the origins
    # whose 168 h look-back contains it, become ineligible
    hole = origins[20] + pd.Timedelta(hours=5)
    d.loc[hole, config.TARGET] = np.nan
    after = protocol.eligible_origins(d)
    assert origins[20] not in after
    for o in after:
        assert not (o - pd.Timedelta(hours=167) <= hole <= o + pd.Timedelta(hours=24))


def test_partitions_are_chronological_and_disjoint(df):
    p = protocol.make_partitions(df, n_test=10, n_val=5)
    assert len(p.test) == 10 and len(p.validation) == 5
    assert p.validation[-1] < p.test[0]
    last_val_target = p.validation[-1] + pd.Timedelta(hours=24)
    assert last_val_target <= p.test[0]          # no shared target hours
    assert p.train_end == p.validation[0]
    assert (np.diff(p.test.asi8) == 24 * 3_600_000_000_000).all()
    with pytest.raises(ValueError):
        protocol.make_partitions(df, n_test=200, n_val=5)


def test_target_history_features_match_frame_and_guard(df):
    frame = F.build_feature_frame(df[[config.TARGET]], dropna=False)
    y = df[config.TARGET]
    for ts in [df.index[300], df.index[777], df.index[1200]]:
        got = F.target_history_features(y.loc[:ts - pd.Timedelta(hours=1)], ts)
        for k, v in got.items():
            assert v == pytest.approx(frame.loc[ts, k]), k
    with pytest.raises(AssertionError, match="leakage"):
        F.target_history_features(y.loc[:df.index[300]], df.index[300])
