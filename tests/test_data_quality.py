"""Incomplete-hour detection and exclusion (Phase 2A, fix B1)."""
import numpy as np
import pandas as pd
import pytest

from src import config, data


def _raw(start, periods, freq="10min", seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=periods, freq=freq)
    return pd.DataFrame({
        "date": idx.strftime("%Y-%m-%d %H:%M:%S"),
        "Appliances": rng.integers(10, 100, periods).astype(float),
        "lights": rng.integers(0, 30, periods).astype(float),
        "T1": rng.normal(20, 1, periods),
        "T_out": rng.normal(5, 2, periods),
        "rv1": 0.0, "rv2": 0.0,
    })


def test_infer_readings_per_hour():
    assert data.infer_readings_per_hour(
        pd.date_range("2016-01-01", periods=50, freq="10min"))[0] == 6
    assert data.infer_readings_per_hour(
        pd.date_range("2016-01-01", periods=50, freq="15min"))[0] == 4
    with pytest.raises(ValueError):
        data.infer_readings_per_hour(
            pd.date_range("2016-01-01", periods=50, freq="7min"))


def test_duplicate_timestamps_rejected():
    raw = _raw("2016-01-01 00:00", 18)
    raw = pd.concat([raw, raw.iloc[[3]]])
    with pytest.raises(ValueError, match="duplicate"):
        data.to_hourly_checked(raw)


def test_edge_and_interior_incomplete_hours():
    # starts at 00:30 (hour 00 has 3 readings), ends at 05:00 (hour 05 has 1)
    raw = _raw("2016-01-01 00:30", 28)
    # remove two readings from hour 02 -> interior incomplete hour (4 of 6)
    raw = raw[~raw["date"].isin(["2016-01-01 02:10:00", "2016-01-01 02:20:00"])]
    hourly, counts, expected = data.to_hourly_checked(raw)
    assert expected == 6
    assert counts.loc["2016-01-01 00:00"] == 3
    assert counts.loc["2016-01-01 02:00"] == 4
    assert counts.loc["2016-01-01 05:00"] == 1

    clean, rep = data.drop_incomplete_hours(hourly, counts, expected)
    assert clean.index[0] == pd.Timestamp("2016-01-01 01:00")
    assert clean.index[-1] == pd.Timestamp("2016-01-01 04:00")
    assert rep["incomplete_hours_total"] == 3
    assert rep["incomplete_hours_trimmed_at_edges"] == 2
    assert rep["incomplete_hours_masked_interior"] == 1
    # interior incomplete hour: energy masked, sensors kept (unbiased means)
    assert np.isnan(clean.loc["2016-01-01 02:00", "Appliances"])
    assert np.isnan(clean.loc["2016-01-01 02:00", "lights"])
    assert not np.isnan(clean.loc["2016-01-01 02:00", "T1"])
    # complete hours: sum of the six raw readings
    r = raw.assign(date=pd.to_datetime(raw["date"])).set_index("date")
    assert clean.loc["2016-01-01 03:00", "Appliances"] == \
        r.loc["2016-01-01 03:00":"2016-01-01 03:50", "Appliances"].sum()


def test_missing_target_reading_counts_as_incomplete():
    raw = _raw("2016-01-01 00:00", 18)
    raw.loc[7, "Appliances"] = np.nan          # hour 01 now has 5 readings
    hourly, counts, expected = data.to_hourly_checked(raw)
    clean, rep = data.drop_incomplete_hours(hourly, counts, expected)
    assert rep["incomplete_hours_masked_interior"] == 1
    assert np.isnan(clean.loc["2016-01-01 01:00", "Appliances"])


@pytest.mark.skipif(not config.RAW_CSV.exists(), reason="UCI csv not present")
def test_real_data_only_final_hour_is_incomplete():
    hourly, rep = data.load_checked()
    assert rep["expected_readings_per_hour"] == 6
    assert rep["incomplete_hours"] == {"2016-05-27 18:00:00": 1}
    assert rep["hourly_bins_before"] - rep["hourly_bins_after"] == 1
    assert hourly.index[-1] == pd.Timestamp("2016-05-27 17:00")
    assert hourly[config.TARGET].notna().all()


@pytest.mark.skipif(not config.RAW_CSV.exists(), reason="UCI csv not present")
def test_legacy_load_fixed_and_historical_switch():
    fixed = data.load()
    hist = data.load(drop_incomplete=False)
    assert fixed.index[-1] == pd.Timestamp("2016-05-27 17:00")
    assert hist.index[-1] == pd.Timestamp("2016-05-27 18:00")
    assert hist.loc["2016-05-27 18:00", config.TARGET] == 430  # the partial bin
    pd.testing.assert_frame_equal(fixed, hist.iloc[:-1], check_freq=False)
