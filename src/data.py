"""Data acquisition, hourly aggregation and completeness checks.

Downloads the UCI Appliances Energy Prediction dataset (10-minute sampling),
parses the timestamp, checks for missing values and resamples to hourly.

If the machine has no internet access, `make_synthetic()` produces a series
with the same structure (daily + weekly cycles + noise) so the pipeline and
tests can still run end to end.
"""
from __future__ import annotations

import io
import sys
import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def download(force: bool = False) -> pd.DataFrame:
    """Fetch the raw 10-minute CSV, caching it under data/raw/."""
    if config.RAW_CSV.exists() and not force:
        return pd.read_csv(config.RAW_CSV)

    import requests
    last_err = None
    for url in config.DATA_MIRRORS:
        try:
            print(f"[data] downloading {url}", file=sys.stderr)
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            config.RAW_CSV.write_text(r.text)
            return df
        except Exception as e:  # try the next mirror
            last_err = e
            print(f"[data] failed: {e}", file=sys.stderr)
    raise RuntimeError(f"Could not download dataset. Last error: {last_err}")


# --------------------------------------------------------------------------
# Prepare / resample
# --------------------------------------------------------------------------
def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the timestamp and bin the 10-minute data up to hourly values.

    * `Appliances` and `lights` are energy in Wh -> summed over the hour.
    * All sensor / weather channels are instantaneous -> averaged over the hour.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # drop the two random variables shipped for a benchmarking exercise
    df = df.drop(columns=[c for c in ("rv1", "rv2") if c in df.columns])

    energy_cols = [c for c in ("Appliances", "lights") if c in df.columns]
    mean_cols = [c for c in df.columns if c not in energy_cols]

    agg = {c: "sum" for c in energy_cols}
    agg.update({c: "mean" for c in mean_cols})
    hourly = df.resample(config.FREQ).agg(agg)
    return hourly


# --------------------------------------------------------------------------
# Completeness of hourly bins (Phase 2A fix for the partial final hour)
# --------------------------------------------------------------------------
ENERGY_COLS = ("Appliances", "lights")


def infer_readings_per_hour(timestamps) -> tuple[int, pd.Timedelta]:
    """Infer the raw sampling interval and how many readings fill one hour.

    The interval is the most common gap between consecutive timestamps. It
    must divide an hour exactly, otherwise "a complete hour" is undefined.
    """
    ts = pd.Series(pd.to_datetime(pd.Index(timestamps))).sort_values()
    diffs = ts.diff().dropna()
    if diffs.empty:
        raise ValueError("Need at least two timestamps to infer sampling.")
    step = diffs.mode().iloc[0]
    hour = pd.Timedelta(hours=1)
    if step <= pd.Timedelta(0) or hour % step != pd.Timedelta(0):
        raise ValueError(f"Sampling interval {step} does not divide one hour.")
    return int(hour / step), step


def to_hourly_checked(df: pd.DataFrame, target: str = config.TARGET):
    """Aggregate raw readings to hourly AND flag incomplete hours.

    Returns ``(hourly, counts, expected)`` where ``counts`` is the number of
    non-missing target readings that fell in each hourly bin (0 for an empty
    bin) and ``expected`` is the number of readings in a full hour.

    Why this matters: energy columns are *summed*, so an hour built from 1 of
    6 readings understates consumption by ~5/6. Averaged sensor columns are
    still unbiased means of the readings that exist.
    """
    raw = df.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    if raw["date"].duplicated().any():
        n = int(raw["date"].duplicated().sum())
        raise ValueError(f"{n} duplicate raw timestamps; resolve before use.")
    expected, _step = infer_readings_per_hour(raw["date"])
    hourly = to_hourly(raw)  # unchanged aggregation rules
    counts = (raw.set_index("date").sort_index()[target]
              .resample(config.FREQ).count()
              .reindex(hourly.index, fill_value=0))
    return hourly, counts, expected


def drop_incomplete_hours(hourly: pd.DataFrame, counts: pd.Series,
                          expected: int) -> tuple[pd.DataFrame, dict]:
    """Exclude hours with fewer than ``expected`` readings from modelling.

    * Incomplete hours at the start/end of the series are trimmed away.
    * Incomplete hours in the interior cannot be deleted without breaking the
      regular hourly grid that lags rely on, so their energy columns are set
      to NaN. Forecast origins whose history or target window touches such an
      hour are then ineligible (see ``protocol.eligible_origins``).
    """
    counts = counts.reindex(hourly.index, fill_value=0)
    incomplete = counts < expected
    complete_idx = hourly.index[~incomplete.values]
    if len(complete_idx) == 0:
        raise ValueError("No complete hours in the data.")
    first, last = complete_idx.min(), complete_idx.max()

    out = hourly.loc[first:last].copy()
    interior = incomplete.loc[first:last]
    for col in ENERGY_COLS:
        if col in out.columns and interior.any():  # keep dtype if nothing to mask
            out[col] = out[col].astype(float)
            out.loc[interior.values, col] = np.nan

    edge_mask = incomplete & ((hourly.index < first) | (hourly.index > last))
    report = {
        "expected_readings_per_hour": int(expected),
        "hourly_bins_before": int(len(hourly)),
        "hourly_bins_after": int(len(out)),
        "incomplete_hours_total": int(incomplete.sum()),
        "incomplete_hours_trimmed_at_edges": int(edge_mask.sum()),
        "incomplete_hours_masked_interior": int(interior.sum()),
        "complete_hours_used": int((~interior).sum()),
        "incomplete_hours": {str(ts): int(c)
                             for ts, c in counts[incomplete].items()},
        "first_hour": str(first), "last_hour": str(last),
    }
    return out, report


def load_checked(force: bool = False, synthetic: bool = False,
                 n_synthetic_days: int = 137):
    """Strict loader for the backtest: no silent synthetic fallback.

    Returns ``(hourly, report)``. Covariates are NOT interpolated, so no value
    is ever filled in from the future; missing energy stays NaN.
    """
    if synthetic:
        hourly = make_synthetic(n_days=n_synthetic_days)
        report = {"expected_readings_per_hour": None, "source": "synthetic",
                  "hourly_bins_before": len(hourly),
                  "hourly_bins_after": len(hourly),
                  "incomplete_hours_total": 0,
                  "incomplete_hours_trimmed_at_edges": 0,
                  "incomplete_hours_masked_interior": 0,
                  "complete_hours_used": len(hourly), "incomplete_hours": {}}
        return hourly, report
    raw = download(force=force)
    hourly, counts, expected = to_hourly_checked(raw)
    hourly, report = drop_incomplete_hours(hourly, counts, expected)
    report["source"] = config.RAW_CSV.relative_to(config.ROOT).as_posix()
    report["raw_rows"] = int(len(raw))
    return hourly.asfreq(config.FREQ), report


def missing_report(df: pd.DataFrame) -> pd.Series:
    """Count missing values per column plus any gaps in the hourly index."""
    gaps = pd.date_range(df.index.min(), df.index.max(), freq=config.FREQ)
    n_index_gaps = len(gaps) - len(df)
    report = df.isna().sum()
    report["__index_gaps__"] = n_index_gaps
    return report


def load(force: bool = False, synthetic: bool = False,
         drop_incomplete: bool = True) -> pd.DataFrame:
    """One-call entry point used by the legacy single-window pipeline.

    Phase 2A change: hours with fewer readings than a full hour are excluded
    (``drop_incomplete=True``). On the UCI file this removes exactly one bin,
    2016-05-27 18:00, which held 1 of 6 readings. Pass
    ``drop_incomplete=False`` to reproduce the historical (pre-fix) series.
    Energy columns are never interpolated; covariates are, as before.
    """
    if synthetic:
        return make_synthetic()
    try:
        raw = download(force=force)
        if drop_incomplete:
            hourly, counts, expected = to_hourly_checked(raw)
            hourly, report = drop_incomplete_hours(hourly, counts, expected)
            print(f"[data] removed {report['incomplete_hours_total']} "
                  f"incomplete hour(s): {report['incomplete_hours']}",
                  file=sys.stderr)
        else:
            hourly = to_hourly(raw)
        hourly = hourly.asfreq(config.FREQ)
        cov = [c for c in hourly.columns if c not in ENERGY_COLS]
        hourly[cov] = hourly[cov].interpolate(limit_direction="both")
        if not drop_incomplete:  # historical behaviour: interpolate everything
            hourly = hourly.interpolate(limit_direction="both")
        return hourly
    except Exception as e:
        print(f"[data] falling back to synthetic data: {e}", file=sys.stderr)
        return make_synthetic()


# --------------------------------------------------------------------------
# Synthetic fallback (structure mirrors the real series)
# --------------------------------------------------------------------------
def make_synthetic(n_days: int = 137) -> pd.DataFrame:
    """Hourly series with daily + weekly seasonality, useful for tests."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    idx = pd.date_range("2016-01-11", periods=n_days * 24, freq="h")
    t = np.arange(len(idx))
    daily = 40 * np.sin(2 * np.pi * (t % 24) / 24 - 1.2).clip(min=-0.3)
    weekly = 15 * np.sin(2 * np.pi * (t % 168) / 168)
    base = 90 + daily + weekly
    spikes = rng.random(len(idx)) < 0.05
    noise = rng.gamma(2.0, 15.0, len(idx))
    appliances = np.clip(base + noise + spikes * rng.gamma(3, 60, len(idx)), 10, None)

    temp_out = 6 + 8 * np.sin(2 * np.pi * (t % 24) / 24 - 2.0) + rng.normal(0, 1.5, len(idx))
    rh_out = 80 - 1.5 * temp_out + rng.normal(0, 4, len(idx))
    t1 = 20 + 2 * np.sin(2 * np.pi * (t % 24) / 24) + rng.normal(0, 0.5, len(idx))
    rh1 = 40 + rng.normal(0, 3, len(idx))

    out = pd.DataFrame(
        {
            "Appliances": appliances,
            "lights": rng.integers(0, 40, len(idx)).astype(float),
            "T1": t1, "RH_1": rh1,
            "T2": t1 - 1, "RH_2": rh1 + 2,
            "T_out": temp_out, "RH_out": rh_out.clip(1, 100),
            "Press_mm_hg": 755 + rng.normal(0, 5, len(idx)),
            "Windspeed": np.abs(rng.normal(4, 2, len(idx))),
            "Tdewpoint": temp_out - 4 + rng.normal(0, 1, len(idx)),
        },
        index=idx,
    )
    # Phase 2B: Visibility completes the outdoor-weather set used by the
    # weather_oracle scenario. Drawn last so all other columns are unchanged.
    out["Visibility"] = np.clip(40 + rng.normal(0, 10, len(idx)), 1, 66)
    return out
