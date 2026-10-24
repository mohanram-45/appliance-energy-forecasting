"""Rolling-origin backtest: leakage protection, identical targets, refit policy,
and an end-to-end synthetic smoke run of the CLI."""
import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from src import backtest, config, data, forecasters, protocol

HAS_XGB = importlib.util.find_spec("xgboost") is not None
HAS_SM = importlib.util.find_spec("statsmodels") is not None
HAS_CHRONOS = (importlib.util.find_spec("torch") is not None
               and importlib.util.find_spec("chronos") is not None)
FAST_XGB = {"n_estimators": 25, "max_depth": 3, "n_jobs": 1}


@pytest.fixture(scope="module")
def df():
    d = data.make_synthetic(n_days=60)
    d["Visibility"] = 40.0
    return d.asfreq("h")


@pytest.fixture(scope="module")
def parts(df):
    return protocol.make_partitions(df, n_test=4, n_val=2)


def _models(include_slow=True):
    m = {n: forecasters.build(n) for n in forecasters.BENCHMARK_NAMES}
    if include_slow and HAS_XGB:
        m["xgboost"] = forecasters.build("xgboost", xgb_params=FAST_XGB)
    if include_slow and HAS_SM:
        m["sarimax"] = forecasters.build("sarimax", sarimax_order=(1, 0, 0),
                                         sarimax_seasonal_order=(0, 0, 0, 0))
    return m


def _perturb_after(df, origin, keep=()):
    """Replace every non-calendar value after the origin with garbage."""
    d = df.copy()
    rng = np.random.default_rng(1)
    after = d.index > origin
    for c in d.columns:
        if c in keep:
            continue
        d.loc[after, c] = rng.normal(1e4, 5e3, after.sum())
    return d


def test_benchmark_values_and_alignment(df, parts):
    res = backtest.run_backtest(df, _models(False), parts.test, verbose=False)
    f = res["forecasts"]
    o = parts.test[0]
    sn = f[(f.model == "seasonal_naive_24") & (f.origin == o)]
    prev_day = df.loc[o - pd.Timedelta(hours=23):o, config.TARGET].to_numpy()
    np.testing.assert_allclose(sn["yhat"], prev_day)
    sw = f[(f.model == "seasonal_naive_168") & (f.origin == o)]
    prev_week = df.loc[sw.target_time - pd.Timedelta(hours=168), config.TARGET]
    np.testing.assert_allclose(sw["yhat"], prev_week.to_numpy())
    # y_true is the actual value at each target timestamp
    np.testing.assert_allclose(sn["y_true"], df.loc[sn.target_time, config.TARGET])
    assert (sn["step"].to_numpy() == np.arange(1, 25)).all()


@pytest.mark.parametrize("scenario", ["operational", "weather_oracle"])
def test_no_future_information_reaches_any_model(df, parts, scenario):
    """Forecasts at an origin must not change when everything after it is
    replaced by garbage (except what the scenario explicitly allows)."""
    origin = parts.test[1]
    keep = protocol.SCENARIOS[scenario]["future_columns"]
    clean = backtest.run_backtest(df, _models(), [origin], scenario=scenario,
                                  verbose=False)["forecasts"]
    dirty_df = _perturb_after(df, origin, keep=keep)
    dirty = backtest.run_backtest(dirty_df, _models(), [origin], scenario=scenario,
                                  verbose=False)["forecasts"]
    pd.testing.assert_series_equal(clean["yhat"], dirty["yhat"])
    # sanity: the truth did change, so the test has teeth
    assert not np.allclose(clean["y_true"], dirty["y_true"])


@pytest.mark.skipif(not HAS_XGB, reason="xgboost not installed")
def test_xgboost_recursion_feeds_predictions_not_truth(df, parts, monkeypatch):
    from src import features as F
    seen = []
    orig = F.target_history_features

    def spy(y_past, ts, **kw):
        seen.append((y_past.index.max(), pd.Timestamp(ts)))
        return orig(y_past, ts, **kw)
    monkeypatch.setattr(F, "target_history_features", spy)
    origin = parts.test[0]
    m = forecasters.build("xgboost", xgb_params=FAST_XGB).fit(
        protocol.history_view(df, origin))
    m.predict(protocol.history_view(df, origin), protocol.future_view(df, origin))
    assert len(seen) == 24
    assert all(last < ts for last, ts in seen)


def test_identical_origins_and_targets_enforced(df, parts):
    res = backtest.run_backtest(df, _models(), parts.test, verbose=False)
    f = res["forecasts"]
    keys = {m: tuple(zip(g.origin, g.target_time)) for m, g in f.groupby("model")}
    assert len(set(keys.values())) == 1
    bad = f[~((f.model == "mean") & (f.step == 24))]
    with pytest.raises(AssertionError):
        backtest.check_identical_targets(bad)


@pytest.mark.skipif(not HAS_XGB, reason="xgboost not installed")
def test_refit_policy(df, parts):
    origins = parts.test  # 4 origins
    m = {"xgboost": forecasters.build("xgboost", xgb_params=FAST_XGB)}
    log = backtest.run_backtest(df, m, origins, refit_every=0, verbose=False)["fit_log"]
    assert len(log) == 1 and log.fit_data_end.iloc[0] == origins[0]
    m = {"xgboost": forecasters.build("xgboost", xgb_params=FAST_XGB)}
    log = backtest.run_backtest(df, m, origins, refit_every=2, verbose=False)["fit_log"]
    assert list(log.fit_data_end) == [origins[0], origins[2]]
    # untrained models never fit
    log = backtest.run_backtest(df, _models(False), origins, refit_every=1,
                                verbose=False)["fit_log"]
    assert log.empty


def test_window_and_aggregate_metrics(df, parts):
    res = backtest.run_backtest(df, _models(False), parts.test, verbose=False)
    w = backtest.window_metrics(res["forecasts"], df)
    assert len(w) == len(forecasters.BENCHMARK_NAMES) * len(parts.test)
    a = backtest.aggregate_metrics(res["forecasts"], w)
    assert (a["n_windows"] == len(parts.test)).all()
    assert (a["n_points"] == 24 * len(parts.test)).all()
    # pooled RMSE == sqrt(mean of per-window MSE) when windows are equal size
    for m in a["model"]:
        ww = w[w.model == m]
        assert a.set_index("model").loc[m, "pooled_RMSE"] == \
            pytest.approx(np.sqrt(np.mean(ww["RMSE"] ** 2)))
    h = backtest.horizon_metrics(res["forecasts"])
    assert set(h["step"]) == set(range(1, 25))


def test_output_dir_protection(tmp_path):
    with pytest.raises(ValueError):
        backtest.prepare_run_dir("x", "operational", base=config.METRIC_DIR)
    with pytest.raises(ValueError):
        backtest.prepare_run_dir("..", "operational", base=tmp_path)
    out = backtest.prepare_run_dir("run", "operational", base=tmp_path)
    assert out == (tmp_path / "operational" / "run").resolve()
    with pytest.raises(FileExistsError):       # never reused, even if empty
        backtest.prepare_run_dir("run", "operational", base=tmp_path)


def test_sarimax_requires_explicit_order():
    with pytest.raises(ValueError, match="explicit order"):
        forecasters.build("sarimax")


def test_cli_synthetic_smoke(tmp_path):
    import run_backtest
    models = "benchmarks,xgboost" if HAS_XGB else "benchmarks"
    rc = run_backtest.main([
        "--synthetic", "--synthetic-days", "60", "--n-test-origins", "3",
        "--n-val-origins", "2", "--models", models,
        "--xgb-params", ",".join(f"{k}={v}" for k, v in FAST_XGB.items()),
        "--partition", "test",
        "--output-dir", str(tmp_path), "--run-name", "smoke"])
    assert rc == 0
    out = tmp_path / "operational" / "smoke"
    for f in ("forecasts.csv", "window_metrics.csv", "aggregate_metrics.csv",
              "horizon_metrics.csv", "fit_log.csv", "run_config.json"):
        assert (out / f).exists(), f
    cfg = json.loads((out / "run_config.json").read_text())
    assert cfg["synthetic"] is True and len(cfg["origins"]) == 3
    agg = pd.read_csv(out / "aggregate_metrics.csv")
    assert (agg["n_windows"] == 3).all()


@pytest.mark.skipif(not HAS_CHRONOS, reason="torch + chronos not installed")
def test_chronos_adapter_alignment(df, parts):
    origin = parts.test[0]
    m = forecasters.build("chronos")
    p = m.predict(protocol.history_view(df, origin), protocol.future_view(df, origin))
    assert p.index.equals(protocol.target_index(origin))
    assert (p["interval_level"] == 0.80).all()
