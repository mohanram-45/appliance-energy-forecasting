"""Phase 2B: SARIMAX selection, convergence reporting, scenario separation,
legacy-output protection, unique run folders, Chronos input/alignment and the
XGBoost recursion diagnostic."""
import importlib.util
import json
import warnings

import numpy as np
import pandas as pd
import pytest

from src import backtest, config, data, forecasters, protocol

HAS_XGB = importlib.util.find_spec("xgboost") is not None
HAS_SM = importlib.util.find_spec("statsmodels") is not None
FAST_XGB = {"n_estimators": 25, "max_depth": 3, "n_jobs": 1}


@pytest.fixture(scope="module")
def df():
    d = data.make_synthetic(n_days=60)
    d["Visibility"] = 40.0
    return d.asfreq("h")


@pytest.fixture(scope="module")
def parts(df):
    return protocol.make_partitions(df, n_test=4, n_val=3)


# --------------------------------------------------------------------------
# SARIMAX selection
# --------------------------------------------------------------------------
def _grid(rows):
    return pd.DataFrame(rows, columns=["p", "d", "q", "status", "aic"])


def test_stage1_never_compares_aic_across_d():
    from src import sarimax_selection as S
    g = _grid([(1, 0, 1, "converged", 100.0),   # best d=0
               (2, 0, 1, "converged", 120.0),
               (1, 1, 1, "converged", 900.0),   # d=1 has much higher AIC...
               (0, 1, 1, "converged", 950.0),
               (3, 0, 3, "not_converged", 50.0),  # lowest AIC but not converged
               (2, 2, 2, "failed", np.nan)])
    cand = S.stage1_candidates(g, top_k=1)
    # ...but the best d=1 model is still a candidate: AIC ranks only within d
    assert set(map(tuple, cand[["p", "d", "q"]].to_numpy())) == {(1, 0, 1), (1, 1, 1)}
    assert (cand["status"] == "converged").all()


def test_choose_uses_validation_and_prefers_converged():
    from src import sarimax_selection as S
    s2 = pd.DataFrame({"p": [1, 2, 3], "d": [0, 1, 2], "q": [1, 1, 1],
                       "stage1_aic": [1.0, 2.0, 3.0],
                       "val_RMSE": [300.0, 250.0, 200.0],
                       "fit_converged": [True, True, False]})
    best = S.choose(s2)
    assert (best.p, best.d) == (2, 1)        # lowest RMSE among converged


@pytest.mark.skipif(not HAS_SM, reason="statsmodels not installed")
def test_fit_with_diagnostics_reports_nonconvergence_despite_global_ignore():
    from src import sarimax_model  # noqa: F401  (its import sets a global ignore)
    from src.sarimax_selection import fit_with_diagnostics
    y = data.make_synthetic(n_days=10)[config.TARGET].asfreq("h")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _res, info = fit_with_diagnostics(y, (3, 0, 3), (0, 0, 0, 0), maxiter=2)
    assert info["converged"] is False
    assert info["n_warnings"] >= 1 and "Convergence" in info["warnings"]
    _res, info = fit_with_diagnostics(y, (1, 0, 0), (0, 0, 0, 0), maxiter=200)
    assert info["converged"] is True and info["error"] == ""


@pytest.mark.skipif(not HAS_SM, reason="statsmodels not installed")
def test_select_order_never_touches_test(df, parts, monkeypatch):
    from src import sarimax_selection as S
    seen = {"origins": []}
    real_run = backtest.run_backtest

    def spy(d, models, origins, *a, **k):
        seen["origins"] += list(origins)
        return real_run(d, models, origins, *a, **k)
    monkeypatch.setattr(backtest, "run_backtest", spy)
    sel = S.select_order(df, parts, fast=True, top_k=1, progress=False)
    assert set(seen["origins"]) <= set(parts.validation)
    assert not set(seen["origins"]) & set(parts.test)
    assert pd.Timestamp(sel["stage1_window"][1]) <= parts.train_end
    assert sel["test_origins_used"] is False
    assert sel["grid_size"] == 8
    assert {"grid", "candidates", "validation"} <= set(sel["tables"])


@pytest.mark.skipif(not HAS_SM, reason="statsmodels not installed")
def test_sarimax_fit_log_records_convergence(df, parts):
    m = {"sarimax": forecasters.build("sarimax", sarimax_order=(1, 0, 0),
                                      sarimax_seasonal_order=(0, 0, 0, 0))}
    log = backtest.run_backtest(df, m, parts.validation[:1], verbose=False)["fit_log"]
    assert {"converged", "n_warnings", "warnings"} <= set(log.columns)
    assert bool(log["converged"].iloc[0]) is True


# --------------------------------------------------------------------------
# Scenario separation, unique folders, legacy protection
# --------------------------------------------------------------------------
def test_scenarios_written_to_separate_labelled_folders(tmp_path):
    import run_backtest
    for sc in ("operational", "weather_oracle"):
        assert run_backtest.main(["--synthetic", "--synthetic-days", "60",
                                  "--n-test-origins", "2", "--n-val-origins", "2",
                                  "--scenario", sc, "--output-dir", str(tmp_path),
                                  "--run-name", "r"]) == 0
    op, orc = tmp_path / "operational" / "r", tmp_path / "weather_oracle" / "r"
    assert not (op / "SCENARIO_NOTICE.txt").exists()
    assert "NOT A REALISTIC" in (orc / "SCENARIO_NOTICE.txt").read_text()
    for folder, sc, valid in ((op, "operational", True), (orc, "weather_oracle", False)):
        for f in ("forecasts.csv", "window_metrics.csv", "aggregate_metrics.csv",
                  "horizon_metrics.csv"):
            t = pd.read_csv(folder / f)
            assert (t["scenario"] == sc).all() and (t["operationally_valid"] == valid).all()
        cfg = json.loads((folder / "run_config.json").read_text())
        assert cfg["operationally_valid"] is valid
        assert cfg["partition"] == "validation"          # the safe default


def test_default_run_names_are_unique(tmp_path, monkeypatch):
    stamps = iter(["20261024-120000", "20261024-120000"])

    class FakeTs:
        @staticmethod
        def now(tz=None):
            class T:
                def strftime(self, fmt):
                    return next(stamps)
            return T()
    first = backtest.prepare_run_dir(None, "operational", base=tmp_path, prefix="validation")
    assert first.name.startswith("validation_")
    monkeypatch.setattr(backtest.pd, "Timestamp", FakeTs)
    a = backtest.prepare_run_dir(None, "operational", base=tmp_path, prefix="x")
    with pytest.raises(FileExistsError):   # same second -> refused, not reused
        backtest.prepare_run_dir(None, "operational", base=tmp_path, prefix="x")
    assert a.exists()


def test_order_file_scenario_and_synthetic_mismatch_rejected(tmp_path):
    import run_backtest
    f = tmp_path / "sel.json"
    f.write_text(json.dumps({"order": [1, 0, 0], "seasonal_order": [0, 0, 0, 0],
                             "scenario": "weather_oracle", "synthetic": True}))
    with pytest.raises(SystemExit, match="scenario"):
        run_backtest.main(["--synthetic", "--synthetic-days", "60",
                           "--n-test-origins", "2", "--n-val-origins", "2",
                           "--models", "sarimax", "--sarimax-order-file", str(f),
                           "--output-dir", str(tmp_path)])


def test_legacy_outputs_are_guarded(tmp_path, monkeypatch):
    legacy_m, legacy_f = tmp_path / "metrics_legacy", tmp_path / "figures_legacy"
    legacy_m.mkdir(), legacy_f.mkdir()
    (legacy_m / "model_comparison.csv").write_text("historical")
    monkeypatch.setattr(config, "LEGACY_METRIC_DIR", legacy_m)
    monkeypatch.setattr(config, "LEGACY_FIG_DIR", legacy_f)
    monkeypatch.setattr(config, "METRIC_DIR", legacy_m)
    monkeypatch.setattr(config, "FIG_DIR", legacy_f)
    monkeypatch.delenv(config.ALLOW_LEGACY_ENV, raising=False)
    from src import evaluate, foundation
    with pytest.raises(RuntimeError, match="historical"):
        evaluate.comparison_table({"m": {"RMSE": 1.0}})
    with pytest.raises(RuntimeError, match="historical"):
        foundation.run(data.make_synthetic(20), allow_fallback=True)
    assert (legacy_m / "model_comparison.csv").read_text() == "historical"
    with pytest.raises(ValueError):
        backtest.prepare_run_dir("x", "operational", base=legacy_m)
    monkeypatch.setenv(config.ALLOW_LEGACY_ENV, "1")      # explicit opt-in
    evaluate.comparison_table({"m": {"RMSE": 1.0}})


def test_legacy_guard_allows_empty_folders(tmp_path, monkeypatch):
    m, f = tmp_path / "m", tmp_path / "f"
    m.mkdir(), f.mkdir()
    (m / ".gitkeep").write_text("")
    for k, v in (("LEGACY_METRIC_DIR", m), ("LEGACY_FIG_DIR", f),
                 ("METRIC_DIR", m), ("FIG_DIR", f)):
        monkeypatch.setattr(config, k, v)
    monkeypatch.delenv(config.ALLOW_LEGACY_ENV, raising=False)
    config.assert_legacy_write_allowed()


# --------------------------------------------------------------------------
# Chronos input preparation and alignment (stubbed: no weights downloaded)
# --------------------------------------------------------------------------
def test_chronos_adapter_inputs_and_alignment(df, parts, monkeypatch):
    from src import foundation
    captured = {}

    def fake(context, horizon, model_name, num_samples=20):
        captured.update(context=np.asarray(context), horizon=horizon,
                        model=model_name, n=num_samples)
        base = np.arange(horizon, dtype=float)
        return base + 10, base, base + 20, "cpu"
    monkeypatch.setattr(foundation, "_chronos_forecast", fake)
    origin = parts.test[0]
    hist, fut = protocol.history_view(df, origin), protocol.future_view(df, origin)
    p = forecasters.build("chronos").predict(hist, fut)
    ctx = captured["context"]
    assert len(ctx) == len(hist)                                  # nothing after origin
    assert ctx[-1] == df.at[origin, config.TARGET]               # ends AT the origin
    np.testing.assert_array_equal(ctx, df.loc[:origin, config.TARGET].to_numpy(float))
    assert captured["horizon"] == 24 and captured["n"] == 20
    assert captured["model"] == foundation.MODEL_NAME
    assert p.index.equals(protocol.target_index(origin))
    np.testing.assert_array_equal(p["yhat"], np.arange(24) + 10.0)
    assert (p["interval_level"] == 0.80).all()
    assert (p["lower"] <= p["yhat"]).all() and (p["yhat"] <= p["upper"]).all()


# --------------------------------------------------------------------------
# XGBoost recursion diagnostic
# --------------------------------------------------------------------------
def test_predicted_input_share_lags_beyond_horizon_always_observed():
    from src import diagnostics
    s = diagnostics.predicted_input_share()
    assert s.loc[s.step == 1, "lags_from_predictions"].item() == 0
    assert s["lags_from_predictions"].max() == 3          # only lag 1, 2, 3
    assert s.loc[s.step == 24, "roll24_predicted_values"].item() == 23


@pytest.mark.skipif(not HAS_XGB, reason="xgboost not installed")
def test_xgb_recursion_diagnostic_runs_on_validation_only(df, parts):
    from src import diagnostics
    out, detail = diagnostics.xgb_recursion_diagnostic(df, parts, params=FAST_XGB)
    assert set(detail["origin"]) == set(parts.validation)
    assert len(out) == 24
    first = out.loc[out.step == 1]
    # at step 1 both variants use identical (observed) inputs
    assert first["RMSE_recursive"].item() == pytest.approx(first["RMSE_teacher_forced"].item())
