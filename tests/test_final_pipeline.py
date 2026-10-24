"""Statistics, SHAP and the end-to-end final experiment (synthetic, offline)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import stats_tests

ROOT = Path(__file__).resolve().parents[1]
HAS_ML = all(importlib.util.find_spec(m) is not None for m in ("xgboost", "statsmodels"))


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def test_holm_matches_hand_calculation():
    adj = stats_tests.holm([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06])


def test_dm_identical_losses_and_clear_difference():
    rng = np.random.default_rng(0)
    a = rng.gamma(2, 100, 30)
    same = stats_tests.diebold_mariano(a, a + 1e-12)
    assert same["dm_p"] > 0.9
    better = stats_tests.diebold_mariano(a * 0.5, a)
    assert better["dm_stat"] < 0 and better["dm_p"] < 0.01
    assert better["nw_lag"] == 3                    # floor(30 ** (1/3))


def test_block_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    a, b = rng.gamma(2, 100, 30), rng.gamma(2, 120, 30)
    r = stats_tests.block_bootstrap_rmse_diff(a, b, n_boot=2000)
    assert r["ci95_low"] <= r["rmse_diff"] <= r["ci95_high"]
    assert r["rmse_diff"] == pytest.approx(np.sqrt(a.mean()) - np.sqrt(b.mean()))


def _toy_forecasts():
    rng = np.random.default_rng(2)
    origins = pd.date_range("2016-04-01 23:00", periods=30, freq="24h")
    rows = []
    for o in origins:
        y = rng.gamma(3, 100, 24)
        for m, noise in (("good", 30), ("bad", 200)):
            yhat = y + rng.normal(0, noise, 24)
            for s in range(24):
                rows.append({"model": m, "origin": o, "step": s + 1, "y_true": y[s],
                             "yhat": yhat[s], "lower": yhat[s] - 100,
                             "upper": yhat[s] + 100, "interval_level": 0.8})
    return pd.DataFrame(rows)


def test_pairwise_orientation_and_evidence():
    p = stats_tests.pairwise_comparisons(_toy_forecasts(), ["bad", "good"])
    r = p.iloc[0]
    assert r.model_A == "good" and r.rmse_A < r.rmse_B and r.rmse_diff < 0
    assert r.ci_excludes_zero and r.evidence.startswith("A better")


def test_interval_coverage():
    cov = stats_tests.interval_coverage(_toy_forecasts()).set_index("model")
    assert cov.loc["good", "coverage"] > cov.loc["bad", "coverage"]
    assert cov.loc["good", "nominal"] == 0.8


# --------------------------------------------------------------------------
# SHAP
# --------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_ML, reason="xgboost/statsmodels not installed")
def test_tree_shap_additivity_and_direct_explanation():
    from src import data, explain, forecasters, protocol
    df = data.make_synthetic(n_days=40).asfreq("h")
    m = forecasters.build("xgboost_direct", xgb_params={"n_estimators": 15, "max_depth": 3})
    hist = protocol.history_view(df, df.index[-30])
    m.fit(hist)
    out = explain.explain_direct(m, hist, n_sample=50)
    assert out["by_horizon"].shape == (len(m.feature_cols), 24)
    assert (out["overall"] >= 0).all()
    assert all(explain.label(f) != "" for f in out["overall"].index)


# --------------------------------------------------------------------------
# end-to-end orchestrator
# --------------------------------------------------------------------------
def _run(args):
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_final_experiment
    return run_final_experiment.main(args)


def test_smoke_options_refuse_default_output():
    with pytest.raises(SystemExit, match="--out"):
        _run(["--synthetic", "--stages", "analysis"])


@pytest.mark.skipif(not HAS_ML, reason="xgboost/statsmodels not installed")
def test_final_experiment_end_to_end_synthetic(tmp_path):
    out = tmp_path / "final"
    args = ["--synthetic", "--fast-sarimax", "--n-jobs", "1", "--out", str(out),
            "--xgb-params", "n_estimators=10,max_depth=2,n_jobs=1",
            "--stages", "sarimax,validation,test,oracle,analysis"]
    assert _run(args) == 0
    dec = json.loads((out / "validation" / "decisions.json").read_text())
    assert dec["xgboost_strategy"] in ("xgboost", "xgboost_direct")
    test_cfg = json.loads((out / "test" / "run_config.json").read_text())
    val_cfg = json.loads((out / "validation" / "run_config.json").read_text())
    assert not set(test_cfg["origins"]) & set(val_cfg["origins"])
    # only the validation-chosen XGBoost strategy is evaluated on test
    test_models = set(pd.read_csv(out / "test" / "aggregate_metrics.csv")["model"])
    assert dec["xgboost_strategy"] in test_models
    assert ({"xgboost", "xgboost_direct"} - {dec["xgboost_strategy"]}).isdisjoint(test_models)
    assert (out / "test_weather_oracle" / "SCENARIO_NOTICE.txt").exists()
    for fig in ("01_model_comparison", "02_actual_vs_predicted", "03_error_by_horizon",
                "04_error_distribution", "05_shap_importance", "07_forecast_examples",
                "08_paired_differences"):
        assert (out / "figures" / f"{fig}.png").stat().st_size > 10_000, fig
    assert "Paired comparison" in (out / "RESULTS.md").read_text(encoding="utf-8")
    # experiment stages are never overwritten; re-running them is a no-op
    before = (out / "test" / "forecasts.csv").read_bytes()
    assert _run(args) == 0
    assert (out / "test" / "forecasts.csv").read_bytes() == before
