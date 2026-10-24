"""Metric definitions, including zero-target handling."""
import numpy as np
import pytest

from src import metrics


def test_rmse_mae_known_values():
    y, f = [1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 1.0, 4.0]
    assert metrics.mae(y, f) == pytest.approx(0.75)
    assert metrics.rmse(y, f) == pytest.approx(np.sqrt(5 / 4))


def test_mape_excludes_zero_actuals_and_reports_them():
    m, n = metrics.mape([0.0, 100.0, 200.0], [5.0, 110.0, 180.0])
    assert n == 1
    assert m == pytest.approx((10 + 10) / 2)
    m, n = metrics.mape([0.0, 0.0], [1.0, 2.0])
    assert np.isnan(m) and n == 2


def test_smape_zero_handling_and_bounds():
    assert metrics.smape([0.0, 0.0], [0.0, 0.0]) == 0.0      # perfect zero
    assert metrics.smape([0.0], [5.0]) == pytest.approx(200.0)
    assert metrics.smape([100.0], [100.0]) == 0.0
    assert metrics.smape([100.0], [50.0]) == pytest.approx(200 * 50 / 150)


def test_mase_scale():
    hist = np.tile(np.arange(24, dtype=float), 3)   # perfectly daily-periodic
    hist[30] += 24.0                                 # one disturbance
    # seasonal diffs: |+24| at 30 and |-24| at 54 -> mean 48/48 = 1.0
    assert metrics.mase([1.0, 2.0], [2.0, 3.0], hist, season=24) == pytest.approx(1.0)
    assert np.isnan(metrics.mase([1.0], [1.0], np.arange(10.0), season=24))


def test_inputs_validated():
    with pytest.raises(ValueError):
        metrics.rmse([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        metrics.mae([1.0, np.nan], [1.0, 2.0])


def test_score_bundle():
    s = metrics.score([0.0, 10.0], [1.0, 10.0])
    assert set(s) >= {"RMSE", "MAE", "MAPE", "sMAPE", "n_points",
                      "n_zero_actuals_excluded_from_MAPE"}
    assert s["n_zero_actuals_excluded_from_MAPE"] == 1
    assert s["MAPE"] == 0.0
