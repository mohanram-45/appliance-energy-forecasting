"""Test isolation: no test may write into the real outputs/ folders.

Before Phase 2A, ``test_chronos_fallback_is_labelled_when_opted_in`` called
``foundation.run`` which writes ``outputs/metrics/chronos_info.csv``; on a
machine without torch that silently replaced the committed result of the
genuine Chronos run with a fallback record. Every test now writes to a
temporary directory instead.
"""
import pytest

from src import config


@pytest.fixture(autouse=True)
def _isolate_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "METRIC_DIR", tmp_path / "metrics")
    monkeypatch.setattr(config, "FIG_DIR", tmp_path / "figures")
    monkeypatch.setattr(config, "BACKTEST_DIR", tmp_path / "backtest")
    for d in ("metrics", "figures", "backtest"):
        (tmp_path / d).mkdir()
    yield
