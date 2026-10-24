"""Time-series foundation model (Amazon Chronos).

Chronos is a pretrained transformer that forecasts a series *zero-shot*: no
task-specific training, just feed it the history. We use a lightweight
checkpoint (``amazon/chronos-t5-small`` by default) that runs on a CPU-only
laptop; the weights download from the Hugging Face hub on first use.

Design decisions:
  * No silent fallback. If Chronos cannot run, this module raises a clear
    error explaining the real reason instead of quietly substituting a naive
    forecast. A fallback is only used if the caller explicitly opts in with
    ``allow_fallback=True``.
  * Deterministic: seeds are set for numpy and torch.
  * Forecast length is exactly ``horizon`` and aligned to the final 24
    timestamps of the series.
  * Appliance energy cannot be negative, so the forecast is clipped at zero.
  * The model name and the device it ran on (cpu/cuda) are recorded.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import config
from .utils import score

# Default lightweight Chronos checkpoint - runs on CPU without a GPU.
MODEL_NAME = "amazon/chronos-t5-small"


def _import_chronos():
    """Import torch + chronos, raising an actionable error if unavailable."""
    try:
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover - depends on environment
        raise ImportError(
            "PyTorch is not installed, so Chronos cannot run. Install the CPU "
            "build with:\n    pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu\n"
            f"(original import error: {e})"
        ) from e
    try:
        from chronos import BaseChronosPipeline
    except Exception as e:
        raise ImportError(
            "The 'chronos-forecasting' package is not installed. Install it "
            "with:\n    pip install chronos-forecasting\n"
            f"(original import error: {e})"
        ) from e
    return torch, BaseChronosPipeline


# Loaded pipelines are cached per (model, device) so a rolling-origin backtest
# loads the weights once instead of once per origin. Forecast behaviour is
# unchanged.
_PIPELINE_CACHE: dict = {}


def _chronos_forecast(context: np.ndarray, horizon: int,
                      model_name: str = MODEL_NAME, num_samples: int = 20):
    """Return (median, low10, high90, device) using Chronos quantiles."""
    torch, BaseChronosPipeline = _import_chronos()

    # deterministic (re-seeded on every call, so each origin is reproducible)
    np.random.seed(config.RANDOM_STATE)
    torch.manual_seed(config.RANDOM_STATE)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    try:
        key = (model_name, device)
        if key not in _PIPELINE_CACHE:
            _PIPELINE_CACHE[key] = BaseChronosPipeline.from_pretrained(
                model_name, device_map=device, torch_dtype=dtype)
        pipe = _PIPELINE_CACHE[key]
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Chronos model '{model_name}'. This usually means "
            "the weights could not be downloaded from the Hugging Face hub "
            "(no internet on first run) or the model name is wrong. "
            f"Original error: {e}"
        ) from e

    ctx = torch.tensor(np.asarray(context, dtype=np.float32))

    # Use the stable sample-based predict() and derive quantiles ourselves.
    # This avoids version differences in predict_quantiles() signatures and
    # works for the default chronos-t5 checkpoint (which returns samples of
    # shape [num_series, num_samples, prediction_length]).
    try:
        samples = pipe.predict(ctx, prediction_length=horizon,
                               num_samples=num_samples)
        s = np.asarray(samples[0].cpu().numpy(), dtype=float)  # (num_samples, horizon)
        median = np.quantile(s, 0.5, axis=0)
        lo = np.quantile(s, 0.1, axis=0)
        hi = np.quantile(s, 0.9, axis=0)
    except Exception:
        # Fallback for checkpoints/versions whose predict() returns quantiles
        # (e.g. Chronos-Bolt): call predict_quantiles positionally.
        quantiles, _mean = pipe.predict_quantiles(
            ctx, horizon, quantile_levels=[0.1, 0.5, 0.9])
        q = np.asarray(quantiles[0].cpu().numpy(), dtype=float)  # (horizon, 3)
        lo, median, hi = q[:, 0], q[:, 1], q[:, 2]

    return np.asarray(median), np.asarray(lo), np.asarray(hi), device


def run(df, horizon=config.HORIZON, model_name=MODEL_NAME,
        allow_fallback=False):
    """Zero-shot 24-hour Chronos forecast on the final ``horizon`` hours.

    Parameters
    ----------
    allow_fallback : bool
        If False (default) any failure raises a clear error. If True, a
        seasonal-naive forecast is substituted and clearly labelled - only for
        convenience during offline development, never for the final result.
    """
    config.assert_legacy_write_allowed("foundation.run")
    y = df[config.TARGET].astype(float)
    train, test = y.iloc[:-horizon], y.iloc[-horizon:]

    try:
        median, lo, hi, device = _chronos_forecast(
            train.values, horizon, model_name)
        name = "Chronos"
        used_fallback = False
    except Exception as e:
        if not allow_fallback:
            # surface the real reason; do NOT silently substitute another model
            raise RuntimeError(
                f"Chronos did not run: {e}\n"
                "Fix the dependency/network issue above, or pass "
                "allow_fallback=True only for offline development."
            ) from e
        from .benchmarks import seasonal_naive_forecast
        print(f"[foundation] WARNING allow_fallback=True: Chronos unavailable "
              f"({e}); substituting seasonal-naive.", file=sys.stderr)
        median = seasonal_naive_forecast(train, horizon).values
        lo, hi = median * 0.6, median * 1.4
        name = "Chronos(fallback=SeasonalNaive)"
        device = "cpu"
        used_fallback = True

    # enforce horizon, alignment, non-negativity
    median = np.clip(np.asarray(median, dtype=float)[:horizon], 0, None)
    lo = np.clip(np.asarray(lo, dtype=float)[:horizon], 0, None)
    hi = np.clip(np.asarray(hi, dtype=float)[:horizon], 0, None)
    assert len(median) == horizon, "Chronos forecast must have length == horizon"
    fc = pd.Series(median, index=test.index)

    info = {
        "model_name": model_name if not used_fallback else name,
        "device": device,
        "used_gpu": device == "cuda",
        "used_fallback": used_fallback,
        "horizon": horizon,
    }
    pd.DataFrame([info]).to_csv(
        config.METRIC_DIR / "chronos_info.csv", index=False)

    return {
        "name": name,
        "forecast": fc,
        "lower": pd.Series(lo, index=test.index),
        "upper": pd.Series(hi, index=test.index),
        "metrics": score(test.values, median),
        "train_tail": train, "test": test,
        "used_fallback": used_fallback,
        "info": info,
    }
