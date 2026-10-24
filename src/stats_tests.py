"""Paired forecast-window comparisons with honest uncertainty.

Unit of analysis: the 24-hour forecast window. Hourly errors inside one window
are strongly autocorrelated (they share an origin, a weather regime and a
daily routine), so treating 720 test hours as 720 independent observations
would overstate the evidence roughly by the within-day correlation. Each
model's loss per window is its mean squared error over that day (MSE_i), and
models are compared through the paired differences

    d_i = MSE_A,i - MSE_B,i        i = 1..n windows (n = 30)

Three complementary summaries are reported:

1. Diebold-Mariano test on d_i with a Newey-West (Bartlett) long-run
   variance, lag L = floor(n^(1/3)) = 3 for n = 30, because consecutive days
   can still be correlated. The Harvey-Leybourne-Newbold small-sample
   correction is applied and p-values use Student-t with n-1 df.
2. Moving-block bootstrap (blocks of 3 consecutive windows, 10,000 draws) of
   the difference in pooled RMSE, giving a 95% percentile interval.
3. Wilcoxon signed-rank test on per-window RMSE differences and the win rate
   (share of windows where A beats B). Wilcoxon assumes independent windows,
   so it is a robustness check, not the primary evidence.

All pairwise tests in one family get a Holm correction. With n = 30 windows
from one household in one season the tests have limited power: a
non-significant result means "not distinguishable with this data", not
"equivalent".
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats


def window_losses(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Wide table: rows = origins, columns = models, values = window MSE."""
    f = forecasts.assign(se=(forecasts["yhat"] - forecasts["y_true"]) ** 2)
    return f.pivot_table(index="origin", columns="model", values="se", aggfunc="mean")


def newey_west_var(d: np.ndarray, lag: int) -> float:
    d = np.asarray(d, float) - np.mean(d)
    n = len(d)
    gamma0 = np.dot(d, d) / n
    s = gamma0
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        s += 2 * w * np.dot(d[k:], d[:-k]) / n
    return max(s, 1e-12)


def diebold_mariano(loss_a, loss_b, lag: int | None = None) -> dict:
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    lag = int(math.floor(n ** (1 / 3))) if lag is None else lag
    lrv = newey_west_var(d, lag)
    dm = d.mean() / math.sqrt(lrv / n)
    h = 1
    hln = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return {"dm_stat": float(stat), "dm_p": float(p), "nw_lag": lag,
            "mean_loss_diff": float(d.mean())}


def block_bootstrap_rmse_diff(loss_a, loss_b, block: int = 3, n_boot: int = 10_000,
                              seed: int = 42) -> dict:
    """CI for pooled RMSE_A - pooled RMSE_B (equal-length windows)."""
    a, b = np.asarray(loss_a, float), np.asarray(loss_b, float)
    n = len(a)
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    diffs = np.sqrt(a[idx].mean(1)) - np.sqrt(b[idx].mean(1))
    point = math.sqrt(a.mean()) - math.sqrt(b.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"rmse_diff": float(point), "ci95_low": float(lo), "ci95_high": float(hi),
            "rel_diff_pct": float(100 * point / math.sqrt(b.mean())),
            "block": block, "n_boot": n_boot}


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def pairwise_comparisons(forecasts: pd.DataFrame, models: list[str],
                         alpha: float = 0.05) -> pd.DataFrame:
    """All pairs among ``models``; A is the model with the lower pooled RMSE."""
    L = window_losses(forecasts)[models].dropna()
    rows = []
    for m1, m2 in itertools.combinations(models, 2):
        a, b = (m1, m2) if L[m1].mean() <= L[m2].mean() else (m2, m1)
        ra, rb = np.sqrt(L[a]), np.sqrt(L[b])
        row = {"model_A": a, "model_B": b, "n_windows": len(L),
               "rmse_A": float(np.sqrt(L[a].mean())), "rmse_B": float(np.sqrt(L[b].mean())),
               "win_rate_A": float((ra < rb).mean())}
        row.update(diebold_mariano(L[a], L[b]))
        row.update(block_bootstrap_rmse_diff(L[a], L[b]))
        try:
            row["wilcoxon_p"] = float(stats.wilcoxon(ra - rb).pvalue)
        except ValueError:
            row["wilcoxon_p"] = float("nan")
        rows.append(row)
    out = pd.DataFrame(rows)
    out["dm_p_holm"] = holm(out["dm_p"])
    out["ci_excludes_zero"] = (out["ci95_high"] < 0) | (out["ci95_low"] > 0)
    out["evidence"] = np.where(
        (out["dm_p_holm"] < alpha) & out["ci_excludes_zero"],
        "A better (significant after Holm, CI excludes 0)",
        np.where(out["ci_excludes_zero"] | (out["dm_p"] < alpha),
                 "suggestive only (not robust to correction)",
                 "not distinguishable"))
    return out


def interval_coverage(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Empirical coverage of prediction intervals where a model provides them."""
    f = forecasts.dropna(subset=["lower", "upper"])
    if f.empty:
        return pd.DataFrame()
    g = f.assign(inside=(f.y_true >= f.lower) & (f.y_true <= f.upper),
                 width=f.upper - f.lower)
    return (g.groupby("model")
            .agg(nominal=("interval_level", "first"), coverage=("inside", "mean"),
                 mean_width=("width", "mean"), n=("inside", "size"))
            .reset_index())
