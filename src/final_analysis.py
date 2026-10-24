"""Tables, statistics, SHAP, figures and README results for outputs/final.

Called by ``scripts/run_final_experiment.py --stages analysis``. Everything
here is derived from saved forecasts, so it can be regenerated at any time
(e.g. after adding the Chronos run) without re-running any model.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import backtest, config, explain, forecasters, protocol, stats_tests  # noqa: E402

# ---- visual system (validated categorical slots 1-3; benchmarks recede) ----
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e6e5e0"
ACTUAL = "#2b2b29"
COLORS = {"xgboost_direct": "#2a78d6", "xgboost": "#2a78d6",
          "sarimax": "#eb6834", "chronos": "#1baf7a"}
REF_GRAY, BENCH_GRAY = "#8a8983", "#c9c8c1"
NAMES = {"xgboost_direct": "XGBoost (direct)", "xgboost": "XGBoost (recursive)",
         "sarimax": "SARIMAX", "chronos": "Chronos-T5 (zero-shot)",
         "mean": "Mean", "naive": "Naive", "drift": "Drift",
         "seasonal_naive_24": "Seasonal naive (24 h)",
         "seasonal_naive_168": "Seasonal naive (168 h)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK2, "axes.edgecolor": "#d5d4cf", "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "legend.fontsize": 9, "lines.solid_capstyle": "round",
})


def _p(p):
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def name(m):
    return NAMES.get(m, m)


def color(m, reference):
    if m in COLORS:
        return COLORS[m]
    return REF_GRAY if m == reference else BENCH_GRAY


def _title(ax, title, subtitle=None):
    ax.set_title(title, pad=22 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9,
                color=INK2, va="bottom")


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def load_forecasts(out: Path):
    parts = [pd.read_csv(out / "test" / "forecasts.csv", parse_dates=["origin", "target_time"])]
    chronos_dir = out / "test_chronos"
    has_chronos = (chronos_dir / "forecasts.csv").exists()
    if has_chronos:
        parts.append(pd.read_csv(chronos_dir / "forecasts.csv",
                                 parse_dates=["origin", "target_time"]))
    f = pd.concat(parts, ignore_index=True)
    assert (f["scenario"] == "operational").all(), "oracle results must not be mixed in"
    backtest.check_identical_targets(f)
    return f.drop(columns=["scenario", "operationally_valid"]), has_chronos


def build(df, parts, out: Path, root: Path | None, xgb_params=None):
    out = Path(out)
    ana, figs = out / "analysis", out / "figures"
    ana.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    decisions = json.loads((out / "validation" / "decisions.json").read_text())
    sel = json.loads((out / "sarimax_selection" / "selected_order.json").read_text())
    ref = decisions["reference_benchmark"]
    xgb = decisions["xgboost_strategy"]

    f, has_chronos = load_forecasts(out)
    windows = backtest.window_metrics(f, df)
    agg = backtest.aggregate_metrics(f, windows).set_index("model")
    hor = backtest.horizon_metrics(f)
    ref_rmse = agg.loc[ref, "pooled_RMSE"]
    agg["skill_vs_reference_pct"] = 100 * (1 - agg["pooled_RMSE"] / ref_rmse)
    contenders = [m for m in (xgb, "sarimax", "chronos") if m in agg.index]
    pair = stats_tests.pairwise_comparisons(f, contenders + [ref])
    cov = stats_tests.interval_coverage(f)

    val = pd.read_csv(out / "validation" / "aggregate_metrics.csv").set_index("model")
    oracle = None
    if (out / "test_weather_oracle" / "aggregate_metrics.csv").exists():
        oracle = pd.read_csv(out / "test_weather_oracle" / "aggregate_metrics.csv")

    # ---- SHAP for the final operational model (refit = the test model) ----
    model = forecasters.build(xgb, xgb_params=xgb_params)
    hist = protocol.history_view(df, parts.test[0])
    model.fit(hist)
    shap = (explain.explain_direct(model, hist) if xgb == "xgboost_direct"
            else explain.explain_recursive(model, hist))

    # ---- tables -----------------------------------------------------------
    agg.reset_index().to_csv(ana / "test_aggregate_metrics.csv", index=False)
    windows.to_csv(ana / "test_window_metrics.csv", index=False)
    hor.to_csv(ana / "test_horizon_metrics.csv", index=False)
    pair.to_csv(ana / "paired_comparisons.csv", index=False)
    cov.to_csv(ana / "interval_coverage.csv", index=False)
    shap["overall"].rename("mean_abs_shap").to_csv(ana / "shap_importance.csv")
    if shap["by_horizon"] is not None:
        shap["by_horizon"].to_csv(ana / "shap_importance_by_horizon.csv")

    best = agg["pooled_RMSE"].idxmin()
    summary = {
        "test_period": [str(parts.test[0] + pd.Timedelta(hours=1)),
                        str(parts.test[-1] + pd.Timedelta(hours=24))],
        "n_test_windows": int(len(parts.test)), "horizon_h": config.HORIZON,
        "reference_benchmark": ref, "xgboost_strategy": xgb,
        "xgboost_validation_RMSE": decisions["xgboost_validation_RMSE"],
        "sarimax_order": [sel["order"], sel["seasonal_order"]],
        "sarimax_selection_val_RMSE": sel["selection_val_RMSE"],
        "best_model_by_pooled_test_RMSE": best, "chronos_included": has_chronos,
        "test_metrics": agg[["pooled_RMSE", "pooled_MAE", "pooled_sMAPE",
                             "window_MASE_mean", "window_RMSE_median",
                             "skill_vs_reference_pct"]].round(3).to_dict(orient="index"),
        "paired": pair.round(4).to_dict(orient="records"),
        "coverage": cov.round(4).to_dict(orient="records"),
        "shap_top10": shap["overall"].head(10).round(3).to_dict(),
    }
    (ana / "results.json").write_text(json.dumps(summary, indent=2, default=str))

    # ---- figures ----------------------------------------------------------
    order = agg.sort_values("pooled_RMSE").index.tolist()
    fig_model_comparison(agg, windows, order, ref, figs / "01_model_comparison.png")
    fig_actual_vs_pred(f, df, parts, best, contenders, ref, figs / "02_actual_vs_predicted.png")
    fig_horizon(hor, contenders, ref, figs / "03_error_by_horizon.png")
    fig_error_distribution(f, windows, contenders, ref, best, figs / "04_error_distribution.png")
    fig_shap(shap, figs / "05_shap_importance.png", figs / "06_shap_by_horizon.png")
    fig_examples(f, windows, best, contenders, ref, figs / "07_forecast_examples.png")
    fig_paired(pair, figs / "08_paired_differences.png")

    md = results_markdown(agg, val, pair, cov, decisions, sel, ref, xgb, best,
                          has_chronos, oracle, shap, parts)
    (out / "RESULTS.md").write_text(md, encoding="utf-8")
    if root is not None:
        update_readme(root / "README.md", md)
    print(f"[analysis] best by pooled test RMSE: {best}; figures -> {figs}")
    return summary


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_model_comparison(agg, windows, order, ref, path):
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(order) + 1.6))
    y = np.arange(len(order))[::-1]
    for yi, m in zip(y, order):
        w = windows.loc[windows.model == m, "RMSE"]
        q1, q3 = w.quantile([0.25, 0.75])
        c = color(m, ref)
        ax.plot([q1, q3], [yi, yi], color=c, lw=7, alpha=0.28, solid_capstyle="round")
        ax.plot(w.median(), yi, marker="|", ms=13, mew=2, color=c)
        ax.plot(agg.loc[m, "pooled_RMSE"], yi, "o", ms=8, color=c, mec=SURFACE, mew=2)
        ax.annotate(f"{agg.loc[m, 'pooled_RMSE']:.0f}", (1.01, yi),
                    xycoords=("axes fraction", "data"), va="center",
                    fontsize=9, color=INK2)
    ax.set_yticks(y, [name(m) for m in order])
    ax.set_xlabel("RMSE (Wh)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(left=0)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=MUTED, ms=7, label="Pooled RMSE, all 720 test hours (value shown)"),
         Line2D([], [], marker="|", ls="", color=MUTED, ms=11, mew=2, label="Median daily RMSE"),
         Line2D([], [], lw=7, color=MUTED, alpha=0.28, label="Middle 50% of daily RMSE")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.14 - 0.3 / len(order)),
              ncol=3, fontsize=8)
    ax.annotate("Pooled\nRMSE", (1.01, 1.0), xycoords="axes fraction", fontsize=8,
                color=INK2, va="bottom")
    _title(ax, "Test accuracy over 30 daily 24-hour forecasts",
           "Lower is better. Operational scenario: no future sensor or weather data.")
    _save(fig, path)


def fig_actual_vs_pred(f, df, parts, best, contenders, ref, path):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.2), gridspec_kw={"hspace": 0.55})
    fb = f[f.model == best].sort_values("target_time")
    a1.plot(fb.target_time, fb.y_true, color=ACTUAL, lw=1.1, label="Actual")
    a1.plot(fb.target_time, fb.yhat, color=color(best, ref), lw=1.4, label=name(best))
    a1.set_ylabel("Energy (Wh per hour)")
    a1.legend(loc="upper left", ncol=2)
    _title(a1, f"Actual vs forecast across the 30-day test period - {name(best)}",
           "Each day is forecast at 23:00 the previous evening, 1-24 hours ahead.")
    wk = parts.test[7:14]
    fw = f[f.origin.isin(wk)]
    a = fw[fw.model == best].sort_values("target_time")
    a2.plot(a.target_time, a.y_true, color=ACTUAL, lw=1.6, label="Actual")
    for m in contenders + [ref]:
        g = fw[fw.model == m].sort_values("target_time")
        a2.plot(g.target_time, g.yhat, color=color(m, ref), lw=1.5 if m != ref else 1.2,
                label=name(m))
    a2.set_ylabel("Energy (Wh per hour)")
    a2.legend(loc="upper left", ncol=len(contenders) + 2, fontsize=8)
    _title(a2, "One test week, all main models",
           f"{wk[0] + pd.Timedelta(hours=1):%d %b} - {wk[-1] + pd.Timedelta(hours=24):%d %b %Y}")
    import matplotlib.dates as mdates
    for ax, step in ((a1, 4), (a2, 1)):
        ax.margins(x=0.01)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=step))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    _save(fig, path)


def fig_horizon(hor, contenders, ref, path):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for m in [ref] + contenders:
        g = hor[hor.model == m]
        ax.plot(g.step, g.RMSE, color=color(m, ref), lw=2, marker="o", ms=4,
                mec=SURFACE, mew=1, label=name(m))
    ax.set_xlabel("Hours ahead (step h forecasts the hour starting at h-1:00)")
    ax.set_ylabel("RMSE (Wh)")
    ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
    ax.legend(loc="upper left")
    _title(ax, "Error by forecast horizon (30 test days)",
           "Horizon and time of day coincide: daytime peaks, not distance alone, drive the error.")
    _save(fig, path)


def fig_error_distribution(f, windows, contenders, ref, best, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"wspace": 0.3})
    ms = contenders + [ref]
    data_ = [windows.loc[windows.model == m, "RMSE"].to_numpy() for m in ms]
    bp = a1.boxplot(data_, vert=False, widths=0.45, patch_artist=True, showfliers=False,
                    medianprops={"color": INK, "lw": 1.5})
    rng = np.random.default_rng(0)
    for i, (m, d) in enumerate(zip(ms, data_), 1):
        bp["boxes"][i - 1].set(facecolor=color(m, ref), alpha=0.25, edgecolor=color(m, ref))
        a1.scatter(d, i + rng.uniform(-0.12, 0.12, len(d)), s=16, color=color(m, ref),
                   edgecolor=SURFACE, linewidth=0.8, zorder=3)
    a1.set_yticks(range(1, len(ms) + 1), [name(m) for m in ms])
    a1.set_xlabel("Daily RMSE (Wh) - one dot per test day")
    a1.grid(axis="y", visible=False)
    _title(a1, "Daily error spread")
    errs = np.concatenate([(f.loc[f.model == m, "yhat"] - f.loc[f.model == m, "y_true"]).to_numpy()
                           for m in (best, ref)])
    lo, hi = np.percentile(errs, [0.5, 99.5])
    bins = np.linspace(lo, hi, 50)
    for m in (best, ref):
        e = (f.loc[f.model == m, "yhat"] - f.loc[f.model == m, "y_true"]).to_numpy()
        a2.hist(e, bins=bins, histtype="step", lw=2, color=color(m, ref),
                label=f"{name(m)} (median {np.median(e):+.0f} Wh)")
    a2.axvline(0, color=INK2, lw=0.8)
    a2.set_xlabel("Hourly error, forecast - actual (Wh)")
    a2.set_ylabel("Hours")
    a2.legend(loc="upper left", fontsize=8)
    e_best = (f.loc[f.model == best, "yhat"] - f.loc[f.model == best, "y_true"]).to_numpy()
    share = np.mean(e_best < 0)
    _title(a2, "Hourly error distribution",
           f"{name(best)} under-forecasts {share:.0%} of hours; 1st-99th pct shown")
    _save(fig, path)


def fig_shap(shap, path_bar, path_heat):
    top = shap["overall"].head(12)[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.barh([explain.label(i) for i in top.index], top.values, height=0.6,
            color="#2a78d6")
    for i, v in enumerate(top.values):
        ax.text(v + top.max() * 0.01, i, f"{v:.0f}", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("Mean |SHAP value| (Wh) - average impact on the forecast")
    ax.grid(axis="y", visible=False)
    kind = "averaged over the 24 horizon models" if shap["kind"] == "direct" else ""
    _title(ax, "What drives the XGBoost forecast (TreeSHAP)",
           f"{shap['n_sample']} recent training rows per model {kind}; associations, not causes.")
    _save(fig, path_bar)
    if shap["by_horizon"] is None:
        return
    bh = shap["by_horizon"].loc[shap["overall"].head(10).index]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(bh.values, aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_yticks(range(len(bh)), [explain.label(i) for i in bh.index])
    ax.set_xticks(range(0, 24, 3), [str(h) for h in range(1, 25, 3)])
    ax.set_xlabel("Forecast step (hours ahead)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("Mean |SHAP| (Wh)", color=INK2)
    cb.outline.set_visible(False)
    first, last = bh[1].idxmax(), bh[24].idxmax()
    _title(ax, "Feature influence by forecast horizon",
           f"Top input at step 1: {explain.label(first)}; at step 24: {explain.label(last)}. "
           "Hour of day is split across 3 encodings.")
    _save(fig, path_heat)


def fig_examples(f, windows, best, contenders, ref, path):
    w = windows[windows.model == best].set_index("origin")["RMSE"]
    picks = [("Easiest test day", w.idxmin()), ("Hardest test day", w.idxmax())]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (label_, o) in zip(axes, picks):
        g = f[f.origin == o]
        a = g[g.model == best].sort_values("step")
        ax.plot(a.step, a.y_true, color=ACTUAL, lw=2.2, label="Actual")
        for m in contenders + [ref]:
            s = g[g.model == m].sort_values("step")
            ax.plot(s.step, s.yhat, color=color(m, ref), lw=1.6, label=name(m))
        day = (o + pd.Timedelta(hours=1))
        ax.set_xlabel("Hour of the day (step 1 = 00:00)")
        _title(ax, f"{label_}: {day:%a %d %b %Y}",
               f"{name(best)} daily RMSE {w.loc[o]:.0f} Wh")
        ax.set_xticks([1, 6, 12, 18, 24])
    axes[0].set_ylabel("Energy (Wh per hour)")
    axes[1].legend(loc="upper left", fontsize=8)
    _save(fig, path)


def fig_paired(pair, path):
    p = pair.sort_values("rmse_diff")
    fig, ax = plt.subplots(figsize=(9.5, 0.6 * len(p) + 1.8))
    y = np.arange(len(p))[::-1]
    for yi, (_, r) in zip(y, p.iterrows()):
        c = "#2a78d6" if r.ci_excludes_zero else MUTED
        ax.plot([r.ci95_low, r.ci95_high], [yi, yi], color=c, lw=2)
        ax.plot(r.rmse_diff, yi, "o", ms=8, color=c, mec=SURFACE, mew=2)
        ax.annotate(f"DM p {_p(r.dm_p)} (Holm {_p(r.dm_p_holm)})\nA better on {r.win_rate_A:.0%} of days",
                    (1.02, yi), xycoords=("axes fraction", "data"), va="center",
                    fontsize=8, color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y, [f"{name(r.model_A)} vs {name(r.model_B)}" for _, r in p.iterrows()])
    ax.set_xlabel("Difference in pooled RMSE, A - B (Wh) with 95% block-bootstrap CI")
    ax.grid(axis="y", visible=False)
    _title(ax, "Is the better model reliably better?",
           "30 paired daily windows. Blue: CI excludes zero. Grey: not distinguishable.")
    _save(fig, path)


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------
def results_markdown(agg, val, pair, cov, decisions, sel, ref, xgb, best,
                     has_chronos, oracle, shap, parts):
    t0 = parts.test[0] + pd.Timedelta(hours=1)
    t1 = parts.test[-1] + pd.Timedelta(hours=24)
    lines = [
        f"**Test period:** {t0:%d %b %Y} – {t1:%d %b %Y} · 30 daily forecasts made at "
        f"23:00 for the next 24 hours (720 hourly targets) · operational scenario "
        f"(only past data + calendar at forecast time) · models fitted once on "
        f"train+validation data.",
        "",
        "| Model | RMSE (Wh) | MAE (Wh) | sMAPE (%) | MASE | Median daily RMSE | RMSE vs "
        f"{name(ref)} |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m, r in agg.sort_values("pooled_RMSE").iterrows():
        mk = f"**{name(m)}**" if m == best else name(m)
        lines.append(f"| {mk} | {r.pooled_RMSE:.1f} | {r.pooled_MAE:.1f} | "
                     f"{r.pooled_sMAPE:.1f} | {r.window_MASE_mean:.2f} | "
                     f"{r.window_RMSE_median:.1f} | {r.skill_vs_reference_pct:+.1f}% |")
    if not has_chronos:
        lines += ["", "*Chronos-T5 has not been run on the test period in this build "
                      "yet - see “Reproduce” (`--stages chronos,analysis`).*"]
    lines += ["", "**Decisions made on validation only (14 days, test unseen):**", "",
              f"- XGBoost strategy: **{name(xgb)}** — validation RMSE "
              + ", ".join(f"{name(k)} {v:.1f}" for k, v in decisions["xgboost_validation_RMSE"].items()),
              f"- SARIMAX order: **{tuple(sel['order'])}×{tuple(sel['seasonal_order'])}** — "
              f"AIC compared only within each differencing order, then validation RMSE "
              f"({sel['selection_val_RMSE']:.1f}); stage-1 fits: {sel['stage1_status_counts']}",
              f"- Reference benchmark: **{name(ref)}** (best benchmark on validation)", "",
              "**Paired comparison over the 30 test days** (Diebold–Mariano on daily MSE "
              "with HAC variance, Holm-corrected; 95% moving-block bootstrap CI for the "
              "RMSE difference):", "",
              "| A (lower RMSE) | B | RMSE diff A−B | 95% CI | DM p (Holm) | A wins | Evidence |",
              "|---|---|---:|---|---:|---:|---|"]
    for _, r in pair.iterrows():
        lines.append(f"| {name(r.model_A)} | {name(r.model_B)} | {r.rmse_diff:+.1f} | "
                     f"[{r.ci95_low:+.1f}, {r.ci95_high:+.1f}] | {_p(r.dm_p_holm).replace('= ', '')} | "
                     f"{r.win_rate_A:.0%} | {r.evidence} |")
    if not cov.empty:
        lines += ["", "**Prediction-interval coverage on test:** " + "; ".join(
            f"{name(r.model)} {r.nominal:.0%} interval covered {r.coverage:.1%} of hours "
            f"(mean width {r.mean_width:.0f} Wh)" for _, r in cov.iterrows())]
    if oracle is not None:
        o = oracle.iloc[0]
        lines += ["", f"**Upper bound, not an operational result:** with *observed* future "
                      f"outdoor weather (`weather_oracle`), {name(o.model)} reaches test RMSE "
                      f"{o.pooled_RMSE:.1f} Wh vs {agg.loc[o.model, 'pooled_RMSE']:.1f} Wh "
                      f"operationally ({100 * (o.pooled_RMSE / agg.loc[o.model, 'pooled_RMSE'] - 1):+.1f}%): "
                      f"an upper bound on what perfect weather forecasts could add."]
    top = [explain.label(i) for i in shap["overall"].head(3).index]
    lines += ["", f"**SHAP (model inputs, not causes):** the most influential inputs are "
                  f"{top[0]}, {top[1]} and {top[2]}."]
    return "\n".join(lines) + "\n"


def update_readme(readme: Path, md: str):
    if not readme.exists():
        return
    s = readme.read_text(encoding="utf-8")
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if start not in s or end not in s:
        return
    new = re.sub(re.escape(start) + r".*?" + re.escape(end),
                 lambda _m: f"{start}\n{md}{end}", s, flags=re.S)
    readme.write_text(new, encoding="utf-8")
