"""Final leakage-safe experiment: selection on validation, one evaluation on test.

    python scripts/run_final_experiment.py                      # every stage
    python scripts/run_final_experiment.py --stages chronos,analysis
    python scripts/run_final_experiment.py --stages analysis    # rebuild figures/tables

Stages (results in outputs/final/<stage>/):

  sarimax     two-stage SARIMAX order selection (training + validation only)
  validation  benchmarks, SARIMAX, recursive and direct XGBoost on the 14
              validation origins -> chooses the XGBoost strategy and the
              reference benchmark (decisions.json). Test data is not read.
  test        benchmarks, SARIMAX and the CHOSEN XGBoost strategy on the 30
              untouched test origins, models fitted once on train+validation
  oracle      chosen XGBoost with observed future weather (upper bound only)
  chronos     zero-shot Chronos-T5-small on the same 30 test origins
  analysis    metrics, paired statistics, interval coverage, SHAP, figures,
              RESULTS.md and the results block in README.md

Experiment stages never overwrite: if their folder exists they are skipped.
Only `analysis` (derived tables and figures) is regenerated on every run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import backtest, config, data, forecasters, protocol  # noqa: E402

OUT = ROOT / "outputs" / "final"
ALL_STAGES = ["sarimax", "validation", "test", "oracle", "chronos", "analysis"]
XGB_PARAMS = None  # only overridden for smoke tests


def log(msg):
    print(f"[final] {msg}", flush=True)


def _run_and_save(df, parts, models, origins, scenario, folder: Path, partition, extra=None):
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.mkdir(exist_ok=False)
    t0 = time.perf_counter()
    res = backtest.run_backtest(df, models, origins, config.HORIZON, scenario, 0)
    windows = backtest.window_metrics(res["forecasts"], df)
    agg = backtest.aggregate_metrics(res["forecasts"], windows)
    hor = backtest.horizon_metrics(res["forecasts"])
    cfg = {"partition": partition, "scenario": scenario,
           "operationally_valid": scenario == "operational",
           "retraining_policy": "fixed: each trainable model fitted once on all "
                                "data up to the first origin of the partition",
           "horizon": config.HORIZON, "origins": [str(o) for o in origins],
           "partitions": parts.summary(),
           "models": {n: m.describe() for n, m in models.items()},
           "runtime_s": round(time.perf_counter() - t0, 1),
           "meta": backtest.run_metadata(), **(extra or {})}
    backtest.save_results(folder, res, windows, agg, hor, cfg)
    log(f"{folder.name}: {len(models)} models x {len(origins)} origins "
        f"in {cfg['runtime_s']} s")
    return agg


def stage_sarimax(df, parts, args):
    folder = OUT / "sarimax_selection"
    if folder.exists():
        return log("sarimax_selection exists - skipped (never overwritten)")
    from src import sarimax_selection
    t0 = time.perf_counter()
    sel = sarimax_selection.select_order(df, parts, "operational",
                                         fast=args.fast_sarimax, top_k=args.top_k,
                                         maxiter=200, n_jobs=args.n_jobs)
    tables = sel.pop("tables")
    folder.mkdir(parents=True)
    tables["grid"].to_csv(folder / "stage1_grid.csv", index=False)
    tables["candidates"].to_csv(folder / "stage1_candidates.csv", index=False)
    tables["validation"].to_csv(folder / "stage2_validation.csv", index=False)
    sel["runtime_s"] = round(time.perf_counter() - t0, 1)
    sel["meta"] = backtest.run_metadata()
    (folder / "selected_order.json").write_text(json.dumps(sel, indent=2, default=str))
    log(f"SARIMAX order {sel['order']}x{sel['seasonal_order']} "
        f"(stage-1 status {sel['stage1_status_counts']}) in {sel['runtime_s']} s")


def _sarimax(scenario="operational"):
    sel = json.loads((OUT / "sarimax_selection" / "selected_order.json").read_text())
    return forecasters.build("sarimax", scenario, sarimax_order=tuple(sel["order"]),
                             sarimax_seasonal_order=tuple(sel["seasonal_order"]),
                             sarimax_order_source="outputs/final/sarimax_selection "
                                                  "(training + validation only)")


def stage_validation(df, parts, args):
    folder = OUT / "validation"
    if folder.exists():
        return log("validation exists - skipped")
    models = {n: forecasters.build(n) for n in forecasters.BENCHMARK_NAMES}
    models["sarimax"] = _sarimax()
    models["xgboost"] = forecasters.build("xgboost", xgb_params=XGB_PARAMS)
    models["xgboost_direct"] = forecasters.build("xgboost_direct", xgb_params=XGB_PARAMS)
    agg = _run_and_save(df, parts, models, parts.validation, "operational",
                        folder, "validation").set_index("model")
    xgb = agg.loc[["xgboost", "xgboost_direct"], "pooled_RMSE"]
    bench = agg.loc[forecasters.BENCHMARK_NAMES, "pooled_RMSE"]
    decisions = {
        "xgboost_strategy": xgb.idxmin(),
        "xgboost_validation_RMSE": xgb.to_dict(),
        "reference_benchmark": bench.idxmin(),
        "benchmark_validation_RMSE": bench.to_dict(),
        "rule": "lowest pooled RMSE on the 14 validation origins; test unseen",
    }
    (folder / "decisions.json").write_text(json.dumps(decisions, indent=2))
    log(f"chosen XGBoost strategy: {decisions['xgboost_strategy']} {xgb.round(1).to_dict()}; "
        f"reference benchmark: {decisions['reference_benchmark']}")


def _decisions():
    return json.loads((OUT / "validation" / "decisions.json").read_text())


def stage_test(df, parts, args):
    folder = OUT / "test"
    if folder.exists():
        return log("test exists - skipped")
    d = _decisions()
    models = {n: forecasters.build(n) for n in forecasters.BENCHMARK_NAMES}
    models["sarimax"] = _sarimax()
    models[d["xgboost_strategy"]] = forecasters.build(d["xgboost_strategy"],
                                                      xgb_params=XGB_PARAMS)
    _run_and_save(df, parts, models, parts.test, "operational", folder, "test",
                  {"decisions_from_validation": d})


def stage_oracle(df, parts, args):
    folder = OUT / "test_weather_oracle"
    if folder.exists():
        return log("test_weather_oracle exists - skipped")
    d = _decisions()
    models = {d["xgboost_strategy"]: forecasters.build(d["xgboost_strategy"], "weather_oracle",
                                                       xgb_params=XGB_PARAMS)}
    _run_and_save(df, parts, models, parts.test, "weather_oracle", folder, "test",
                  {"decisions_from_validation": d})


def stage_chronos(df, parts, args):
    folder = OUT / "test_chronos"
    if folder.exists():
        return log("test_chronos exists - skipped")
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("chronos") is None:
        log("Chronos skipped: torch / chronos-forecasting not installed "
            "(python -m pip install -r requirements-chronos.txt)")
        return
    models = {"chronos": forecasters.build("chronos")}
    try:
        _run_and_save(df, parts, models, parts.test, "operational", folder, "test")
    except Exception as e:
        if folder.exists() and not any(folder.iterdir()):
            folder.rmdir()
        log(f"Chronos failed: {type(e).__name__}: {e}")


def stage_analysis(df, parts, args):
    from src import final_analysis
    # the README results block is only rewritten for the real experiment
    readme_root = ROOT if OUT.resolve() == (ROOT / "outputs" / "final").resolve() else None
    final_analysis.build(df, parts, OUT, readme_root, xgb_params=XGB_PARAMS)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default=",".join(ALL_STAGES))
    ap.add_argument("--n-jobs", type=int, default=2,
                    help="processes for the SARIMAX grid (keep low on 8-16 GB RAM)")
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--out", default=None, help="results folder (default outputs/final)")
    ap.add_argument("--synthetic", action="store_true", help="smoke test only")
    ap.add_argument("--fast-sarimax", action="store_true", help="smoke test only")
    ap.add_argument("--xgb-params", default=None, help="smoke test only: key=value,...")
    a = ap.parse_args(argv)
    global OUT, XGB_PARAMS
    if a.out:
        OUT = Path(a.out)
    elif a.synthetic or a.fast_sarimax or a.xgb_params:
        raise SystemExit("smoke-test options require --out (never outputs/final)")
    if a.xgb_params:
        from run_backtest import parse_params
        XGB_PARAMS = parse_params(a.xgb_params)
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    bad = set(stages) - set(ALL_STAGES)
    if bad:
        raise SystemExit(f"unknown stages {bad}; choose from {ALL_STAGES}")
    df, quality = data.load_checked(synthetic=a.synthetic, n_synthetic_days=75)
    parts = protocol.make_partitions(df)
    log(f"data: {quality['hourly_bins_after']} complete hours; "
        f"validation {len(parts.validation)} / test {len(parts.test)} origins")
    for s in ALL_STAGES:
        if s in stages:
            log(f"--- stage {s} ---")
            globals()[f"stage_{s}"](df, parts, a)

    return 0


if __name__ == "__main__":
    sys.exit(main())
