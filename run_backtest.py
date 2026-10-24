"""Rolling-origin backtest CLI (Phase 2A protocol, Phase 2B selection).

The legacy single-window study is still run by ``run_pipeline.py``; this
script never writes to ``outputs/metrics`` or ``outputs/figures``. Results go
to ``outputs/backtest/<scenario>/<run-name>/``. Run folders are never reused:
without ``--run-name`` a UTC timestamp makes the name unique, and an existing
folder is an error.

Examples
--------
  # inspect data quality, partitions and origins - no models run
  python run_backtest.py --dry-run

  # SARIMAX order: within-d AIC on training data, then validation RMSE
  python run_backtest.py --select-sarimax-order --scenario operational --n-jobs 4

  # validation run (use for every modelling decision)
  python run_backtest.py --partition validation --models benchmarks,xgboost,sarimax \
      --sarimax-order-file outputs/backtest/operational/<selection-run>/selected_order.json

  # test run - only once all choices are frozen
  python run_backtest.py --partition test --models benchmarks,xgboost,sarimax,chronos \
      --sarimax-order-file outputs/backtest/operational/<selection-run>/selected_order.json
"""
from __future__ import annotations

import argparse
import json
import sys

from src import backtest, config, data, forecasters, protocol


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="benchmarks",
                    help="comma list; 'benchmarks' expands to "
                         f"{forecasters.BENCHMARK_NAMES}. Others: xgboost, "
                         "sarimax, chronos")
    ap.add_argument("--partition", choices=["validation", "test"], default="validation",
                    help="default is validation; test must be asked for explicitly")
    ap.add_argument("--scenario", choices=sorted(protocol.SCENARIOS),
                    default="operational")
    ap.add_argument("--n-test-origins", type=int, default=config.N_TEST_ORIGINS)
    ap.add_argument("--n-val-origins", type=int, default=config.N_VAL_ORIGINS)
    ap.add_argument("--origin-hour", type=int, default=config.ORIGIN_HOUR)
    ap.add_argument("--refit-every", type=int, default=0,
                    help="0 = fit once at the first origin (fixed); k = refit "
                         "every k origins on an expanding window")
    ap.add_argument("--sarimax-order", default=None,
                    help="p,d,q (manual; recorded as not selected by the protocol)")
    ap.add_argument("--sarimax-seasonal-order", default="1,0,1,24")
    ap.add_argument("--sarimax-order-file", default=None,
                    help="selected_order.json from --select-sarimax-order")
    ap.add_argument("--select-sarimax-order", action="store_true",
                    help="two-stage selection on training + validation data, "
                         "save it, and exit")
    ap.add_argument("--sarimax-fast-grid", action="store_true",
                    help="tiny grid for smoke tests only")
    ap.add_argument("--sarimax-top-k", type=int, default=2,
                    help="stage-1 candidates kept per differencing order d")
    ap.add_argument("--sarimax-maxiter", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="parallel processes for the stage-1 SARIMAX grid")
    ap.add_argument("--xgb-params", default=None,
                    help="override XGBoost params as key=value pairs, e.g. "
                         "max_depth=4,n_estimators=400 (shell-quoting safe)")
    ap.add_argument("--chronos-num-samples", type=int, default=20)
    ap.add_argument("--synthetic", action="store_true",
                    help="synthetic data (offline smoke runs only)")
    ap.add_argument("--synthetic-days", type=int, default=137)
    ap.add_argument("--run-name", default=None,
                    help="folder name under outputs/backtest/<scenario>/ "
                         "(default: <partition>_<UTC timestamp>); must not exist")
    ap.add_argument("--output-dir", default=None,
                    help=f"base folder (default {config.BACKTEST_DIR})")
    ap.add_argument("--dry-run", action="store_true",
                    help="print data quality and partitions, run no models")
    return ap.parse_args(argv)


def _ints(s):
    return tuple(int(v) for v in s.split(","))


def parse_params(s):
    """'max_depth=4,learning_rate=0.05' -> {'max_depth': 4, 'learning_rate': 0.05}"""
    if not s:
        return None
    out = {}
    for pair in s.split(","):
        k, sep, v = pair.partition("=")
        if not k.strip() or not sep:
            raise ValueError(f"bad --xgb-params entry {pair!r}; use key=value")
        try:
            out[k.strip()] = json.loads(v.strip())
        except json.JSONDecodeError:
            out[k.strip()] = v.strip()
    return out


def _oracle_banner(scenario):
    if scenario != "operational":
        print("\n" + "!" * 72 + "\n" + backtest.ORACLE_NOTICE + "!" * 72 + "\n",
              file=sys.stderr)


def main(argv=None):
    a = parse_args(argv)
    df, quality = data.load_checked(synthetic=a.synthetic,
                                    n_synthetic_days=a.synthetic_days)
    parts = protocol.make_partitions(df, n_test=a.n_test_origins,
                                     n_val=a.n_val_origins,
                                     origin_hour=a.origin_hour)
    print("== data quality ==")
    print(json.dumps(quality, indent=2, default=str))
    print("== partitions ==")
    print(json.dumps(parts.summary(), indent=2))
    if a.dry_run:
        return 0
    _oracle_banner(a.scenario)

    # ---- SARIMAX order selection (training + validation only) ------------
    if a.select_sarimax_order:
        from src import sarimax_selection
        out = backtest.prepare_run_dir(a.run_name, a.scenario, a.output_dir,
                                       prefix="sarimax_selection")
        sel = sarimax_selection.select_order(
            df, parts, a.scenario, fast=a.sarimax_fast_grid,
            top_k=a.sarimax_top_k, maxiter=a.sarimax_maxiter, n_jobs=a.n_jobs)
        tables = sel.pop("tables")
        tables["grid"].to_csv(out / "stage1_grid.csv", index=False)
        tables["candidates"].to_csv(out / "stage1_candidates.csv", index=False)
        tables["validation"].to_csv(out / "stage2_validation.csv", index=False)
        sel["meta"] = backtest.run_metadata()
        sel["synthetic"] = a.synthetic
        (out / "selected_order.json").write_text(json.dumps(sel, indent=2, default=str))
        print(f"stage-1 fits: {sel['stage1_status_counts']}")
        print(tables["validation"].round(2).to_string(index=False))
        print(f"selected order {sel['order']} x {sel['seasonal_order']} "
              f"(validation RMSE {sel['selection_val_RMSE']:.2f}"
              f"{', ON GRID BOUNDARY' if sel['on_grid_boundary'] else ''}) -> {out}")
        return 0

    # ---- build models -----------------------------------------------------
    names = []
    for n in a.models.split(","):
        n = n.strip()
        names += forecasters.BENCHMARK_NAMES if n == "benchmarks" else [n]
    kw = {"chronos_num_samples": a.chronos_num_samples,
          "xgb_params": parse_params(a.xgb_params),
          "sarimax_seasonal_order": _ints(a.sarimax_seasonal_order),
          "sarimax_maxiter": a.sarimax_maxiter}
    if a.sarimax_order_file:
        sel = json.loads(open(a.sarimax_order_file, encoding="utf-8").read())
        if sel.get("synthetic") and not a.synthetic:
            raise SystemExit("order file was selected on SYNTHETIC data")
        if sel.get("scenario") != a.scenario:
            raise SystemExit(f"order file was selected for scenario "
                             f"{sel.get('scenario')!r}, not {a.scenario!r}")
        kw["sarimax_order"] = tuple(sel["order"])
        kw["sarimax_seasonal_order"] = tuple(sel["seasonal_order"])
        kw["sarimax_order_source"] = f"{a.sarimax_order_file} ({sel.get('selected_on')})"
    elif a.sarimax_order:
        kw["sarimax_order"] = _ints(a.sarimax_order)
        kw["sarimax_order_source"] = "command line (not selected by this protocol)"
    models = {n: forecasters.build(n, a.scenario, **kw) for n in names}

    origins = parts.origins(a.partition)
    out = backtest.prepare_run_dir(a.run_name, a.scenario, a.output_dir,
                                   prefix=a.partition)

    result = backtest.run_backtest(df, models, origins, config.HORIZON,
                                   a.scenario, a.refit_every)
    windows = backtest.window_metrics(result["forecasts"], df)
    agg = backtest.aggregate_metrics(result["forecasts"], windows)
    hor = backtest.horizon_metrics(result["forecasts"])
    run_config = {
        "partition": a.partition, "scenario": a.scenario,
        "operationally_valid": a.scenario == "operational",
        "scenario_description": protocol.SCENARIOS[a.scenario]["description"],
        "horizon": config.HORIZON, "origin_hour": a.origin_hour,
        "refit_every": a.refit_every,
        "retraining_policy": "fixed after first fit" if a.refit_every == 0
        else f"refit every {a.refit_every} origin(s), expanding window",
        "origins": [str(o) for o in origins],
        "partitions": parts.summary(), "data_quality": quality,
        "synthetic": a.synthetic,
        "models": {n: m.describe() for n, m in models.items()},
        "meta": backtest.run_metadata(),
    }
    backtest.save_results(out, result, windows, agg, hor, run_config)
    fl = result["fit_log"]
    if not fl.empty and "converged" in fl and (fl["converged"] == False).any():  # noqa: E712
        print("WARNING: some fits did not converge - see fit_log.csv", file=sys.stderr)
    cols = ["model", "n_windows", "pooled_RMSE", "window_RMSE_mean",
            "window_RMSE_median", "window_RMSE_q25", "window_RMSE_q75",
            "pooled_MAE", "pooled_sMAPE", "window_MASE_mean"]
    print(f"[{a.partition} | scenario={a.scenario}"
          f"{' | UPPER BOUND, NOT OPERATIONAL' if a.scenario != 'operational' else ''}]")
    print(agg[[c for c in cols if c in agg]].round(2).to_string(index=False))
    print(f"results -> {out}")
    _oracle_banner(a.scenario)
    return 0


if __name__ == "__main__":
    sys.exit(main())
