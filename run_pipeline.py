from __future__ import annotations

import argparse
import json
import pandas as pd

from src import config, data, eda, benchmarks, sarimax_model, ml_model
from src import foundation, evaluate
from src.utils import plot_forecast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="smaller SARIMAX grid for a quick run")
    ap.add_argument("--synthetic", action="store_true",
                    help="use synthetic data (no download)")
    ap.add_argument("--no-foundation", action="store_true",
                    help="skip the Chronos foundation model")
    ap.add_argument("--allow-chronos-fallback", action="store_true",
                    help="offline dev only: substitute seasonal-naive if "
                         "Chronos cannot run (NOT for final results)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--overwrite-legacy", action="store_true",
                    help="deliberately replace the committed historical "
                         "results in outputs/metrics and outputs/figures")
    args = ap.parse_args()
    if args.overwrite_legacy:
        import os
        os.environ[config.ALLOW_LEGACY_ENV] = "1"
    config.assert_legacy_write_allowed("run_pipeline.py")

    metrics = {}

    # ---- data + EDA ------------------------------------------------
    print("\n== load, resample, EDA ==")
    df = data.load(synthetic=args.synthetic)
    print(df.shape, "hourly rows;", df.index.min(), "->", df.index.max())
    print("missing values:\n", data.missing_report(df).to_string())
    stat = eda.run(df)
    print("stationarity tests:\n", stat.to_string(index=False))

    y = df[config.TARGET]
    train, test = y.iloc[:-config.HORIZON], y.iloc[-config.HORIZON:]

    # ---- benchmarks ------------------------------------------------
    print("\n== benchmark models ==")
    bench = benchmarks.run_all(train, test)
    forecasts = {}
    for name, (fc, m) in bench.items():
        metrics[name] = m
        forecasts[name] = fc
        print(f"  {name:22s} RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}")
    best_bench = min(bench, key=lambda k: bench[k][1]["RMSE"])
    print("  strongest benchmark:", best_bench)

    # ---- SARIMAX ---------------------------------------------------
    print("\n== SARIMAX (AIC grid search) ==")
    sx = sarimax_model.run(df, fast=args.fast, verbose=args.verbose)
    metrics["SARIMAX"] = sx["metrics"]
    forecasts["SARIMAX"] = sx["forecast"]
    print(f"  best order {sx['order']} x {sx['seasonal_order']}  AIC={sx['aic']:.1f}")
    print(f"  RMSE={sx['metrics']['RMSE']:.2f}  Ljung-Box p={sx['diagnostics']['ljung_box_pvalue']:.3f}")
    plot_forecast(sx["train_tail"], sx["test"], sx["forecast"],
                  "SARIMAX 24h forecast", "10_sarimax_forecast.png",
                  lower=sx["lower"], upper=sx["upper"])

    # ---- features + XGBoost ------------------------------------
    print("\n== features + XGBoost ==")
    xgb = ml_model.run(df)
    metrics["XGBoost"] = xgb["metrics"]
    forecasts["XGBoost"] = xgb["forecast"]
    print(f"  24h RMSE={xgb['metrics']['RMSE']:.2f}  "
          f"14d test RMSE={xgb['metrics_14d']['RMSE']:.2f}")
    print("  feature-group importance:\n", xgb["group_importance"].to_string())
    plot_forecast(xgb["train_tail"], xgb["test"], xgb["forecast"],
                  "XGBoost 24h forecast", "11_xgb_forecast.png")

    # ---- foundation model -----------------------------------------
    if not args.no_foundation:
        print("\n== Chronos foundation model ==")
        fm = foundation.run(df, allow_fallback=args.allow_chronos_fallback)
        metrics[fm["name"]] = fm["metrics"]
        forecasts[fm["name"]] = fm["forecast"]
        print(f"  {fm['name']} RMSE={fm['metrics']['RMSE']:.2f}  "
              f"(model={fm['info']['model_name']}, device={fm['info']['device']})")
        plot_forecast(fm["train_tail"], fm["test"], fm["forecast"],
                      f"{fm['name']} 24h forecast", "12_chronos_forecast.png",
                      lower=fm["lower"], upper=fm["upper"])

    # ---- evaluation ------------------------------------------------
    print("\n== evaluation ==")
    comparison = evaluate.comparison_table(metrics)
    evaluate.all_forecasts_plot(test, forecasts, y)
    evaluate.error_diagnostics(test, forecasts, best_bench)
    comparison = evaluate.skill_scores(comparison, best_bench)
    print(comparison.round(2).to_string())

    (config.METRIC_DIR / "summary.json").write_text(json.dumps(
        {"best_benchmark": best_bench,
         "sarimax_order": [sx["order"], sx["seasonal_order"]],
         "metrics": {k: {m: round(float(v), 3) for m, v in d.items()}
                     for k, d in metrics.items()}}, indent=2))
    print("\nDone. Figures in outputs/figures, tables in outputs/metrics.")


if __name__ == "__main__":
    main()
