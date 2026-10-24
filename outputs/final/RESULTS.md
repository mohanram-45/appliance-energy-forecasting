**Test period:** 27 Apr 2016 – 26 May 2016 · 30 daily forecasts made at 23:00 for the next 24 hours (720 hourly targets) · operational scenario (only past data + calendar at forecast time) · models fitted once on train+validation data.

| Model | RMSE (Wh) | MAE (Wh) | sMAPE (%) | MASE | Median daily RMSE | RMSE vs Mean |
|---|---:|---:|---:|---:|---:|---:|
| **SARIMAX** | 375.2 | 217.7 | 32.8 | 0.67 | 250.5 | +10.4% |
| Chronos-T5 (zero-shot) | 405.8 | 197.7 | 27.8 | 0.61 | 277.3 | +3.1% |
| Mean | 418.7 | 290.7 | 47.0 | 0.90 | 301.8 | +0.0% |
| XGBoost (direct) | 422.2 | 253.8 | 34.2 | 0.79 | 304.4 | -0.8% |
| Seasonal naive (168 h) | 463.3 | 245.8 | 31.7 | 0.76 | 318.7 | -10.7% |
| Seasonal naive (24 h) | 477.1 | 263.0 | 34.9 | 0.82 | 330.0 | -13.9% |
| Naive | 481.1 | 262.7 | 41.1 | 0.81 | 348.3 | -14.9% |
| Drift | 481.2 | 262.8 | 41.1 | 0.81 | 348.3 | -14.9% |

**Decisions made on validation only (14 days, test unseen):**

- XGBoost strategy: **XGBoost (direct)** — validation RMSE XGBoost (recursive) 447.3, XGBoost (direct) 408.8
- SARIMAX order: **(1, 1, 6)×(1, 0, 1, 24)** — AIC compared only within each differencing order, then validation RMSE (400.9); stage-1 fits: {'converged': 133, 'not_converged': 14}
- Reference benchmark: **Mean** (best benchmark on validation)

**Paired comparison over the 30 test days** (Diebold–Mariano on daily MSE with HAC variance, Holm-corrected; 95% moving-block bootstrap CI for the RMSE difference):

| A (lower RMSE) | B | RMSE diff A−B | 95% CI | DM p (Holm) | A wins | Evidence |
|---|---|---:|---|---:|---:|---|
| SARIMAX | XGBoost (direct) | -47.0 | [-149.0, +26.3] | 0.897 | 60% | not distinguishable |
| Chronos-T5 (zero-shot) | XGBoost (direct) | -16.4 | [-128.4, +60.5] | 1.000 | 53% | not distinguishable |
| Mean | XGBoost (direct) | -3.5 | [-103.1, +65.6] | 1.000 | 27% | not distinguishable |
| SARIMAX | Chronos-T5 (zero-shot) | -30.6 | [-45.3, -10.8] | 0.036 | 67% | A better (significant after Holm, CI excludes 0) |
| SARIMAX | Mean | -43.5 | [-55.6, -31.0] | < 0.001 | 97% | A better (significant after Holm, CI excludes 0) |
| Chronos-T5 (zero-shot) | Mean | -12.9 | [-39.8, +7.3] | 0.796 | 67% | not distinguishable |

**Prediction-interval coverage on test:** Chronos-T5 (zero-shot) 80% interval covered 58.1% of hours (mean width 230 Wh); SARIMAX 95% interval covered 95.8% of hours (mean width 1405 Wh)

**Upper bound, not an operational result:** with *observed* future outdoor weather (`weather_oracle`), XGBoost (direct) reaches test RMSE 402.8 Wh vs 422.2 Wh operationally (-4.6%): an upper bound on what perfect weather forecasts could add.

**SHAP (model inputs, not causes):** the most influential inputs are Hour of day, Same hour yesterday and Hour of day (sin).
