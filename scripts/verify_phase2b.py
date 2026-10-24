"""Phase 2B local verification - run in YOUR Python environment.

    cd "<path to the project folder>"
    python scripts\\verify_phase2b.py            # ~2-5 minutes
    python scripts\\verify_phase2b.py --skip-timing

What it does (read-only for the project, except its own log folder):
  1. records Python, packages, git state and hardware;
  2. runs the full pytest suite;
  3. runs ``run_backtest.py --dry-run``;
  4. checks partitions, target windows, identical evaluation targets and
     leakage on the REAL data, using validation origins for anything that
     produces forecasts (no test-period metrics are computed);
  5. times a few fits to estimate runtimes of the full experiment.

It never installs packages, never downloads model weights (Chronos is
timed only if the weights are already cached, with the Hugging Face hub
forced offline) and writes only to outputs/verification/phase2b_<UTC time>/.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PACKAGES = ["pandas", "numpy", "scipy", "statsmodels", "scikit-learn", "xgboost",
            "matplotlib", "requests", "pytest", "torch", "chronos-forecasting",
            "transformers", "accelerate"]
MODULES = {"scikit-learn": "sklearn", "chronos-forecasting": "chronos"}
ALT_DISTS = {"xgboost": ("xgboost-cpu",), "torch": ("torch-cpu",)}
RESULTS: dict = {"checks": {}}


def log(msg):
    print(msg, flush=True)


def run(cmd, out_file, timeout=1800):
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                           timeout=timeout, encoding="utf-8", errors="replace")
        text, code = p.stdout + "\n--- stderr ---\n" + p.stderr, p.returncode
    except Exception as e:
        text, code = f"could not run {cmd}: {e}", -1
    Path(out_file).write_text(text, encoding="utf-8")
    return code, round(time.perf_counter() - t0, 1), text


def check(name):
    """Decorator: record PASS/FAIL/SKIP with details, never crash the script."""
    def deco(fn):
        def wrapper(*a, **k):
            try:
                detail = fn(*a, **k)
                status = "SKIP" if isinstance(detail, str) and detail.startswith("SKIP") else "PASS"
            except AssertionError as e:
                status, detail = "FAIL", f"assertion: {e}"
            except Exception as e:
                status, detail = "ERROR", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
            RESULTS["checks"][name] = {"status": status, "detail": detail}
            log(f"  [{status}] {name}")
            return detail
        return wrapper
    return deco


# --------------------------------------------------------------------------
# 1. environment and hardware
# --------------------------------------------------------------------------
def environment():
    env = {"python_version": sys.version, "executable": sys.executable,
           "prefix": sys.prefix, "base_prefix": sys.base_prefix,
           "in_venv": sys.prefix != sys.base_prefix,
           "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
           "platform": platform.platform(), "machine": platform.machine()}
    import importlib.util
    pk = {}
    for p in PACKAGES:
        module = MODULES.get(p, p)
        if importlib.util.find_spec(module) is None:
            pk[p] = None            # not importable -> treated as missing
            continue
        pk[p] = "installed (version unknown)"
        for dist in (p, *ALT_DISTS.get(p, ())):   # e.g. xgboost ships as xgboost-cpu
            try:
                pk[p] = md.version(dist)
                break
            except md.PackageNotFoundError:
                pass
    env["packages"] = pk
    try:
        env["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           capture_output=True, text=True).stdout.strip()
        env["git_status"] = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                           capture_output=True, text=True).stdout.splitlines()
    except Exception as e:
        env["git_commit"] = f"unavailable: {e}"
    return env


def hardware():
    hw = {"logical_cpus": os.cpu_count()}
    if sys.platform == "win32":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            hw["cpu_name"] = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except Exception as e:
            hw["cpu_name"] = f"unavailable: {e}"
        try:
            import ctypes

            class MEM(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MEM(); m.dwLength = ctypes.sizeof(MEM)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            hw["ram_total_gb"] = round(m.ullTotalPhys / 2**30, 1)
            hw["ram_available_gb"] = round(m.ullAvailPhys / 2**30, 1)
        except Exception as e:
            hw["ram"] = f"unavailable: {e}"
        try:
            ps = ("Get-CimInstance Win32_Processor | Select-Object -First 1 "
                  "NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Json; "
                  "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM "
                  "| ConvertTo-Json")
            p = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                               capture_output=True, text=True, timeout=60)
            hw["cim"] = p.stdout.strip()
        except Exception as e:
            hw["cim"] = f"unavailable: {e}"
    else:
        hw["cpu_name"] = platform.processor()
        try:
            hw["ram_total_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
        except Exception:
            pass
    try:
        import torch
        hw["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            hw["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        hw["torch_cuda_available"] = None
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    hw["chronos_t5_small_cached"] = (hub / "models--amazon--chronos-t5-small").exists()
    try:
        import shutil
        hw["disk_free_gb_project_drive"] = round(shutil.disk_usage(ROOT).free / 2**30, 1)
    except Exception:
        pass
    return hw


# --------------------------------------------------------------------------
# 4. protocol checks on the real data
# --------------------------------------------------------------------------
def protocol_checks():
    import numpy as np
    import pandas as pd
    from src import backtest, config, data, forecasters, protocol

    df, quality = data.load_checked()
    parts = protocol.make_partitions(df)
    RESULTS["data_quality"] = quality
    RESULTS["partitions"] = parts.summary()
    H = config.HORIZON
    hour = pd.Timedelta(hours=1)

    @check("data: exactly one incomplete hour removed (2016-05-27 18:00, 1/6 readings)")
    def _():
        assert quality["expected_readings_per_hour"] == 6
        assert quality["incomplete_hours"] == {"2016-05-27 18:00:00": 1}, quality["incomplete_hours"]
        assert df.index.is_monotonic_increasing and df.index.freq == "h"
        return f"{quality['hourly_bins_before']} -> {quality['hourly_bins_after']} hourly bins"
    _()

    @check("splits: 30 test + 14 validation origins, all with complete 24-h targets and 168-h look-back")
    def _():
        assert len(parts.test) == 30 and len(parts.validation) == 14
        for name, origins in (("validation", parts.validation), ("test", parts.test)):
            for o in origins:
                tgt = protocol.target_index(o, H)
                assert o.hour == config.ORIGIN_HOUR
                assert len(tgt) == 24 and tgt[0] == o + hour and tgt[-1] == o + 24 * hour
                assert tgt.isin(df.index).all(), f"{name} {o}: target hour missing"
                assert df.loc[tgt, config.TARGET].notna().all(), f"{name} {o}: NaN target"
                look = df.loc[o - 167 * hour:o, config.TARGET]
                assert len(look) == 168 and look.notna().all(), f"{name} {o}: look-back"
        return "44 origins x 24 targets complete"
    _()

    @check("splits: chronological, non-overlapping, test targets after train/validation")
    def _():
        val_t = set().union(*[set(protocol.target_index(o)) for o in parts.validation])
        tst_t = set().union(*[set(protocol.target_index(o)) for o in parts.test])
        assert not val_t & tst_t
        assert len(tst_t) == 30 * 24 and len(val_t) == 14 * 24
        assert max(val_t) <= parts.test[0] < min(tst_t)
        assert parts.train_end == parts.validation[0] < min(val_t)
        return (f"validation targets {min(val_t)} .. {max(val_t)}; "
                f"test targets {min(tst_t)} .. {max(tst_t)}")
    _()

    @check("operational future view = calendar features only")
    def _():
        for o in list(parts.validation) + list(parts.test):
            fut = protocol.future_view(df, o, H, "operational")
            assert list(fut.columns) == protocol.CALENDAR_COLS
        return "checked all 44 origins"
    _()

    have_xgb = RESULTS["env"]["packages"].get("xgboost") is not None
    have_sm = RESULTS["env"]["packages"].get("statsmodels") is not None

    def models():
        m = {n: forecasters.build(n) for n in forecasters.BENCHMARK_NAMES}
        if have_xgb:
            m["xgboost"] = forecasters.build("xgboost")
        if have_sm:   # cheap order: this checks plumbing, not model quality
            m["sarimax"] = forecasters.build("sarimax", sarimax_order=(1, 0, 1),
                                             sarimax_seasonal_order=(1, 0, 0, 24),
                                             sarimax_order_source="verification only")
        return m

    @check("identical origins and target timestamps for every model (validation origins)")
    def _():
        res = backtest.run_backtest(df, models(), parts.validation, verbose=False)
        f = res["forecasts"]
        expected = {(o, t) for o in parts.validation for t in protocol.target_index(o)}
        per = {m: set(zip(g.origin, g.target_time)) for m, g in f.groupby("model")}
        for m, k in per.items():
            assert k == expected, f"{m} differs"
        return f"models: {sorted(per)}; {len(expected)} (origin, target) pairs each"
    _()

    def perturb_after(d, origin, keep=()):
        d = d.copy()
        rng = np.random.default_rng(7)
        after = d.index > origin
        for c in d.columns:
            if c not in keep:
                d.loc[after, c] = rng.normal(5000, 2000, after.sum())
        return d

    @check("leakage: operational forecasts unchanged when ALL data after the origin is replaced")
    def _():
        out = []
        for o in parts.validation[[0, -1]]:
            a = backtest.run_backtest(df, models(), [o], verbose=False)["forecasts"]
            b = backtest.run_backtest(perturb_after(df, o), models(), [o], verbose=False)["forecasts"]
            assert np.array_equal(a["yhat"].to_numpy(), b["yhat"].to_numpy()), f"changed at {o}"
            assert not np.allclose(a["y_true"], b["y_true"])
            out.append(str(o))
        return f"origins {out}; models {sorted(models())}"
    _()

    @check("leakage: weather_oracle uses ONLY outdoor weather from the future")
    def _():
        o = parts.validation[0]
        keep = protocol.SCENARIOS["weather_oracle"]["future_columns"]
        a = backtest.run_backtest(df, models(), [o], scenario="weather_oracle",
                                  verbose=False)["forecasts"]
        b = backtest.run_backtest(perturb_after(df, o, keep), models(), [o],
                                  scenario="weather_oracle", verbose=False)["forecasts"]
        assert np.array_equal(a["yhat"].to_numpy(), b["yhat"].to_numpy())
        if have_xgb:   # and it really does use the weather -> must stay labelled
            c = backtest.run_backtest(perturb_after(df, o), {"xgboost": forecasters.build(
                "xgboost", "weather_oracle")}, [o], scenario="weather_oracle",
                verbose=False)["forecasts"]
            assert not np.allclose(a[a.model == "xgboost"]["yhat"].to_numpy(),
                                   c["yhat"].to_numpy())
        return "non-weather future values have no effect; weather does (upper bound)"
    _()

    @check("XGBoost recursion diagnostic (validation only)")
    def _():
        if not have_xgb:
            return "SKIP: xgboost not installed"
        from src import diagnostics
        out, d = diagnostics.xgb_recursion_diagnostic(df, parts)
        RESULTS["xgb_recursion"] = out.round(2).to_dict(orient="records")
        rec = float(np.sqrt(((d.recursive - d.y_true) ** 2).mean()))
        tf = float(np.sqrt(((d.teacher_forced - d.y_true) ** 2).mean()))
        RESULTS["xgb_recursion_pooled"] = {"recursive_RMSE": rec, "teacher_forced_RMSE": tf}
        return f"validation pooled RMSE recursive {rec:.1f} vs teacher-forced {tf:.1f}"
    _()
    return df, parts


# --------------------------------------------------------------------------
# 5. timing
# --------------------------------------------------------------------------
def timing(df, parts):
    import numpy as np
    from src import config, protocol, forecasters
    t = {}
    have = RESULTS["env"]["packages"]

    @check("timing: XGBoost fit + one 24-step recursive forecast")
    def _():
        if have.get("xgboost") is None:
            return "SKIP: xgboost not installed"
        o = parts.validation[0]
        t0 = time.perf_counter()
        m = forecasters.build("xgboost").fit(protocol.history_view(df, o))
        t["xgb_fit_s"] = round(time.perf_counter() - t0, 2)
        t0 = time.perf_counter()
        m.predict(protocol.history_view(df, o), protocol.future_view(df, o))
        t["xgb_forecast_s"] = round(time.perf_counter() - t0, 2)
        return t
    _()

    @check("timing: SARIMAX single fits on the 45-day training window")
    def _():
        if have.get("statsmodels") is None:
            return "SKIP: statsmodels not installed"
        from src.sarimax_selection import fit_with_diagnostics, training_window
        y, _x = training_window(df, parts.train_end)
        fits = []
        for order in [(1, 0, 1), (2, 1, 2), (3, 0, 6)]:
            _r, info = fit_with_diagnostics(y, order, (1, 0, 1, 24), None, maxiter=200)
            fits.append({"order": order, "seconds": info["seconds"],
                         "converged": info["converged"], "n_warnings": info["n_warnings"]})
        t["sarimax_fits"] = fits
        mean_s = float(np.mean([f["seconds"] for f in fits]))
        t["sarimax_grid_147_estimate_min_single_process"] = round(147 * mean_s / 60, 1)
        return fits
    _()

    @check("timing: Chronos one forecast (only if weights already cached; offline)")
    def _():
        if have.get("torch") is None or have.get("chronos-forecasting") is None:
            return "SKIP: torch / chronos-forecasting not installed"
        if not RESULTS["hardware"].get("chronos_t5_small_cached"):
            return "SKIP: amazon/chronos-t5-small not cached - not downloading without approval"
        os.environ["HF_HUB_OFFLINE"] = "1"
        o = parts.validation[0]
        m = forecasters.build("chronos")
        t0 = time.perf_counter()
        m.predict(protocol.history_view(df, o), protocol.future_view(df, o))
        t["chronos_first_forecast_s_incl_load"] = round(time.perf_counter() - t0, 2)
        t0 = time.perf_counter()
        m.predict(protocol.history_view(df, o), protocol.future_view(df, o))
        t["chronos_forecast_s"] = round(time.perf_counter() - t0, 2)
        return t
    _()
    RESULTS["timing"] = t


def junit_counts(path):
    try:
        root = ET.parse(path).getroot()
        s = root if root.tag == "testsuite" else root.find("testsuite")
        tests, fail, err, skip = (int(s.get(k, 0)) for k in ("tests", "failures", "errors", "skipped"))
        failed = [f"{c.get('classname')}::{c.get('name')}" for c in s.iter("testcase")
                  if c.find("failure") is not None or c.find("error") is not None]
        skipped = [f"{c.get('classname')}::{c.get('name')}: {c.find('skipped').get('message', '')}"
                   for c in s.iter("testcase") if c.find("skipped") is not None]
        return {"total": tests, "passed": tests - fail - err - skip, "failed": fail,
                "errors": err, "skipped": skip, "failed_tests": failed, "skipped_tests": skipped}
    except Exception as e:
        return {"error": f"could not parse junit xml: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--skip-pytest", action="store_true")
    a = ap.parse_args()
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out = ROOT / "outputs" / "verification" / f"phase2b_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    log(f"Phase 2B verification -> {out}")

    log("1. environment")
    RESULTS["env"] = environment()
    RESULTS["hardware"] = hardware()
    run([sys.executable, "-m", "pip", "freeze"], out / "pip_freeze.txt", 300)

    if not a.skip_pytest:
        log("2. pytest (full suite)")
        code, secs, _ = run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider",
                             "-rA", "--junitxml", str(out / "pytest.xml")], out / "pytest.txt")
        RESULTS["pytest"] = {"exit_code": code, "seconds": secs, **junit_counts(out / "pytest.xml")}
        log(f"   {RESULTS['pytest']}")

    log("3. dry run")
    code, secs, text = run([sys.executable, "run_backtest.py", "--dry-run"], out / "dry_run.txt")
    RESULTS["dry_run"] = {"exit_code": code, "seconds": secs}

    log("4. protocol checks on real data")
    df = parts = None
    try:
        df, parts = protocol_checks()
    except Exception as e:
        RESULTS["checks"]["protocol setup"] = {"status": "ERROR", "detail":
                                               f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}"}
    if not a.skip_timing and df is not None:
        log("5. timing")
        timing(df, parts)

    (out / "verification.json").write_text(json.dumps(RESULTS, indent=2, default=str),
                                           encoding="utf-8")
    statuses = [c["status"] for c in RESULTS["checks"].values()]
    summary = {s: statuses.count(s) for s in ("PASS", "FAIL", "ERROR", "SKIP")}
    log(f"\nchecks: {summary}")
    if "pytest" in RESULTS:
        p = RESULTS["pytest"]
        log(f"pytest: passed {p.get('passed')}, failed {p.get('failed')}, "
            f"errors {p.get('errors')}, skipped {p.get('skipped')}")
    log(f"dry run exit code: {RESULTS['dry_run']['exit_code']}")
    log(f"All logs in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
