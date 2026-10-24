"""Central configuration for the appliance-energy forecasting pipeline.

Keeping the knobs in one place makes the whole study reproducible: change a
value here and every part of the pipeline picks it up.
"""
from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
FIG_DIR = ROOT / "outputs" / "figures"
METRIC_DIR = ROOT / "outputs" / "metrics"
for _d in (DATA_DIR, FIG_DIR, METRIC_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# The committed results of the original single-window study live here.
# They are research history: nothing may overwrite them by accident.
LEGACY_METRIC_DIR = METRIC_DIR
LEGACY_FIG_DIR = FIG_DIR
ALLOW_LEGACY_ENV = "FORECAST_ALLOW_LEGACY_OVERWRITE"


def assert_legacy_write_allowed(what: str = "this step"):
    """Refuse to overwrite the historical outputs/metrics and outputs/figures.

    Writing is allowed only when (a) the target folders are not the legacy
    ones (tests redirect them to a temp dir), (b) they hold no results yet,
    or (c) the user explicitly opted in via ``run_pipeline.py
    --overwrite-legacy`` or the environment variable above set to "1".
    """
    import os
    if os.environ.get(ALLOW_LEGACY_ENV) == "1":
        return
    legacy = {LEGACY_METRIC_DIR.resolve(), LEGACY_FIG_DIR.resolve()}
    targets = [d for d in (METRIC_DIR, FIG_DIR) if d.resolve() in legacy]
    existing = [p for d in targets if d.exists()
                for p in d.iterdir() if p.name != ".gitkeep"]
    if existing:
        raise RuntimeError(
            f"{what} would overwrite {len(existing)} historical result file(s) "
            f"in outputs/metrics or outputs/figures. These are the committed "
            f"results of the original study. Use run_backtest.py for new "
            f"experiments, or re-run the legacy study deliberately with "
            f"`python run_pipeline.py --overwrite-legacy` "
            f"(or set {ALLOW_LEGACY_ENV}=1).")

# --- Data ------------------------------------------------------------------
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00374/energydata_complete.csv"
)
# A couple of mirrors are tried in order if the primary UCI host is down.
DATA_MIRRORS = [
    DATA_URL,
    "https://raw.githubusercontent.com/LuisM78/"
    "Appliances-energy-prediction-data/master/energydata_complete.csv",
]
RAW_CSV = DATA_DIR / "energydata_complete.csv"

# --- Forecasting problem ---------------------------------------------------
TARGET = "Appliances"          # appliance energy use in Wh (10-min original)
FREQ = "h"                     # we model at hourly resolution
HORIZON = 24                   # forecast the next 24 hours
DAILY_SEASON = 24              # hours in a day
WEEKLY_SEASON = 24 * 7         # hours in a week
ML_TEST_DAYS = 14              # legacy XGBoost run: last 14 days held out

# --- SARIMAX grid search ---------------------------------------------------
# Original study grid: p in [0,6], d in [0,2], q in [0,6] (147 models).
P_RANGE = range(0, 7)
D_RANGE = range(0, 3)
Q_RANGE = range(0, 7)
# Seasonal search is kept small because a daily period (m=24) is expensive.
SEASONAL_ORDER_CANDIDATES = [(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1)]
SARIMAX_TRAIN_DAYS = 45        # cap history for tractable m=24 SARIMA fits

RANDOM_STATE = 42

# --- Rolling-origin backtest (Phase 2A) -------------------------------------
# A forecast origin is the timestamp of the LAST observed hour. The forecast
# targets are origin + 1h ... origin + HORIZON h. Nothing after the origin is
# visible to any model (see src/protocol.py).
BACKTEST_DIR = ROOT / "outputs" / "backtest"   # never writes to outputs/metrics
ORIGIN_HOUR = 23            # origin at 23:00 -> forecast the next calendar day
ORIGIN_STEP_HOURS = 24      # one origin per day -> non-overlapping test windows
N_TEST_ORIGINS = 30         # final block of eligible origins
N_VAL_ORIGINS = 14          # block immediately before the test origins
MIN_TRAIN_HOURS = 24 * 28   # minimum history before the first validation origin
MAX_LOOKBACK_HOURS = 168    # longest lag used by any feature (weekly)
EXPECTED_READING_MINUTES = 10  # raw sampling interval (verified from the data)
