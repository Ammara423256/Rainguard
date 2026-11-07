
# config.py
from pathlib import Path

# --- Paths ---
PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH    = PROJECT_DIR / "Data_Input (2).csv"

# --- Reproducibility ---
SEED = 8456

# --- Data schema ---
STATIONS   = ["MABAD", "MANGLA", "MURREE"]
DATE_COL   = "Date"
TARGET_COL = "Raintomorrow"

# --- QC thresholds ---
MAD_K                = 5.0
MIN_PLAUSIBLE_TEMP   = -25.0
MAX_PLAUSIBLE_TEMP   = 55.0
MIN_PLAUSIBLE_PRECIP = 0.0
MAX_PLAUSIBLE_PRECIP = 400.0
PLACEHOLDER_RATIO    = 0.90

# Alert thresholds = 60th, 80th, 95th percentiles of pooled-RF
# predicted probability on the 1971-2003 training set.
ALERT_TIERS = {
    "Normal":  {"min": 0.00, "action": "No action. Log and monitor."},
    "Watch":   {"min": 0.31, "action": "Notify duty officer; pre-stage sandbags."},
    "Warning": {"min": 0.50, "action": "Alert local response team; check drains."},
    "Heavy":   {"min": 0.69, "action": "Escalate to district authority (HUMAN APPROVAL required)."},
}
MODEL_PARAMS = {"n_estimators": 300, "max_depth": 8, "random_state": SEED}
