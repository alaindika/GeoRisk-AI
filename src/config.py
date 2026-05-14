from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"

# Reproducibility
RANDOM_SEED = 42

# Target labels
LABEL_MAP = {
    0: "Low",
    1: "Medium",
    2: "High",
}

RISK_TO_LABEL = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
}

# Core model features
FEATURE_COLUMNS = [
    "rainfall_48hr",
    "pressure_drop_3hr",
    "temp_anomaly",
    "wind_speed",
    "humidity",
    "snowfall_rate",
    "season",
]

TARGET_COLUMN = "risk_label"

# Model outputs
MODEL_PATH = ARTIFACTS_DIR / "georisk_rf_model.joblib"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.json"

# Synthetic dataset settings
N_SAMPLES = 6000