from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "students.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "performance_model.joblib"
REPORTS_DIR = ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "metrics.json"
EDA_PLOT_PATH = REPORTS_DIR / "eda.png"

FEATURE_COLUMNS = [
    "hours_studied",
    "attendance_pct",
    "previous_grade",
    "sleep_hours",
    "extracurricular",
    "parent_education",
]
TARGET_SCORE = "final_score"
TARGET_PASSED = "passed"
CATEGORICAL_COLUMNS = ["parent_education"]
NUMERIC_COLUMNS = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]

PARENT_EDUCATION_LEVELS = [
    "high_school",
    "associates",
    "bachelors",
    "masters",
    "phd",
]
PASS_THRESHOLD = 60
RANDOM_STATE = 42
TEST_SIZE = 0.2
