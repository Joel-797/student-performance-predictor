from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "students.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "performance_model.joblib"

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
