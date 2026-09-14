"""Predict a student's final exam score from study-related features."""

from __future__ import annotations

import argparse

import pandas as pd
from joblib import load

from src.config import (
    FEATURE_COLUMNS,
    MODEL_PATH,
    PARENT_EDUCATION_LEVELS,
    PASS_THRESHOLD,
)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run: python -m src.train"
        )
    return load(MODEL_PATH)


def student_frame(
    hours_studied: float,
    attendance_pct: float,
    previous_grade: float,
    sleep_hours: float,
    extracurricular: int,
    parent_education: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "hours_studied": hours_studied,
                "attendance_pct": attendance_pct,
                "previous_grade": previous_grade,
                "sleep_hours": sleep_hours,
                "extracurricular": extracurricular,
                "parent_education": parent_education,
            }
        ]
    )[FEATURE_COLUMNS]


def predict_score(
    hours_studied: float,
    attendance_pct: float,
    previous_grade: float,
    sleep_hours: float,
    extracurricular: int,
    parent_education: str,
    model=None,
) -> dict[str, float | str]:
    if model is None:
        model = load_model()
    row = student_frame(
        hours_studied,
        attendance_pct,
        previous_grade,
        sleep_hours,
        extracurricular,
        parent_education,
    )
    score = float(model.predict(row)[0])
    score = max(0.0, min(100.0, score))
    outcome = "pass" if score >= PASS_THRESHOLD else "fail"
    return {"predicted_score": round(score, 1), "predicted_outcome": outcome}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict student final score")
    parser.add_argument("--hours-studied", type=float, required=True)
    parser.add_argument("--attendance-pct", type=float, required=True)
    parser.add_argument("--previous-grade", type=float, required=True)
    parser.add_argument("--sleep-hours", type=float, required=True)
    parser.add_argument("--extracurricular", type=int, choices=[0, 1], required=True)
    parser.add_argument("--parent-education", choices=PARENT_EDUCATION_LEVELS, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict_score(
        args.hours_studied,
        args.attendance_pct,
        args.previous_grade,
        args.sleep_hours,
        args.extracurricular,
        args.parent_education,
    )
    print(f"Predicted final score: {result['predicted_score']}")
    print(f"Predicted outcome: {result['predicted_outcome']}")


if __name__ == "__main__":
    main()
