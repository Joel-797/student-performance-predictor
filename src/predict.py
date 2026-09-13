"""Predict a student's final score from study habits."""

from __future__ import annotations

import argparse

import pandas as pd
from joblib import load

from src.config import FEATURE_COLUMNS, MODEL_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict student final score")
    parser.add_argument("--hours-studied", type=float, required=True)
    parser.add_argument("--attendance-pct", type=float, required=True)
    parser.add_argument("--previous-grade", type=float, required=True)
    parser.add_argument("--sleep-hours", type=float, required=True)
    parser.add_argument("--extracurricular", type=int, choices=[0, 1], required=True)
    parser.add_argument(
        "--parent-education",
        choices=["high_school", "associates", "bachelors", "masters", "phd"],
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit("No trained model found. Run: python -m src.train")

    args = parse_args()
    row = pd.DataFrame(
        [
            {
                "hours_studied": args.hours_studied,
                "attendance_pct": args.attendance_pct,
                "previous_grade": args.previous_grade,
                "sleep_hours": args.sleep_hours,
                "extracurricular": args.extracurricular,
                "parent_education": args.parent_education,
            }
        ]
    )[FEATURE_COLUMNS]

    model = load(MODEL_PATH)
    score = float(model.predict(row)[0])
    passed = score >= 60
    print(f"Predicted final score: {score:.1f}")
    print(f"Predicted outcome: {'pass' if passed else 'fail'}")


if __name__ == "__main__":
    main()
