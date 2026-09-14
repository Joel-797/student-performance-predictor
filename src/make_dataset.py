"""Create a reproducible synthetic student-performance dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DATA_PATH, PARENT_EDUCATION_LEVELS, PASS_THRESHOLD, RANDOM_STATE


def build_dataset(n_rows: int = 250, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parent_education = rng.choice(
        PARENT_EDUCATION_LEVELS,
        size=n_rows,
        p=[0.28, 0.18, 0.32, 0.16, 0.06],
    )
    education_bonus = {
        "high_school": 0,
        "associates": 1.5,
        "bachelors": 3.0,
        "masters": 4.5,
        "phd": 5.5,
    }

    hours_studied = np.clip(rng.normal(10.5, 4.2, n_rows), 1.0, 25.0)
    attendance_pct = np.clip(rng.normal(82, 12, n_rows), 40.0, 100.0)
    previous_grade = np.clip(rng.normal(68, 12, n_rows), 30.0, 98.0)
    sleep_hours = np.clip(rng.normal(6.8, 1.1, n_rows), 3.5, 10.0)
    extracurricular = rng.binomial(1, 0.45, n_rows)

    score = (
        -8.0
        + 1.55 * hours_studied
        + 0.22 * attendance_pct
        + 0.38 * previous_grade
        + 1.1 * sleep_hours
        + 2.0 * extracurricular
        + np.array([education_bonus[level] for level in parent_education])
        + rng.normal(0, 5.0, n_rows)
    )
    final_score = np.clip(np.round(score, 1), 0, 100)

    return pd.DataFrame(
        {
            "student_id": [f"S{i:03d}" for i in range(1, n_rows + 1)],
            "hours_studied": np.round(hours_studied, 1),
            "attendance_pct": np.round(attendance_pct, 1),
            "previous_grade": np.round(previous_grade, 1),
            "sleep_hours": np.round(sleep_hours, 1),
            "extracurricular": extracurricular,
            "parent_education": parent_education,
            "final_score": final_score,
            "passed": (final_score >= PASS_THRESHOLD).astype(int),
        }
    )


def main() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = build_dataset()
    df.to_csv(DATA_PATH, index=False)
    print(f"Wrote {len(df)} rows to {DATA_PATH}")


if __name__ == "__main__":
    main()
