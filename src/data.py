"""Load and split the student-performance dataset."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CATEGORICAL_COLUMNS,
    DATA_PATH,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    RANDOM_STATE,
    TARGET_SCORE,
    TEST_SIZE,
)


def load_students(path: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA_PATH)
    missing = [col for col in FEATURE_COLUMNS + [TARGET_SCORE] if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    return df


def features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLUMNS].copy(), df[TARGET_SCORE].copy()


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x, y = features_and_target(df)
    return train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def numeric_feature_names() -> list[str]:
    return list(NUMERIC_COLUMNS)


def categorical_feature_names() -> list[str]:
    return list(CATEGORICAL_COLUMNS)
