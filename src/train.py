"""Train a student performance model and save it to models/."""

from __future__ import annotations

import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_COLUMNS,
    DATA_PATH,
    FEATURE_COLUMNS,
    MODEL_DIR,
    MODEL_PATH,
    TARGET_SCORE,
)


def build_pipeline() -> Pipeline:
    numeric = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]
    )


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    x = df[FEATURE_COLUMNS]
    y = df[TARGET_SCORE]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    preds = pipeline.predict(x_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dump(pipeline, MODEL_PATH)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Test MAE: {mae:.2f}")
    print(f"Test R^2: {r2:.3f}")
    print("Sample predictions:")
    for actual, pred in zip(y_test.head(5), preds[:5], strict=False):
        print(f"  actual={actual:.1f}  predicted={pred:.1f}")


if __name__ == "__main__":
    main()
