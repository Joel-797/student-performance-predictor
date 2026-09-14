"""Train, evaluate, and save the exam-score model."""

from __future__ import annotations

import json

import numpy as np
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import METRICS_PATH, MODEL_DIR, MODEL_PATH, REPORTS_DIR
from src.data import (
    categorical_feature_names,
    load_students,
    numeric_feature_names,
    split_data,
)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_feature_names()),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_feature_names(),
            ),
        ]
    )


def candidate_models() -> dict[str, object]:
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=8,
            random_state=42,
        ),
    }


def evaluate(y_true, y_pred) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main() -> None:
    df = load_students()
    x_train, x_test, y_train, y_test = split_data(df)

    comparison: dict[str, dict[str, float]] = {}
    fitted: dict[str, Pipeline] = {}

    for name, estimator in candidate_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        preds = pipeline.predict(x_test)
        comparison[name] = evaluate(y_test, preds)
        fitted[name] = pipeline
        print(
            f"{name}: MAE={comparison[name]['mae']:.2f} "
            f"RMSE={comparison[name]['rmse']:.2f} "
            f"R^2={comparison[name]['r2']:.3f}"
        )

    best_name = min(comparison, key=lambda name: comparison[name]["mae"])
    best_pipeline = fitted[best_name]
    best_metrics = comparison[best_name]
    train_preds = best_pipeline.predict(x_train)
    test_preds = best_pipeline.predict(x_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    dump(best_pipeline, MODEL_PATH)

    report = {
        "best_model": best_name,
        "n_rows": int(len(df)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "comparison": comparison,
        "train": evaluate(y_train, train_preds),
        "test": best_metrics,
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nBest model: {best_name}")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print("Sample test predictions:")
    for actual, pred in zip(y_test.head(5), test_preds[:5], strict=False):
        print(f"  actual={actual:.1f}  predicted={pred:.1f}")


if __name__ == "__main__":
    main()
