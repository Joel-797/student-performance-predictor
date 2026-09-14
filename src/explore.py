"""Exploratory data analysis for the student dataset."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import DATA_PATH, EDA_PLOT_PATH, REPORTS_DIR, TARGET_SCORE
from src.data import load_students


def summarize(df: pd.DataFrame) -> None:
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nNumeric summary:")
    print(df.describe(include="number").round(2))
    print("\nParent education counts:")
    print(df["parent_education"].value_counts())
    print(f"\nPass rate: {df['passed'].mean():.2%}")
    numeric = df.select_dtypes(include="number")
    print("\nCorrelation with final_score:")
    print(numeric.corr()[TARGET_SCORE].sort_values(ascending=False).round(3))


def save_plots(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    sns.histplot(df[TARGET_SCORE], bins=20, ax=axes[0, 0], kde=True)
    axes[0, 0].set_title("Final score distribution")

    sns.scatterplot(
        data=df,
        x="hours_studied",
        y=TARGET_SCORE,
        hue="passed",
        ax=axes[0, 1],
        alpha=0.75,
    )
    axes[0, 1].set_title("Study hours vs final score")

    sns.boxplot(data=df, x="parent_education", y=TARGET_SCORE, ax=axes[1, 0])
    axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 0].set_title("Score by parent education")

    corr = df[
        [
            "hours_studied",
            "attendance_pct",
            "previous_grade",
            "sleep_hours",
            "extracurricular",
            TARGET_SCORE,
        ]
    ].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=axes[1, 1])
    axes[1, 1].set_title("Feature correlations")

    fig.tight_layout()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(EDA_PLOT_PATH, dpi=120)
    print(f"Saved EDA chart to {EDA_PLOT_PATH}")
    plt.close(fig)


def main() -> None:
    df = load_students(DATA_PATH)
    summarize(df)
    save_plots(df)


if __name__ == "__main__":
    main()
