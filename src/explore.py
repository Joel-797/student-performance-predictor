"""Quick charts of the student dataset. Run this file in PyCharm."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import DATA_PATH, ROOT


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(df.describe(include="all"))
    print("\nPass rate:", df["passed"].mean())

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.scatterplot(
        data=df,
        x="hours_studied",
        y="final_score",
        hue="passed",
        ax=axes[0],
    )
    axes[0].set_title("Study time vs final score")
    sns.boxplot(data=df, x="parent_education", y="final_score", ax=axes[1])
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_title("Score by parent education")
    fig.tight_layout()
    out = ROOT / "models" / "eda.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"Saved chart to {out}")
    plt.show()


if __name__ == "__main__":
    main()
