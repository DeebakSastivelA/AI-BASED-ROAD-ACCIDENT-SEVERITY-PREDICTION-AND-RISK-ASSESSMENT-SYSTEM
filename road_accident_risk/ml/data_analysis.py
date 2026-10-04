from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ml.preprocessing import TARGET_COLUMN, clean_dataset
from ml.utils import DATA_PATH, EDA_DIR, ensure_project_dirs, load_dataset


def summarize_dataset(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": df[TARGET_COLUMN].value_counts().sort_index().to_dict(),
    }


def save_target_distribution(df: pd.DataFrame, output_path: Path) -> None:
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    percentages = counts / counts.sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    counts.plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].set_title("Accident Severity Distribution")
    axes[0].set_xlabel("Severity Class")
    axes[0].set_ylabel("Count")
    for bar, value in zip(axes[0].patches, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 10, str(value), ha="center", va="bottom")

    percentages.plot(kind="bar", ax=axes[1], color="darkorange")
    axes[1].set_title("Accident Severity Percentages")
    axes[1].set_xlabel("Severity Class")
    axes[1].set_ylabel("Percentage (%)")
    for bar, value in zip(axes[1].patches, percentages.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.5, f"{value:.1f}%", ha="center", va="bottom")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_categorical_plot(df: pd.DataFrame, feature_name: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    count_plot = df[feature_name].value_counts().head(10)
    count_plot.plot(kind="bar", ax=axes[0], color="mediumseagreen")
    axes[0].set_title(f"{feature_name} Distribution")
    axes[0].set_xlabel(feature_name)
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=30)

    severity_table = pd.crosstab(df[feature_name], df[TARGET_COLUMN]).sort_index()
    sns.heatmap(severity_table, annot=False, cmap="YlOrRd", ax=axes[1])
    axes[1].set_title(f"{feature_name} vs Severity")
    axes[1].set_xlabel("Severity")
    axes[1].set_ylabel(feature_name)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_eda_charts(df: pd.DataFrame) -> None:
    feature_names = [
        "Weather_conditions",
        "Light_conditions",
        "Types_of_Junction",
        "Road_surface_type",
        "Type_of_collision",
        "Driving_experience",
        "Cause_of_accident",
        "Vehicle_movement",
    ]

    save_target_distribution(df, EDA_DIR / "target_distribution.png")
    for feature in feature_names:
        save_categorical_plot(df, feature, EDA_DIR / f"{feature.lower()}_vs_severity.png")


def main() -> None:
    ensure_project_dirs()
    raw_df = load_dataset(DATA_PATH)
    cleaned_df = clean_dataset(raw_df)
    summary = summarize_dataset(cleaned_df)
    print(summary)
    generate_eda_charts(cleaned_df)
    print(f"EDA charts saved to {EDA_DIR}")


if __name__ == "__main__":
    main()
