from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "cleaned.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
EDA_DIR = REPORTS_DIR / "eda"
MODEL_EVAL_DIR = REPORTS_DIR / "model_evaluation"


def ensure_project_dirs() -> None:
    for directory in [DATA_PATH.parent, MODELS_DIR, REPORTS_DIR, EDA_DIR, MODEL_EVAL_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path | str = DATA_PATH) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    return pd.read_csv(dataset_path)


def save_json(data: Dict[str, Any], path: Path | str) -> None:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def load_json(path: Path | str) -> Dict[str, Any]:
    target_path = Path(path)
    if not target_path.exists():
        return {}
    with target_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    text = " ".join(text.split())
    if text.lower() in {"", "unknown", "nan", "n/a", "na", "none", "null"}:
        return "Unknown"
    return text


def standardize_string_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_text)


def infer_risk_mapping(target_series: Iterable[Any]) -> Dict[Any, str]:
    values = list(dict.fromkeys(pd.Series(target_series).dropna().astype(str).str.strip().tolist()))
    if not values:
        return {}
    try:
        numeric_values = sorted(pd.to_numeric(values, errors="coerce").dropna().astype(int).tolist())
    except Exception:
        numeric_values = sorted([int(float(v)) for v in values if str(v).strip() not in {"", "Unknown"}])

    if not numeric_values:
        return {value: f"Class {value}" for value in values}

    counts = pd.Series(target_series).dropna().astype(int).value_counts().sort_index()
    if len(numeric_values) == 1:
        return {numeric_values[0]: "High"}

    low_label = int(counts.idxmax())
    high_label = int(counts.idxmin())
    unique_labels = numeric_values
    middle_labels = [value for value in unique_labels if value not in {low_label, high_label}]
    middle_label = middle_labels[0] if middle_labels else unique_labels[1]
    mapping = {low_label: "Low", middle_label: "Medium", high_label: "High"}
    return mapping


def safe_load_model(path: Path | str):
    import joblib

    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)
