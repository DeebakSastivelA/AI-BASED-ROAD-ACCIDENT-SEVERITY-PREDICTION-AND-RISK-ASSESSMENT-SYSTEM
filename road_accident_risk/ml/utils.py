from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
PRIMARY_DATASET_CANDIDATES = [
    WORKSPACE_ROOT / "road_accident_balanced_reality_based_18000.csv",
    PROJECT_ROOT / "road_accident_balanced_reality_based_18000.csv",
    WORKSPACE_ROOT / "cleaned.csv",
    PROJECT_ROOT / "data" / "cleaned.csv",
]
DATA_PATH = next((path for path in PRIMARY_DATASET_CANDIDATES if path.exists()), WORKSPACE_ROOT / "road_accident_balanced_reality_based_18000.csv")
LEGACY_DATASET_PATH = WORKSPACE_ROOT / "cleaned.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
EDA_DIR = REPORTS_DIR / "eda"
MODEL_EVAL_DIR = REPORTS_DIR / "model_evaluation"

SEVERITY_MAP = {
    0: "Fatal Injury",
    1: "Serious Injury",
    2: "Slight Injury",
}

RISK_MAP = {
    0: "Very High",
    1: "High",
    2: "Low",
}


def ensure_project_dirs() -> None:
    for directory in [MODELS_DIR, REPORTS_DIR, EDA_DIR, MODEL_EVAL_DIR, PROJECT_ROOT]:
        directory.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path | str = DATA_PATH) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            "road_accident_balanced_reality_based_18000.csv must be placed in the project root next to cleaned.csv. "
            f"Dataset not found: {dataset_path}"
        )
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
    numeric_values = pd.Series(target_series).dropna().astype(int).unique().tolist()
    if not numeric_values:
        return {}
    mapping = {}
    for value in sorted(set(numeric_values)):
        if value in SEVERITY_MAP:
            mapping[value] = RISK_MAP.get(value, "Unknown")
    return mapping


def safe_load_model(path: Path | str):
    import joblib

    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)
