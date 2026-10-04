from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ml.data_analysis import generate_eda_charts
from ml.evaluate_models import evaluate_model, save_confusion_matrix, select_best_model
from ml.preprocessing import TARGET_COLUMN, clean_dataset, identify_feature_columns, make_model_pipeline
from ml.utils import DATA_PATH, MODELS_DIR, MODEL_EVAL_DIR, ensure_project_dirs, infer_risk_mapping, load_dataset, save_json


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return make_json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def train_and_evaluate_models() -> dict:
    ensure_project_dirs()
    raw_df = load_dataset(DATA_PATH)
    cleaned_df = clean_dataset(raw_df)
    generate_eda_charts(cleaned_df)

    feature_columns = identify_feature_columns(cleaned_df)
    X = cleaned_df[feature_columns]
    y = cleaned_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    logistic_model = LogisticRegression(
        max_iter=500,
        solver="lbfgs",
        C=1.0,
        class_weight="balanced",
        random_state=42,
    )
    random_forest_model = RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
    )

    logistic_pipeline = make_model_pipeline(logistic_model, X_train)
    random_forest_pipeline = make_model_pipeline(random_forest_model, X_train)

    logistic_pipeline.fit(X_train, y_train)
    random_forest_pipeline.fit(X_train, y_train)

    model_results = {
        "logistic_regression": evaluate_model(logistic_pipeline, X_test, y_test),
        "random_forest": evaluate_model(random_forest_pipeline, X_test, y_test),
    }

    save_confusion_matrix(
        model_results["logistic_regression"]["confusion_matrix"],
        model_results["logistic_regression"]["classes"],
        str(MODEL_EVAL_DIR / "logistic_confusion_matrix.png"),
        "Logistic Regression Confusion Matrix",
    )
    save_confusion_matrix(
        model_results["random_forest"]["confusion_matrix"],
        model_results["random_forest"]["classes"],
        str(MODEL_EVAL_DIR / "random_forest_confusion_matrix.png"),
        "Random Forest Confusion Matrix",
    )

    best_model_name = select_best_model(model_results)
    best_model = logistic_pipeline if best_model_name == "logistic_regression" else random_forest_pipeline

    joblib.dump(logistic_pipeline, MODELS_DIR / "logistic_regression_pipeline.joblib")
    joblib.dump(random_forest_pipeline, MODELS_DIR / "random_forest_pipeline.joblib")
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")

    risk_mapping = infer_risk_mapping(y)

    metadata = {
        "selected_model": best_model_name,
        "classes": sorted(y.unique().tolist()),
        "feature_names": feature_columns,
        "risk_mapping": {str(key): value for key, value in risk_mapping.items()},
        "metrics": make_json_safe(model_results[best_model_name]),
        "all_model_metrics": {
            "logistic_regression": make_json_safe(model_results["logistic_regression"]),
            "random_forest": make_json_safe(model_results["random_forest"]),
        },
        "model_parameters": {
            "logistic_regression": {
                "max_iter": 500,
                "C": 1.0,
                "solver": "lbfgs",
                "class_weight": "balanced",
                "random_state": 42,
            },
            "random_forest": {
                "n_estimators": 250,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "class_weight": "balanced",
                "random_state": 42,
            },
        },
        "training_info": {
            "dataset_path": str(DATA_PATH),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "target_distribution": y.value_counts().sort_index().to_dict(),
            "random_state": 42,
            "test_size": 0.2,
            "stratified": True,
        },
    }

    save_json(metadata, MODELS_DIR / "model_metadata.json")
    return metadata


if __name__ == "__main__":
    result = train_and_evaluate_models()
    print("Training complete.")
    print(f"Best model selected: {result['selected_model']}")
    print(f"Best Macro F1: {result['metrics']['macro_f1']:.4f}")
