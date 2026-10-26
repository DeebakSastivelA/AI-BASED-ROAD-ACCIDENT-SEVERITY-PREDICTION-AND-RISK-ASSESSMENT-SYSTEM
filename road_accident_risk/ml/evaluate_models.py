from __future__ import annotations

from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from ml.utils import SEVERITY_MAP


def evaluate_model(model_pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    predictions = model_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
    recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
    macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
    macro_precision = precision_score(y_test, predictions, average="macro", zero_division=0)
    macro_recall = recall_score(y_test, predictions, average="macro", zero_division=0)
    classes = sorted(y_test.unique().tolist())
    confusion = confusion_matrix(y_test, predictions, labels=classes)

    class_report = {}
    for class_label in classes:
        class_mask = y_test == class_label
        if not class_mask.any():
            class_report[str(class_label)] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
            continue
        true_positive = np.sum((predictions == class_label) & class_mask)
        precision_value = true_positive / max(np.sum(predictions == class_label), 1)
        recall_value = true_positive / max(np.sum(class_mask), 1)
        f1_value = (2 * precision_value * recall_value) / max(precision_value + recall_value, 1e-12)
        class_report[str(class_label)] = {
            "precision": float(precision_value),
            "recall": float(recall_value),
            "f1": float(f1_value),
            "support": int(class_mask.sum()),
        }

    evaluation = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
        "classes": classes,
        "class_report": class_report,
        "confusion_matrix": confusion,
        "predictions": predictions,
    }
    return evaluation


def save_confusion_matrix(confusion: np.ndarray, labels: List[Any], output_path: str, title: str) -> None:
    plt.figure(figsize=(7, 5))
    plt.imshow(confusion, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()

    tick_labels = [SEVERITY_MAP.get(int(label), str(label)) for label in labels]
    plt.xticks(np.arange(len(labels)), tick_labels, rotation=20)
    plt.yticks(np.arange(len(labels)), tick_labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            value = int(confusion[i, j])
            plt.text(j, i, value, ha="center", va="center", color="white" if value > confusion.max() / 2 else "black")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def select_best_model(model_results: Dict[str, Dict[str, Any]]) -> str:
    ranked_models = sorted(
        model_results.items(),
        key=lambda item: (
            item[1].get("macro_f1", 0.0),
            item[1].get("macro_recall", 0.0),
            item[1].get("f1", 0.0),
            item[1].get("accuracy", 0.0),
        ),
        reverse=True,
    )
    return ranked_models[0][0]
