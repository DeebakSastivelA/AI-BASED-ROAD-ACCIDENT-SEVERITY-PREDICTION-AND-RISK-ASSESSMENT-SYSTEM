from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ml.utils import infer_risk_mapping


def format_feature_name(encoded_name: str) -> str:
    if "__" not in encoded_name:
        return encoded_name
    feature_name, value = encoded_name.split("__", 1)
    return f"{feature_name} = {value}"


def explain_model_prediction(model_pipeline, input_row: pd.Series, target_mapping: Dict[Any, str]) -> Dict[str, Any]:
    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]
    encoded_features = preprocessor.get_feature_names_out()
    transformed = preprocessor.transform(pd.DataFrame([input_row]))

    prediction = model_pipeline.predict(pd.DataFrame([input_row]))[0]
    probabilities = model_pipeline.predict_proba(pd.DataFrame([input_row]))[0]
    predicted_index = int(np.argmax(probabilities))
    class_order = model.classes_

    if hasattr(model, "coef_"):
        coefficients = model.coef_[predicted_index]
        score_pairs = sorted(
            [(abs(coeff), encoded_name) for coeff, encoded_name in zip(coefficients, encoded_features)],
            reverse=True,
        )[:8]
        top_features = [
            {"feature": format_feature_name(encoded_name), "weight": float(abs(coeff))}
            for coeff, encoded_name in score_pairs
        ]
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        score_pairs = sorted(
            [(float(importance), encoded_name) for importance, encoded_name in zip(importances, encoded_features)],
            reverse=True,
        )[:8]
        top_features = [
            {"feature": format_feature_name(encoded_name), "weight": float(importance)}
            for importance, encoded_name in score_pairs
        ]
    else:
        top_features = []

    risk_label = target_mapping.get(prediction, "Unknown")
    explanation_lines = [
        f"The selected conditions are most consistent with {risk_label} severity.",
    ]

    for entry in top_features[:5]:
        feature_name = entry["feature"]
        if feature_name.lower().startswith("light_conditions") or "Light_conditions" in feature_name:
            explanation_lines.append(f"{feature_name} is a strong contributor to the current risk profile.")
        elif feature_name.lower().startswith("weather_conditions") or "Weather_conditions" in feature_name:
            explanation_lines.append(f"{feature_name} shifts the prediction toward more hazardous conditions.")
        elif feature_name.lower().startswith("types_of_junction") or "Types_of_Junction" in feature_name:
            explanation_lines.append(f"{feature_name} suggests a junction-related risk factor.")
        elif feature_name.lower().startswith("type_of_collision") or "Type_of_collision" in feature_name:
            explanation_lines.append(f"{feature_name} is important in explaining the severity estimate.")
        elif feature_name.lower().startswith("driving_experience") or "Driving_experience" in feature_name:
            explanation_lines.append(f"{feature_name} is associated with the current driving risk pattern.")
        else:
            explanation_lines.append(f"{feature_name} contributes to the predicted severity for this case.")

    return {
        "prediction": int(prediction),
        "prediction_proba": float(probabilities[predicted_index]),
        "top_features": top_features,
        "explanation_lines": explanation_lines[:5],
        "class_labels": [str(label) for label in class_order],
        "class_probabilities": {str(label): float(probability) for label, probability in zip(class_order, probabilities)},
    }


def generate_safety_recommendation(input_row: pd.Series) -> str:
    selected = input_row.to_dict()
    light = str(selected.get("Light_conditions", "")).lower()
    weather = str(selected.get("Weather_conditions", "")).lower()
    junction = str(selected.get("Types_of_Junction", "")).lower()
    collision = str(selected.get("Type_of_collision", "")).lower()
    pedestrian = str(selected.get("Pedestrian_movement", "")).lower()
    cause = str(selected.get("Cause_of_accident", "")).lower()

    if "darkness" in light or "lights unlit" in light or "no lighting" in light:
        return "Reduce speed and increase following distance in low-visibility conditions. Use headlights appropriately and stay highly alert for pedestrians and cyclists."
    if "rain" in weather or "fog" in weather or "snow" in weather or "wind" in weather:
        return "Drive at reduced speed and maintain additional braking distance in poor weather. Increase caution around junctions and turning vehicles."
    if "junction" in junction or "crossing" in junction or "shape" in junction:
        return "Approach intersections cautiously, check both directions, and yield to crossing vehicles and pedestrians before proceeding."
    if "pedestrian" in collision or "pedestrian" in pedestrian:
        return "Keep a safe speed and remain extra alert for unexpected pedestrian movement, especially near crossings and vehicle stops."
    if "changing lane" in cause or "overtaking" in cause or "moving backward" in cause:
        return "Avoid risky lane changes and overtaking when visibility or space is limited. Check mirrors and blind spots carefully before maneuvering."
    if "speed" in cause or "driving carelessly" in cause or "no distancing" in cause:
        return "Stay within the speed limit, keep safe following distance, and avoid aggressive driving to reduce accident severity."
    return "Maintain a defensive driving style, keep a safe following distance, and stay attentive to road, weather, and pedestrian activity."
