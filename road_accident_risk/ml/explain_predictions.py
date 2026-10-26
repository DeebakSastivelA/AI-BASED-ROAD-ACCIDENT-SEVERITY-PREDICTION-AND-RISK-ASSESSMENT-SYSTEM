from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def format_feature_name(encoded_name: str) -> str:
    clean_name = str(encoded_name)
    clean_name = clean_name.replace("categorical__", "")
    clean_name = clean_name.replace("numeric__", "")
    clean_name = clean_name.replace("remainder__", "")
    clean_name = clean_name.replace("__", " - ")
    clean_name = clean_name.replace("_", " ")
    clean_name = clean_name.strip()
    if clean_name == "":
        return "Field value"
    return clean_name


def explain_model_prediction(model_pipeline, input_row: pd.Series, target_mapping: Dict[Any, str]) -> Dict[str, Any]:
    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]
    encoded_features = preprocessor.get_feature_names_out()
    input_frame = pd.DataFrame([input_row])

    prediction = model_pipeline.predict(input_frame)[0]
    probabilities = model_pipeline.predict_proba(input_frame)[0]
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
    explanation_lines = [f"The selected conditions are most consistent with {risk_label} severity."]

    for entry in top_features[:5]:
        feature_name = entry["feature"]
        readable = feature_name.lower()
        if "light" in readable:
            explanation_lines.append(f"{feature_name} increased the probability of a severe outcome in this scenario.")
        elif "weather" in readable:
            explanation_lines.append(f"{feature_name} shifted the risk estimate toward a more hazardous condition.")
        elif "junction" in readable:
            explanation_lines.append(f"{feature_name} contributed to the predicted severity level for this road context.")
        elif "collision" in readable or "movement" in readable:
            explanation_lines.append(f"{feature_name} was a strong driver of the predicted accident severity.")
        elif "driving" in readable or "experience" in readable:
            explanation_lines.append(f"{feature_name} affected the risk profile for this driver profile.")
        else:
            explanation_lines.append(f"{feature_name} contributed to the predicted severity for this case.")

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

    if "darkness" in light or "no lighting" in light or "dark" in light:
        return "Reduce speed and increase following distance in low-visibility conditions. Use headlights appropriately and stay highly alert for pedestrians and cyclists."
    if any(token in weather for token in ["rain", "fog", "snow", "wind"]):
        return "Drive at reduced speed and maintain additional braking distance in poor weather. Increase caution around junctions and turning vehicles."
    if any(token in junction for token in ["junction", "crossing", "roundabout", "T", "Y"]):
        return "Approach intersections cautiously, check both directions, and yield to crossing vehicles and pedestrians before proceeding."
    if "pedestrian" in collision or "pedestrian" in pedestrian:
        return "Keep a safe speed and remain extra alert for unexpected pedestrian movement, especially near crossings and vehicle stops."
    if any(token in cause for token in ["changing lane", "overtaking", "reverse", "speed violation", "careless"]):
        return "Avoid risky lane changes and overtaking when visibility or space is limited. Check mirrors and blind spots carefully before maneuvering."
    return "Maintain a defensive driving style, keep a safe following distance, and stay attentive to road, weather, and pedestrian activity."
