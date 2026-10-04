from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from ml.explain_predictions import explain_model_prediction, generate_safety_recommendation
from ml.preprocessing import TARGET_COLUMN, clean_dataset, get_feature_options
from ml.utils import DATA_PATH, MODELS_DIR, infer_risk_mapping, load_dataset, load_json, safe_load_model


st.set_page_config(page_title="Road Accident Risk AI", page_icon="🚗", layout="wide")


@st.cache_data
def load_clean_dataset() -> pd.DataFrame:
    df = load_dataset(DATA_PATH)
    return clean_dataset(df)


@st.cache_resource
def load_best_model_and_metadata():
    model_path = MODELS_DIR / "best_model.joblib"
    metadata_path = MODELS_DIR / "model_metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("The trained model files are missing. Run 'python ml/train_models.py' first.")
    model = safe_load_model(model_path)
    metadata = load_json(metadata_path)
    return model, metadata


@st.cache_data
def get_feature_choices() -> Dict[str, List[str]]:
    df = load_clean_dataset()
    return get_feature_options(df)


def display_metric_card(title: str, value: str, accent: str = "#4f8bf9") -> None:
    st.markdown(
        f"""
        <div style="padding: 1.2rem; border-radius: 12px; background: linear-gradient(135deg, {accent}, #0f172a); box-shadow: 0 10px 30px rgba(0,0,0,0.15); margin-bottom: 1rem;">
            <div style="font-size: 0.8rem; color: rgba(255,255,255,0.75); text-transform: uppercase; letter-spacing: 0.08em;">{title}</div>
            <div style="font-size: 2rem; font-weight: 700; color: white; margin-top: 0.5rem;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def default_feature_values(options: Dict[str, List[str]]) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    for column, values in options.items():
        if values:
            defaults[column] = values[0]
    return defaults


def build_prediction_frame(form_inputs: Dict[str, Any], feature_columns: List[str]) -> pd.DataFrame:
    record = {column: form_inputs.get(column, "Unknown") for column in feature_columns}
    return pd.DataFrame([record], columns=feature_columns)


def home_page() -> None:
    df = load_clean_dataset()
    model, metadata = load_best_model_and_metadata()
    feature_count = len(df.columns) - 1
    best_model_name = metadata.get("selected_model", "N/A")
    best_metrics = metadata.get("all_model_metrics", {}).get(best_model_name, {})
    macro_f1 = best_metrics.get("macro_f1", 0.0)

    st.title("AI-Based Road Accident Severity Prediction and Risk Assessment System")
    st.caption("Machine Learning–Powered Accident Severity Analysis")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        display_metric_card("Total Records", f"{len(df):,}")
    with col2:
        display_metric_card("Features", f"{feature_count}")
    with col3:
        display_metric_card("Models", "2")
    with col4:
        display_metric_card("Best Model", str(best_model_name).replace("_", " ").title())
    with col5:
        display_metric_card("Best Macro F1", f"{macro_f1:.3f}")

    st.markdown("""
    This project analyzes real accident data to predict severity patterns, visualize risk drivers, and explain the most influential factors behind each prediction.
    """)

    st.subheader("Workflow")
    st.markdown("DATA → EDA → PREPROCESSING → ML MODELS → EVALUATION → RISK PREDICTION → EXPLANATION")

    if st.button("Predict Accident Risk", use_container_width=True):
        st.session_state.page = "Predict Risk"
        st.rerun()


def eda_page() -> None:
    df = load_clean_dataset()
    st.title("Exploratory Data Analysis")

    st.subheader("Dataset Overview")
    overview_cols = st.columns(6)
    with overview_cols[0]:
        st.metric("Rows", f"{df.shape[0]:,}")
    with overview_cols[1]:
        st.metric("Columns", f"{df.shape[1]}")
    with overview_cols[2]:
        st.metric("Missing Values", f"{df.isna().sum().sum()}")
    with overview_cols[3]:
        st.metric("Duplicate Rows", f"{df.duplicated().sum()}")
    with overview_cols[4]:
        st.metric("Target Classes", str(sorted(df[TARGET_COLUMN].unique().tolist())))
    with overview_cols[5]:
        st.metric("Target Mean", f"{df[TARGET_COLUMN].mean():.2f}")

    st.write("### Shape")
    st.write(df.shape)
    st.write("### Columns")
    st.dataframe(pd.DataFrame({"Column": df.columns, "Data Type": df.dtypes.astype(str).values}), use_container_width=True)
    st.write("### Missing Values")
    st.dataframe(df.isna().sum().reset_index().rename(columns={"index": "Column", 0: "Missing Count"}), use_container_width=True)
    st.write("### Duplicate Rows")
    st.write(f"{df.duplicated().sum()} duplicate records found")

    st.subheader("Target Distribution")
    target_counts = df[TARGET_COLUMN].value_counts().sort_index()
    target_percent = target_counts / target_counts.sum() * 100
    summary_df = pd.DataFrame({"Count": target_counts, "Percentage": target_percent})
    st.dataframe(summary_df, use_container_width=True)
    st.bar_chart(summary_df["Count"])

    st.subheader("Categorical Feature Analysis")
    feature_options = [column for column in df.columns if column != TARGET_COLUMN]
    selected_feature = st.selectbox("Select a feature for category analysis", feature_options)
    count_df = df[selected_feature].value_counts().reset_index()
    count_df.columns = [selected_feature, "Count"]
    st.dataframe(count_df, use_container_width=True)
    severity_table = pd.crosstab(df[selected_feature], df[TARGET_COLUMN])
    st.bar_chart(severity_table)

    st.subheader("Important Visualizations")
    chart_files = [
        "target_distribution.png",
        "weather_conditions_vs_severity.png",
        "light_conditions_vs_severity.png",
        "types_of_junction_vs_severity.png",
        "road_surface_type_vs_severity.png",
        "type_of_collision_vs_severity.png",
        "driving_experience_vs_severity.png",
        "cause_of_accident_vs_severity.png",
        "vehicle_movement_vs_severity.png",
    ]
    for file_name in chart_files:
        path = MODELS_DIR.parent / "reports" / "eda" / file_name
        if path.exists():
            st.image(str(path), use_container_width=True)

    st.subheader("Data Quality Checks")
    st.write("- Missing values inspected and normalized.")
    st.write("- Duplicate rows removed.")
    st.write("- Empty strings and whitespace standardized.")
    st.write("- Inconsistent category labels normalized to a consistent format.")
    st.write("- Invalid target values were converted to NaN and removed before modeling.")


def dataset_page() -> None:
    df = load_clean_dataset()
    st.title("Dataset")
    st.metric("Total Records", f"{len(df):,}")
    st.metric("Target Column", TARGET_COLUMN)
    st.metric("Number of Classes", f"{df[TARGET_COLUMN].nunique()}")

    search_text = st.text_input("Search the dataset")
    if search_text:
        mask = df.astype(str).apply(lambda col: col.str.contains(search_text, case=False, na=False)).any(axis=1)
        filtered_df = df[mask]
    else:
        filtered_df = df

    st.dataframe(filtered_df, use_container_width=True, height=500)

    st.subheader("Dataset Statistics")
    stats = pd.DataFrame({
        "Metric": ["Shape", "Missing Values", "Duplicate Count", "Class Distribution"],
        "Value": [
            str(df.shape),
            str(df.isna().sum().sum()),
            str(df.duplicated().sum()),
            str(df[TARGET_COLUMN].value_counts().to_dict()),
        ],
    })
    st.dataframe(stats, use_container_width=True)


def model_performance_page() -> None:
    _, metadata = load_best_model_and_metadata()
    st.title("Model Performance")

    performance = metadata.get("all_model_metrics", {})
    if not performance:
        st.warning("No model evaluation data has been generated yet. Run the training script first.")
        return

    for model_name, metrics in performance.items():
        st.subheader(model_name.replace("_", " ").title())
        metric_table = pd.DataFrame(
            [
                {"Metric": "Accuracy", "Value": metrics.get("accuracy", 0.0)},
                {"Metric": "Precision", "Value": metrics.get("precision", 0.0)},
                {"Metric": "Recall", "Value": metrics.get("recall", 0.0)},
                {"Metric": "F1", "Value": metrics.get("f1", 0.0)},
                {"Metric": "Macro F1", "Value": metrics.get("macro_f1", 0.0)},
            ]
        )
        st.dataframe(metric_table, use_container_width=True)

        confusion_path = MODELS_DIR.parent / "reports" / "model_evaluation" / (
            "logistic_confusion_matrix.png" if model_name == "logistic_regression" else "random_forest_confusion_matrix.png"
        )
        if confusion_path.exists():
            st.image(str(confusion_path), use_container_width=True)

    st.subheader("Best Model")
    selected_model = metadata.get("selected_model", "unknown")
    best_metrics = performance.get(selected_model, {})
    st.write(f"Best Model: {selected_model.replace('_', ' ').title()}")
    st.write(f"Macro F1: {best_metrics.get('macro_f1', 0.0):.4f}")
    st.write(f"Accuracy: {best_metrics.get('accuracy', 0.0):.4f}")


def about_page() -> None:
    st.title("About")
    st.markdown("""
    ### What is the project?
    This is an AI/ML system for predicting road accident severity based on accident-related conditions and driver behavior patterns.

    ### What ML concepts are used?
    - Classification
    - Exploratory Data Analysis
    - Data preprocessing
    - Categorical encoding
    - Feature engineering
    - Train/test splitting
    - Model comparison
    - Evaluation metrics
    - Feature importance
    - Explainable AI
    - Risk assessment

    ### Models
    - Logistic Regression
    - Random Forest

    ### Technology
    - Python
    - Scikit-learn
    - Pandas
    - NumPy
    - Streamlit
    - Matplotlib
    - Seaborn
    """)


def predict_page() -> None:
    df = load_clean_dataset()
    options = get_feature_choices()
    st.title("Predict Accident Risk")

    input_values = {}
    with st.form("accident_prediction_form"):
        st.subheader("DRIVER INFORMATION")
        driver_cols = st.columns(2)
        with driver_cols[0]:
            input_values["Age_band_of_driver"] = st.selectbox("Age band", options["Age_band_of_driver"])
            input_values["Sex_of_driver"] = st.selectbox("Sex", options["Sex_of_driver"])
            input_values["Educational_level"] = st.selectbox("Educational level", options["Educational_level"])
        with driver_cols[1]:
            input_values["Vehicle_driver_relation"] = st.selectbox("Vehicle-driver relation", options["Vehicle_driver_relation"])
            input_values["Driving_experience"] = st.selectbox("Driving experience", options["Driving_experience"])

        st.subheader("ROAD INFORMATION")
        road_cols = st.columns(3)
        with road_cols[0]:
            input_values["Lanes_or_Medians"] = st.selectbox("Lanes / Medians", options["Lanes_or_Medians"])
        with road_cols[1]:
            input_values["Types_of_Junction"] = st.selectbox("Junction type", options["Types_of_Junction"])
        with road_cols[2]:
            input_values["Road_surface_type"] = st.selectbox("Road surface type", options["Road_surface_type"])

        st.subheader("ENVIRONMENT")
        environment_cols = st.columns(2)
        with environment_cols[0]:
            input_values["Light_conditions"] = st.selectbox("Light conditions", options["Light_conditions"])
        with environment_cols[1]:
            input_values["Weather_conditions"] = st.selectbox("Weather conditions", options["Weather_conditions"])

        st.subheader("ACCIDENT INFORMATION")
        accident_cols = st.columns(2)
        with accident_cols[0]:
            input_values["Type_of_collision"] = st.selectbox("Type of collision", options["Type_of_collision"])
            input_values["Vehicle_movement"] = st.selectbox("Vehicle movement", options["Vehicle_movement"])
        with accident_cols[1]:
            input_values["Pedestrian_movement"] = st.selectbox("Pedestrian movement", options["Pedestrian_movement"])
            input_values["Cause_of_accident"] = st.selectbox("Cause of accident", options["Cause_of_accident"])

        submitted = st.form_submit_button("PREDICT ACCIDENT RISK", use_container_width=True)

    if submitted:
        model, metadata = load_best_model_and_metadata()
        feature_columns = metadata.get("feature_names", [column for column in df.columns if column != TARGET_COLUMN])
        input_frame = build_prediction_frame(input_values, feature_columns)
        prediction = int(model.predict(input_frame)[0])
        probabilities = model.predict_proba(input_frame)[0]
        class_labels = model.classes_
        max_index = int(probabilities.argmax())
        confidence = float(probabilities[max_index] * 100)
        risk_mapping = {int(key): value for key, value in metadata.get("risk_mapping", {}).items()}
        risk_level = risk_mapping.get(prediction, "Unknown")
        predicted_label = risk_level if prediction in risk_mapping else str(prediction)

        st.subheader("PREDICTED ACCIDENT SEVERITY")
        st.markdown(f"<h2>Predicted Severity: <span style='color:#4f8bf9'>{predicted_label}</span></h2>", unsafe_allow_html=True)
        st.markdown(f"<h3>Prediction Confidence: {confidence:.2f}%</h3>", unsafe_allow_html=True)

        probability_df = pd.DataFrame(
            {"Severity Class": [str(label) for label in class_labels], "Probability": [float(prob) for prob in probabilities]}
        )
        probability_df["Probability"] = probability_df["Probability"] * 100
        st.dataframe(probability_df, use_container_width=True)

        explanation = explain_model_prediction(model, input_frame.iloc[0], risk_mapping)
        st.subheader("WHY IS THIS RISK?")
        for line in explanation["explanation_lines"]:
            st.write(f"- {line}")

        st.subheader("SAFETY RECOMMENDATION")
        recommendation = generate_safety_recommendation(input_frame.iloc[0])
        st.info(recommendation)

        st.markdown("### Risk Summary")
        st.write(f"Severity: {predicted_label}")
        st.write(f"Risk Level: {risk_level}")
        st.write(f"Confidence: {confidence:.2f}%")


def main() -> None:
    pages = {
        "Home": home_page,
        "EDA": eda_page,
        "Predict Risk": predict_page,
        "Dataset": dataset_page,
        "Model Performance": model_performance_page,
        "About": about_page,
    }

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Menu", list(pages.keys()))
    pages[selection]()


if __name__ == "__main__":
    main()
