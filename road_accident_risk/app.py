from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from ml.explain_predictions import explain_model_prediction, generate_safety_recommendation
from ml.preprocessing import TARGET_COLUMN, clean_dataset, get_feature_options
from ml.utils import DATA_PATH, MODELS_DIR, infer_risk_mapping, load_dataset, load_json, safe_load_model

SEVERITY_MAPPING = {
    0: {"severity": "Fatal Injury", "risk": "Very High"},
    1: {"severity": "Serious Injury", "risk": "High"},
    2: {"severity": "Slight Injury", "risk": "Low"},
}


st.set_page_config(page_title="Road Accident Risk AI", page_icon="🚗", layout="wide")

st.markdown(
    """
    <style>
        :root {
            --bg: #081223;
            --bg-soft: #111827;
            --panel: rgba(15, 23, 42, 0.88);
            --panel-soft: rgba(17, 24, 39, 0.72);
            --border: rgba(148, 163, 184, 0.2);
            --text: #e5eefb;
            --muted: #a8bbd8;
            --primary: #5bb4ff;
            --green: #39d98a;
            --green-soft: rgba(57, 217, 138, 0.12);
            --amber: #fbbf24;
            --amber-soft: rgba(251, 191, 36, 0.12);
            --red: #ef4444;
            --red-soft: rgba(239, 68, 68, 0.12);
            --dark-red: #7f1d1d;
            --shadow: 0 16px 40px rgba(15, 23, 42, 0.45);
        }

        .stApp {
            background: linear-gradient(180deg, #081223 0%, #0d1729 100%);
            color: var(--text);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(9,16,29,1) 0%, rgba(21,30,49,0.96) 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.18);
            box-shadow: 10px 0 30px rgba(0, 0, 0, 0.22);
            padding: 0.8rem 0.9rem 1.2rem 0.9rem;
        }

        [data-testid="stSidebar"] > div {
            background: rgba(15, 23, 42, 0.38);
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.08);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p {
            color: #edf6ff !important;
        }

        [data-testid="stSidebar"] .css-1d391kg {
            font-size: 0.8rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: rgba(191, 219, 254, 0.9);
            margin: 0.5rem 0 1rem 0;
            padding-left: 0.2rem;
        }

        [data-testid="stSidebar"] [role="radiogroup"] {
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
            margin-top: 0.35rem;
        }

        [data-testid="stSidebar"] label {
            border-radius: 12px;
            padding: 0.7rem 0.8rem;
            transition: all 0.22s ease;
            border: 1px solid transparent;
            margin: 0;
        }

        [data-testid="stSidebar"] label:hover {
            background: rgba(91, 180, 255, 0.08);
            border-color: rgba(91, 180, 255, 0.16);
            transform: translateX(2px);
        }

        [data-testid="stSidebar"] label:has(input:checked) {
            background: linear-gradient(90deg, rgba(91, 180, 255, 0.22), rgba(59, 130, 246, 0.08));
            border: 1px solid rgba(96, 165, 250, 0.45);
            box-shadow: 0 0 0 1px rgba(96, 165, 250, 0.2), 0 8px 18px rgba(59, 130, 246, 0.18);
        }

        [data-testid="stSidebar"] .stRadio > div {
            gap: 0.5rem;
        }

        [data-testid="stSidebar"] .stRadio label p {
            margin: 0;
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--text);
        }

        div[data-testid="stMainBlockContainer"] {
            padding-top: 1.25rem;
        }

        div.stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            width: 100%;
            border: 1px solid rgba(96, 165, 250, 0.55);
            border-radius: 14px;
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.45), rgba(37, 99, 235, 0.82));
            color: #edf6ff;
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            padding: 0.8rem 1.1rem;
            box-shadow: 0 12px 24px rgba(59, 130, 246, 0.18);
            transition: all 0.2s ease;
        }

        div.stButton > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            border-color: rgba(191, 219, 254, 0.8);
            background: linear-gradient(135deg, rgba(96, 165, 250, 0.55), rgba(29, 78, 216, 0.95));
            box-shadow: 0 14px 28px rgba(59, 130, 246, 0.28);
            transform: translateY(-1px);
        }

        div.stButton > button:focus:not(:active),
        div[data-testid="stFormSubmitButton"] > button:focus:not(:active) {
            box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.2), 0 12px 24px rgba(59, 130, 246, 0.18);
            outline: none;
        }

        .prediction-card {
            border-radius: 18px;
            padding: 1.2rem 1.25rem;
            margin: 0.5rem 0 1rem 0;
            border: 1px solid rgba(148, 163, 184, 0.16);
            box-shadow: 0 18px 38px rgba(15, 23, 42, 0.28);
            transition: all 0.25s ease;
        }

        .risk-low {
            background: linear-gradient(135deg, rgba(57, 217, 138, 0.18), rgba(15, 23, 42, 0.8));
            border-color: rgba(57, 217, 138, 0.4);
            box-shadow: 0 0 0 1px rgba(57, 217, 138, 0.1), 0 18px 38px rgba(57, 217, 138, 0.12);
        }

        .risk-medium {
            background: linear-gradient(135deg, rgba(251, 191, 36, 0.16), rgba(15, 23, 42, 0.8));
            border-color: rgba(251, 191, 36, 0.38);
            box-shadow: 0 0 0 1px rgba(251, 191, 36, 0.1), 0 18px 38px rgba(251, 191, 36, 0.12);
        }

        .risk-high {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.18), rgba(15, 23, 42, 0.8));
            border-color: rgba(239, 68, 68, 0.38);
            box-shadow: 0 0 0 1px rgba(239, 68, 68, 0.1), 0 18px 38px rgba(239, 68, 68, 0.12);
        }

        .risk-very-high {
            background: linear-gradient(135deg, rgba(127, 29, 29, 0.82), rgba(15, 23, 42, 0.9));
            border-color: rgba(248, 113, 113, 0.7);
            box-shadow: 0 0 0 1px rgba(248, 113, 113, 0.18), 0 0 24px rgba(239, 68, 68, 0.35);
            animation: riskPulse 1.6s ease-in-out infinite;
        }

        @keyframes riskPulse {
            0%, 100% { box-shadow: 0 0 0 1px rgba(248, 113, 113, 0.18), 0 0 20px rgba(239, 68, 68, 0.22); }
            50% { box-shadow: 0 0 0 1px rgba(248, 113, 113, 0.22), 0 0 30px rgba(239, 68, 68, 0.52); }
        }

        .confidence-card {
            border-radius: 16px;
            padding: 1rem 1.1rem;
            background: linear-gradient(135deg, rgba(91, 180, 255, 0.14), rgba(15, 23, 42, 0.82));
            border: 1px solid rgba(96, 165, 250, 0.22);
            box-shadow: 0 8px 20px rgba(59, 130, 246, 0.08);
            margin-bottom: 1rem;
        }

        .confidence-value {
            font-size: 2rem;
            font-weight: 800;
            color: #dfeeff;
            margin-top: 0.35rem;
        }

        .probability-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0 0.55rem;
            margin-top: 0.4rem;
            margin-bottom: 1.25rem;
        }

        .probability-table th,
        .probability-table td {
            text-align: left;
            padding: 0.8rem 0.9rem;
        }

        .probability-table thead th {
            color: rgba(226, 232, 240, 0.8);
            font-size: 0.74rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            border-bottom: 1px solid rgba(148, 163, 184, 0.18);
        }

        .probability-table tbody tr {
            background: rgba(15, 23, 42, 0.56);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 12px;
        }

        .probability-table tbody td {
            border-top: 1px solid rgba(148, 163, 184, 0.1);
            border-bottom: 1px solid rgba(148, 163, 184, 0.1);
            color: var(--text);
        }

        .risk-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 90px;
            border-radius: 999px;
            padding: 0.35rem 0.8rem;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            border: 1px solid transparent;
        }

        .risk-badge.low { background: rgba(57, 217, 138, 0.14); color: #a7f3d0; border-color: rgba(57, 217, 138, 0.34); }
        .risk-badge.medium { background: rgba(251, 191, 36, 0.14); color: #fde68a; border-color: rgba(251, 191, 36, 0.34); }
        .risk-badge.high { background: rgba(239, 68, 68, 0.16); color: #fecaca; border-color: rgba(239, 68, 68, 0.36); }
        .risk-badge.very-high { background: rgba(127, 29, 29, 0.5); color: #fee2e2; border-color: rgba(248, 113, 113, 0.5); animation: badgePulse 1.5s ease-in-out infinite; }

        @keyframes badgePulse {
            0%, 100% { box-shadow: 0 0 0 rgba(239, 68, 68, 0); }
            50% { box-shadow: 0 0 18px rgba(239, 68, 68, 0.28); }
        }

        .probability-percent {
            font-weight: 700;
            color: #f8fbff;
            text-align: right;
        }

        .explanation-list {
            list-style: none;
            padding: 0;
            margin: 0.5rem 0 0 0;
        }

        .explanation-list li {
            background: rgba(15, 23, 42, 0.46);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 12px;
            padding: 0.8rem 0.9rem;
            margin-bottom: 0.65rem;
            color: #edf6ff;
            line-height: 1.5;
        }

        .explanation-list li strong {
            color: #7dd3fc;
            font-weight: 700;
        }

        .section-card {
            background: rgba(15, 23, 42, 0.42);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }

        @media (max-width: 768px) {
            [data-testid="stSidebar"] {
                padding: 0.5rem 0.5rem 1rem 0.5rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def get_severity_info(class_value: Any) -> Dict[str, str]:
    try:
        class_key = int(class_value)
    except (TypeError, ValueError):
        return {"severity": str(class_value), "risk": "Unknown"}
    return SEVERITY_MAPPING.get(class_key, {"severity": f"Class {class_key}", "risk": "Unknown"})


def get_risk_theme(risk_label: str) -> Dict[str, str]:
    normalized = str(risk_label).strip().lower()
    if "low" in normalized or "slight" in normalized:
        return {"class": "risk-low", "color": "#39d98a", "accent": "#39d98a", "background": "rgba(57, 217, 138, 0.12)"}
    if "high" in normalized or "serious" in normalized:
        return {"class": "risk-high", "color": "#ef4444", "accent": "#ef4444", "background": "rgba(239, 68, 68, 0.12)"}
    if "very high" in normalized or "fatal" in normalized:
        return {"class": "risk-very-high", "color": "#fca5a5", "accent": "#ef4444", "background": "rgba(127, 29, 29, 0.22)"}
    return {"class": "risk-high", "color": "#ef4444", "accent": "#ef4444", "background": "rgba(239, 68, 68, 0.12)"}


def format_explanation_line(line: str) -> str:
    text = str(line).replace("categorical = ", "").strip()
    replacements = {
        "Age_band_of_driver_": "Age band: ",
        "Sex_of_driver_": "Driver sex: ",
        "Educational_level_": "Education level: ",
        "Vehicle_driver_relation_": "Vehicle-driver relation: ",
        "Driving_experience_": "Driving experience: ",
        "Lanes_or_Medians_": "Road layout: ",
        "Types_of_Junction_": "Junction type: ",
        "Road_surface_type_": "Road surface: ",
        "Light_conditions_": "Light conditions: ",
        "Weather_conditions_": "Weather conditions: ",
        "Type_of_collision_": "Collision type: ",
        "Vehicle_movement_": "Vehicle movement: ",
        "Pedestrian_movement_": "Pedestrian movement: ",
        "Cause_of_accident_": "Cause of accident: ",
        "Age_band_of_driver": "Age band: ",
        "Sex_of_driver": "Driver sex: ",
        "Educational_level": "Education level: ",
        "Vehicle_driver_relation": "Vehicle-driver relation: ",
        "Driving_experience": "Driving experience: ",
        "Lanes_or_Medians": "Road layout: ",
        "Types_of_Junction": "Junction type: ",
        "Road_surface_type": "Road surface: ",
        "Light_conditions": "Light conditions: ",
        "Weather_conditions": "Weather conditions: ",
        "Type_of_collision": "Collision type: ",
        "Vehicle_movement": "Vehicle movement: ",
        "Pedestrian_movement": "Pedestrian movement: ",
        "Cause_of_accident": "Cause of accident: ",
    }
    for old, new in replacements.items():
        if old in text:
            text = text.replace(old, new)
    text = text.replace("_", " ")
    return text


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
        st.session_state.navigation_target = "Predict Risk"
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
        predicted_info = get_severity_info(prediction)
        predicted_severity = predicted_info["severity"]
        risk_level = predicted_info["risk"]
        risk_level_display = risk_level.upper()
        risk_theme = get_risk_theme(risk_level)

        st.subheader("PREDICTED ACCIDENT SEVERITY")
        st.markdown(
            f"""
            <div class="prediction-card {risk_theme['class']}">
                <div style="font-size: 0.75rem; color: rgba(255,255,255,0.7); letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.45rem;">Accident Severity</div>
                <div style="font-size: 2rem; font-weight: 800; color: white; line-height: 1.2;">{predicted_severity}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="confidence-card">
                <div style="font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase; color: rgba(191,219,254,0.8);">Risk Level</div>
                <div class="confidence-value" style="color: {risk_theme['color']};">{risk_level_display}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        probability_rows = []
        for class_label, probability in zip(class_labels, probabilities):
            severity_info = get_severity_info(class_label)
            risk_name = severity_info["risk"]
            risk_key = str(risk_name).lower()
            if "very" in risk_key:
                badge_class = "very-high"
            elif "high" in risk_key:
                badge_class = "high"
            else:
                badge_class = "low"
            probability_rows.append(
                {
                    "Accident Severity": severity_info["severity"],
                    "Risk Level": risk_name,
                    "Probability": float(probability) * 100,
                    "Badge Class": badge_class,
                }
            )

        st.markdown(
            f"""
            <div class="section-card">
                <div style="font-size: 0.8rem; letter-spacing: 0.12em; text-transform: uppercase; color: rgba(191,219,254,0.8); margin-bottom: 0.5rem;">Severity Probability</div>
                <table class="probability-table">
                    <thead>
                        <tr>
                            <th>Accident Severity</th>
                            <th>Risk Level</th>
                            <th style="text-align:right;">Probability</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(f"<tr><td>{row['Accident Severity']}</td><td><span class='risk-badge {row['Badge Class']}'>{row['Risk Level']}</span></td><td class='probability-percent'>{row['Probability']:.2f}%</td></tr>" for row in probability_rows)}
                    </tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        severity_mapping = {int(class_label): get_severity_info(class_label)["severity"] for class_label in class_labels}
        explanation = explain_model_prediction(model, input_frame.iloc[0], severity_mapping)
        if explanation.get("explanation_lines"):
            explanation["explanation_lines"][0] = (
                f"The selected conditions are most consistent with {predicted_severity} severity, which corresponds to {risk_level} Risk."
            )
        st.subheader("WHY IS THIS RISK?")
        formatted_lines = [format_explanation_line(line) for line in explanation["explanation_lines"]]
        explanation_html = "".join(f"<li><strong>✓</strong> {line}</li>" for line in formatted_lines)
        st.markdown(f"<ul class='explanation-list'>{explanation_html}</ul>", unsafe_allow_html=True)

        st.subheader("SAFETY RECOMMENDATION")
        recommendation = generate_safety_recommendation(input_frame.iloc[0])
        st.info(recommendation)

        st.markdown("### Risk Summary")
        st.write(f"Severity: {predicted_severity}")
        st.write(f"Risk Level: {risk_level_display}")
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

    if "menu_selection" not in st.session_state:
        st.session_state.menu_selection = "Home"

    page_names = list(pages.keys())
    if "navigation_target" in st.session_state and st.session_state.navigation_target in page_names:
        st.session_state.menu_selection = st.session_state.navigation_target
        st.session_state.navigation_target = ""

    selected_index = page_names.index(st.session_state.menu_selection) if st.session_state.menu_selection in page_names else 0

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Menu", page_names, index=selected_index, key="menu_selection")
    pages[selection]()


if __name__ == "__main__":
    main()
