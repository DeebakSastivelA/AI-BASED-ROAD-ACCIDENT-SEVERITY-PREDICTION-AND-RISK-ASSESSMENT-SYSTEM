# AI-Based Road Accident Severity Prediction and Risk Assessment System

## 1. Project Overview

This project builds a complete machine learning pipeline for road accident severity prediction using the real dataset stored in `data/cleaned.csv`. The system performs data exploration, data quality checks, preprocessing, model training, evaluation, model comparison, and risk prediction using a polished Streamlit dashboard.

## 2. Problem Statement

Road traffic accidents create significant safety, health, and economic costs. Accurate prediction of accident severity helps support decision-making, identify high-risk conditions, and guide preventive safety recommendations.

## 3. Objective

The main objective is to predict accident severity from real accident-condition features and present the results through an explainable, user-friendly Streamlit interface.

## 4. Dataset

The dataset is loaded from `data/cleaned.csv` and contains accident records with the following key columns:

- Age_band_of_driver
- Sex_of_driver
- Educational_level
- Vehicle_driver_relation
- Driving_experience
- Lanes_or_Medians
- Types_of_Junction
- Road_surface_type
- Light_conditions
- Weather_conditions
- Type_of_collision
- Vehicle_movement
- Pedestrian_movement
- Cause_of_accident
- Accident_severity

The target column is `Accident_severity` and is not used as an input feature during model training.

## 5. EDA

The project includes an end-to-end exploratory data analysis phase that examines:

- dataset shape and structure
- missing values
- duplicate rows
- target distribution
- category distribution by feature
- feature relationship with severity
- saved charts under `reports/eda/`

## 6. Preprocessing

Preprocessing is applied carefully to avoid data leakage:

- missing values are inspected and normalized
- empty strings and whitespace are cleaned
- invalid or inconsistent category values are standardized
- duplicate records are removed
- the target column is checked and cleaned
- categorical features are encoded with `OneHotEncoder(handle_unknown="ignore")`
- numerical features are imputed and scaled when present
- train/test split uses 80/20 with random_state=42 and stratification

## 7. Machine Learning Models

### Logistic Regression

A logistic regression classifier is trained using a scikit-learn pipeline with preprocessing and class balancing support. This model is useful for interpretability and relatively stable performance on structured tabular data.

### Random Forest

A random forest classifier is trained with class balancing and a scikit-learn preprocessing pipeline. This model is evaluated alongside logistic regression for comparison.

## 8. Evaluation Metrics

The trained models are evaluated with:

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1
- Weighted F1
- confusion matrix
- per-class precision, recall, and F1

These values are computed from the actual model predictions on the untouched test set.

## 9. Model Selection

The best model is selected programmatically based on actual evaluation results, prioritizing:

1. Macro F1
2. Per-class recall
3. F1
4. Accuracy

The winning pipeline is saved as `models/best_model.joblib` and the metadata is stored in `models/model_metadata.json`.

## 10. Explainable Prediction

The prediction interface explains the model output with the most influential encoded features and the reason the model likely selected the predicted severity. This explanation is generated from the trained model and the user input and is presented under the "Why is this risk?" section.

## 11. Streamlit Application

The app contains the following sections:

- Home
- EDA
- Predict Risk
- Dataset
- Model Performance
- About

The dashboard provides a polished and presentation-friendly interface for exploring data, comparing models, and predicting accident risk.

## 12. Installation

Create a virtual environment and install dependencies:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Then install the project dependencies:

```bash
pip install -r requirements.txt
```

## 13. Training and Running

Train the models:

```bash
python ml/train_models.py
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## 14. Project Structure

```text
road_accident_risk/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   └── cleaned.csv
├── ml/
│   ├── __init__.py
│   ├── data_analysis.py
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── explain_predictions.py
│   └── utils.py
├── models/
│   ├── logistic_regression_pipeline.joblib
│   ├── random_forest_pipeline.joblib
│   ├── best_model.joblib
│   └── model_metadata.json
├── reports/
│   ├── eda/
│   └── model_evaluation/
├── assets/
│   └── images/
└── .gitignore
```

## 15. Future Improvements

Possible future enhancements include:

- more accident datasets from multiple regions
- geographic risk mapping
- real-time traffic and weather integration
- additional explainable AI methods
- more advanced ML models and ensemble methods

## 16. Final Notes

This project is designed to demonstrate the complete machine learning lifecycle using real data, transparent preprocessing, objective model comparison, and user-friendly risk assessment.
