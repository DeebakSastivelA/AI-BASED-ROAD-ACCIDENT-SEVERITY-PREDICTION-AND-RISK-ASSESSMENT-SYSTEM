# AI-Based Road Accident Severity Prediction and Risk Assessment System

## Overview

This project predicts road accident severity using the balanced synthetic dataset `road_accident_balanced_reality_based_18000.csv`, which is the active dataset for training, evaluation, and prediction. The dashboard is built with Streamlit and uses two scikit-learn classifiers: logistic regression and random forest.

The dataset is intentionally synthetic and balanced for academic use. It is not a representation of real-world accident statistics.

## Severity mapping

The target variable is `Accident_severity` with the required mapping:

- 0 = Fatal Injury
- 1 = Serious Injury
- 2 = Slight Injury

The user-facing risk labels are:

- 0 -> Very High
- 1 -> High
- 2 -> Low

## Data source

The primary dataset is located at the workspace root and is loaded automatically by the project:

- `road_accident_balanced_reality_based_18000.csv`

The original `cleaned.csv` file is kept untouched and is not used as the active training dataset.

## Included functionality

- Data cleaning and validation
- Exploratory data analysis charts
- Stratified 80/20 train/test split
- Preprocessing with `OneHotEncoder(handle_unknown="ignore")`
- Training of both logistic regression and random forest models
- Model comparison using macro F1
- Saved model artifacts and metadata
- Streamlit dashboard for home, EDA, dataset, prediction, model performance, and about pages
- Explainable safety recommendations based on the selected scenario

## Project structure

```text
new ml/
├── cleaned.csv
├── requirements.txt
├── road_accident_balanced_reality_based_18000.csv
├── road_accident_risk/
│   ├── app.py
│   ├── README.md
│   ├── requirements.txt
│   ├── assets/
│   ├── data/
│   ├── ml/
│   ├── models/
│   └── reports/
└── .venv/
```

## Training

From the project folder:

```bash
cd road_accident_risk
python ml/train_models.py
```

This generates the trained pipeline files and evaluation artifacts in `models/` and `reports/`.

## Running the app

```bash
cd road_accident_risk
streamlit run app.py
```

The app loads the saved best model and uses the active dataset metadata to generate predictions. It does not force a prediction; the user selects the scenario and the model determines the outcome.

## Notes

- The app uses the model prediction to determine the displayed result.
- The dataset is synthetic and balanced for experimentation.
- All model outputs and chart artifacts are stored under the project’s `models/` and `reports/` folders.
- The app does not overwrite or rename `cleaned.csv`.
