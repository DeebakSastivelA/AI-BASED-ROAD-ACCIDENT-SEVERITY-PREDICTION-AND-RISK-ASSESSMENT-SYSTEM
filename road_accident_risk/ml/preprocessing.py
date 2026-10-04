from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.utils import normalize_text


TARGET_COLUMN = "Accident_severity"


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    for column in cleaned.columns:
        if column == TARGET_COLUMN:
            cleaned[column] = pd.to_numeric(cleaned[column].astype(str).str.strip(), errors="coerce")
        else:
            cleaned[column] = cleaned[column].map(normalize_text)
            cleaned[column] = cleaned[column].replace({"Unknown": "Unknown"})

    cleaned = cleaned.dropna(subset=[TARGET_COLUMN]).copy()
    cleaned = cleaned.drop_duplicates().copy()
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)
    return cleaned


def identify_feature_columns(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> List[str]:
    return [column for column in df.columns if column != target_column]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()
    numerical_columns = [column for column in X.columns if column not in categorical_columns]

    transformers: List[Tuple[str, Pipeline, List[str]]] = []

    if numerical_columns:
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", num_pipeline, numerical_columns))

    if categorical_columns:
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", cat_pipeline, categorical_columns))

    if not transformers:
        raise ValueError("No feature columns available for preprocessing.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_model_pipeline(model, X: pd.DataFrame) -> Pipeline:
    preprocessor = build_preprocessor(X)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def get_feature_options(df: pd.DataFrame) -> Dict[str, List[str]]:
    feature_columns = identify_feature_columns(df)
    options: Dict[str, List[str]] = {}
    for column in feature_columns:
        values = sorted(df[column].dropna().unique().tolist())
        options[column] = values
    return options
