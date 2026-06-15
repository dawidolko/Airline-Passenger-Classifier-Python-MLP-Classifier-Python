"""
Data preprocessing module for Airline Passenger Satisfaction.
Responsible for cleaning, removing unrealistic values, the stratified split,
building the pipeline (imputation + scaling + one-hot) and transforming features.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from airline_project.config import (
    CATEGORICAL_COLUMNS,
    DROP_COLUMNS,
    NUMERIC_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    SERVICE_SCORE_COLUMNS,
)


def ustaw_seedy(seed: int = RANDOM_STATE) -> None:
    """
    Sets the random seed for the random and numpy modules.
    Ensures reproducibility of data splits and weight initialization.
    """
    random.seed(seed)
    np.random.seed(seed)


def remove_unrealistic_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces illogical values with NaN according to domain thresholds:
    - Age: outside the range [0, 100]
    - Flight Distance: <= 0
    - Delays (Departure/Arrival): < 0 or > 1440 min (24h)
    - Service scores (14 columns): outside the range [0, 5]
    """
    output = df.copy()

    if "Age" in output.columns:
        age = pd.to_numeric(output["Age"], errors="coerce")
        output.loc[(age < 0) | (age > 100), "Age"] = np.nan

    if "Flight Distance" in output.columns:
        distance = pd.to_numeric(output["Flight Distance"], errors="coerce")
        output.loc[distance <= 0, "Flight Distance"] = np.nan

    for delay_col in ["Departure Delay in Minutes", "Arrival Delay in Minutes"]:
        if delay_col in output.columns:
            delay = pd.to_numeric(output[delay_col], errors="coerce")
            output.loc[(delay < 0) | (delay > 1440), delay_col] = np.nan

    for score_col in SERVICE_SCORE_COLUMNS:
        if score_col in output.columns:
            score = pd.to_numeric(output[score_col], errors="coerce")
            output.loc[(score < 0) | (score > 5), score_col] = np.nan

    return output


def raportuj_braki_i_odstajace(df: pd.DataFrame, etap: str) -> None:
    """
    Prints diagnostics of missing data (NaN) and numeric statistics.
    Called before and after cleaning so the effect of preprocessing is visible.
    """
    n_rows = len(df)
    n_nan_total = int(df.isna().sum().sum())
    n_rows_with_nan = int(df.isna().any(axis=1).sum())
    pct_rows_nan = n_rows_with_nan / n_rows * 100 if n_rows > 0 else 0

    print(f"\n--- Data diagnostics [{etap}] ---")
    print(f"  Number of rows: {n_rows}")
    print(f"  Total number of NaN across the dataset: {n_nan_total}")
    print(f"  Rows with at least 1 missing value: {n_rows_with_nan} ({pct_rows_nan:.2f}%)")

    cols_with_nan = df.columns[df.isna().any()].tolist()
    if cols_with_nan:
        print(f"  Columns with missing values ({len(cols_with_nan)}):")
        for col in cols_with_nan:
            n = int(df[col].isna().sum())
            print(f"    • {col}: {n} NaN ({n / n_rows * 100:.2f}%)")
    else:
        print("  No columns with NaN values.")
    print()


def przygotuj_surowe_dane(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw DataFrame in the following steps:
    1. Drops unnecessary columns (id, index, satisfaction)
    2. Casts numeric columns (errors='coerce' → unparseable → NaN)
    3. Removes outlier / illogical values (remove_unrealistic_values)
    4. Reports how many missing values/NaN remain after cleaning
    """
    output = df.copy()
    output.columns = output.columns.str.strip()

    for col in DROP_COLUMNS:
        if col in output.columns:
            output = output.drop(columns=[col])

    raportuj_braki_i_odstajace(output, "BEFORE cleaning")

    for col in NUMERIC_COLUMNS:
        if col in output.columns:
            output[col] = pd.to_numeric(output[col], errors="coerce")

    output = remove_unrealistic_values(output)

    raportuj_braki_i_odstajace(output, "AFTER removing outlier values")

    return output


def zakoduj_target(series: pd.Series, label_to_index: dict[str, int]) -> np.ndarray:
    """
    Maps the target's text labels to numeric indices (0, 1, 2).
    """
    return series.map(label_to_index).astype(int).to_numpy()


def podziel_dane_stratyfikowane(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates a train (70%) / validation (15%) / test (15%) split
    preserving class proportions (stratification on TARGET_COLUMN).
    """
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - TRAIN_RATIO),
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )
    val_ratio_in_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_ratio_in_temp),
        random_state=RANDOM_STATE,
        stratify=temp_df[TARGET_COLUMN],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


@dataclass
class PreprocessingArtifacts:
    """Holds the fitted transformer and the list of feature names after transformation."""
    transformer: ColumnTransformer
    feature_names: list[str]


def fit_preprocessor(train_df: pd.DataFrame) -> PreprocessingArtifacts:
    """
    Builds and fits the preprocessing pipeline exclusively on the training set:
    - Numeric: SimpleImputer(median) → StandardScaler
    - Categorical: SimpleImputer(mode) → OneHotEncoder
    Returns PreprocessingArtifacts with the fitted transformer and the feature list.

    IMPORTANT: imputation and scaling are fitted ONLY on train — this prevents data leakage.
    Missing values (NaN) left after remove_unrealistic_values are imputed here with the median/mode.
    """
    features_df = train_df.drop(columns=[TARGET_COLUMN])
    n_nan_before = int(features_df.isna().sum().sum())
    print(f"  Imputing missing values in train: {n_nan_before} NaN to fill "
          f"(median for numeric, mode for categorical)")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, [c for c in NUMERIC_COLUMNS if c in train_df.columns]),
            ("cat", categorical_pipeline, [c for c in CATEGORICAL_COLUMNS if c in train_df.columns]),
        ],
        remainder="drop",
    )

    features = train_df.drop(columns=[TARGET_COLUMN])
    transformer.fit(features)

    numeric_cols = [c for c in NUMERIC_COLUMNS if c in train_df.columns]
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in train_df.columns]
    cat_features = list(transformer.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_cols))
    all_features = numeric_cols + cat_features

    return PreprocessingArtifacts(transformer=transformer, feature_names=all_features)


def transformuj_cechy(df: pd.DataFrame, artifacts: PreprocessingArtifacts) -> np.ndarray:
    """
    Transforms the DataFrame into a float32 numeric matrix
    using the previously fitted preprocessor (without refitting).
    """
    features = df.drop(columns=[TARGET_COLUMN])
    matrix = artifacts.transformer.transform(features)
    return matrix.astype(np.float32)
