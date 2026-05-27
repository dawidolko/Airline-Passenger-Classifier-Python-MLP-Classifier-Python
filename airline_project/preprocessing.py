"""
Moduł preprocessingu danych Airline Passenger Satisfaction.
Odpowiada za czyszczenie, usuwanie wartości nierealnych, podział stratyfikowany,
budowę pipeline'u (imputacja + skalowanie + one-hot) oraz transformację cech.
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
    Ustawia ziarno losowości dla modułów random i numpy.
    Zapewnia powtarzalność podziałów danych i inicjalizacji wag.
    """
    random.seed(seed)
    np.random.seed(seed)


def remove_unrealistic_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zastępuje wartości nielogiczne na NaN zgodnie z progami dziedzinowymi:
    - Age: poza zakresem [0, 100]
    - Flight Distance: <= 0
    - Opóźnienia (Departure/Arrival): < 0 lub > 1440 min (24h)
    - Oceny usług (14 kolumn): poza zakresem [0, 5]
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
    Wypisuje diagnostykę braków danych (NaN) i statystyki numeryczne.
    Wywoływana przed i po czyszczeniu, żeby było widać efekt preprocessingu.
    """
    n_rows = len(df)
    n_nan_total = int(df.isna().sum().sum())
    n_rows_with_nan = int(df.isna().any(axis=1).sum())
    pct_rows_nan = n_rows_with_nan / n_rows * 100 if n_rows > 0 else 0

    print(f"\n--- Diagnostyka danych [{etap}] ---")
    print(f"  Liczba wierszy: {n_rows}")
    print(f"  Łączna liczba NaN w całym zbiorze: {n_nan_total}")
    print(f"  Wiersze z przynajmniej 1 brakiem: {n_rows_with_nan} ({pct_rows_nan:.2f}%)")

    cols_with_nan = df.columns[df.isna().any()].tolist()
    if cols_with_nan:
        print(f"  Kolumny z brakami ({len(cols_with_nan)}):")
        for col in cols_with_nan:
            n = int(df[col].isna().sum())
            print(f"    • {col}: {n} NaN ({n / n_rows * 100:.2f}%)")
    else:
        print("  Brak kolumn z wartościami NaN.")
    print()


def przygotuj_surowe_dane(df: pd.DataFrame) -> pd.DataFrame:
    """
    Czyści surowy DataFrame w następujących krokach:
    1. Usuwa zbędne kolumny (id, indeks, satisfaction)
    2. Rzutuje kolumny numeryczne (errors='coerce' → nieparsowalne → NaN)
    3. Usuwa wartości odstające/nielogiczne (remove_unrealistic_values)
    4. Raportuje ile braków/NaN powstało po czyszczeniu
    """
    output = df.copy()
    output.columns = output.columns.str.strip()

    for col in DROP_COLUMNS:
        if col in output.columns:
            output = output.drop(columns=[col])

    raportuj_braki_i_odstajace(output, "PRZED czyszczeniem")

    for col in NUMERIC_COLUMNS:
        if col in output.columns:
            output[col] = pd.to_numeric(output[col], errors="coerce")

    output = remove_unrealistic_values(output)

    raportuj_braki_i_odstajace(output, "PO usunięciu wartości odstających")

    return output


def zakoduj_target(series: pd.Series, label_to_index: dict[str, int]) -> np.ndarray:
    """
    Mapuje etykiety tekstowe targetu na indeksy liczbowe (0, 1, 2).
    """
    return series.map(label_to_index).astype(int).to_numpy()


def podziel_dane_stratyfikowane(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Tworzy podział train (70%) / validation (15%) / test (15%)
    z zachowaniem proporcji klas (stratyfikacja po TARGET_COLUMN).
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
    """Przechowuje dopasowany transformer i listę nazw cech po transformacji."""
    transformer: ColumnTransformer
    feature_names: list[str]


def fit_preprocessor(train_df: pd.DataFrame) -> PreprocessingArtifacts:
    """
    Buduje i dopasowuje preprocessing pipeline wyłącznie na zbiorze treningowym:
    - Numeryczne: SimpleImputer(mediana) → StandardScaler
    - Kategoryczne: SimpleImputer(moda) → OneHotEncoder
    Zwraca PreprocessingArtifacts z dopasowanym transformerem i listą cech.

    WAŻNE: imputacja i skalowanie fitowane TYLKO na train — zapobiega data leakage.
    Braki (NaN) po remove_unrealistic_values są tu uzupełniane medianą/modą.
    """
    features_df = train_df.drop(columns=[TARGET_COLUMN])
    n_nan_before = int(features_df.isna().sum().sum())
    print(f"  Imputacja braków w train: {n_nan_before} NaN do uzupełnienia "
          f"(mediana dla numerycznych, moda dla kategorycznych)")

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
    Transformuje DataFrame do macierzy numerycznej float32
    używając dopasowanego wcześniej preprocessora (bez ponownego fitu).
    """
    features = df.drop(columns=[TARGET_COLUMN])
    matrix = artifacts.transformer.transform(features)
    return matrix.astype(np.float32)
