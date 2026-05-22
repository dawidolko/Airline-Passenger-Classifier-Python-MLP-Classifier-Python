"""
Logika eksperymentu — klasyfikacja jakości wina (Wine Quality).

Etapy:
1. Wczytanie CSV i kontrola braków.
2. Podział na cechy chemiczne (X) i etykietę quality (y) — klasyfikacja wieloklasowa.
3. Skalowanie cech wewnątrz Pipeline (StandardScaler) — bez przecieku danych.
4. Model główny: MLP (sieć neuronowa) w 3 wariantach architektury.
   Modele porównawcze: Random Forest × 3 oraz XGBoost × 3.
5. Stratyfikowana 5-krotna walidacja krzyżowa (accuracy, balanced accuracy, F1-macro).
6. Held-out test: najlepszy MLP (wg balanced accuracy w CV) oceniany na odłożonych 20% danych.
7. Eksport wyników: CSV, tabele LaTeX, macierz pomyłek, wykresy HTML.

Brak przecieku: skalowanie jest wewnątrz Pipeline, więc StandardScaler dopasowuje się
wyłącznie na zbiorze treningowym każdej foldy walidacji oraz na zbiorze treningowym
przed oceną na held-out.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    DATA_PATH,
    FEATURE_COLUMNS,
    ID_COLUMN,
    RANDOM_STATE,
    RESULTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.wykresy import zapisz_wykresy_html

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# XGBoost jest opcjonalny — jeśli nie jest zainstalowany, pomijamy tę rodzinę modeli.
try:
    from xgboost import XGBClassifier

    XGBOOST_DOSTEPNY = True
except Exception:  # ImportError, brak libomp na macOS itd.
    XGBClassifier = None  # type: ignore[misc, assignment]
    XGBOOST_DOSTEPNY = False


# ---------------------------------------------------------------------------
# Wczytanie danych i przekształcenie etykiet
# ---------------------------------------------------------------------------


def wczytaj_pelna_ramke(sciezka: Path) -> pd.DataFrame:
    """
    Wczytuje pełny CSV Wine Quality (WineQT.csv).

    Zbiór zawiera 11 cech chemicznych oraz kolumnę `quality` (ocena 3–9, wieloklasowa).
    Kolumna `Id` służy tylko jako identyfikator wiersza i nie trafia do modelu.
    """
    df = pd.read_csv(sciezka)
    df.columns = df.columns.str.strip()

    if df.isna().any().any():
        raise ValueError(
            "Wykryto brakujące wartości w zbiorze — usuń lub uzupełnij je przed treningiem."
        )

    brakujace = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if brakujace:
        raise ValueError(
            f"W zbiorze brakuje kolumn: {brakujace}. "
            "Oczekiwany plik: data/WineQT.csv (Kaggle: wine-quality-dataset)."
        )

    df = df.copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


def podziel_na_cechy_i_etykiete(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Zwraca macierz cech X (11 parametrów chemicznych) oraz wektor etykiet y (`quality`)."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].values
    return X, y, list(FEATURE_COLUMNS)


def wczytaj_i_sprawdz_dane(sciezka: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Wczytuje plik CSV i zwraca (X, y, nazwy_cech)."""
    df = wczytaj_pelna_ramke(sciezka)
    return podziel_na_cechy_i_etykiete(df)


# ---------------------------------------------------------------------------
# Pipeline i definicje modeli
# ---------------------------------------------------------------------------


def utworz_pipeline(estymator: Any, nazwy_kolumn_cech: list[str]) -> Pipeline:
    """
    Buduje Pipeline: StandardScaler na cechach numerycznych, potem klasyfikator.

    StandardScaler (standaryzacja do średniej 0 i odchylenia 1) jest właściwy dla
    sieci MLP, która jest wrażliwa na skalę cech. ColumnTransformer dopasowuje skaler
    wyłącznie na zbiorze treningowym danej foldy CV — to chroni przed przeciekiem danych.
    """
    preproc = ColumnTransformer(
        transformers=[("num", StandardScaler(), nazwy_kolumn_cech)],
        remainder="drop",
    )
    return Pipeline([("preprocess", preproc), ("clf", estymator)])


def definicja_modeli_glownych() -> list[tuple[str, Any]]:
    """
    Model główny — sieć neuronowa MLP w 3 wariantach architektury.

    Warianty różnią się liczbą i rozmiarem warstw ukrytych oraz regularyzacją (alpha).
    Etykieta `quality` jest całkowitoliczbowa; `early_stopping` wyłączone ze względu na
    kompatybilność ze sklearn przy walidacji wewnętrznej MLP.
    """
    return [
        (
            "mlp_64",
            MLPClassifier(
                hidden_layer_sizes=(64,),
                activation="relu",
                alpha=1e-3,
                learning_rate_init=1e-3,
                max_iter=800,
                early_stopping=False,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "mlp_128_64",
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=1e-3,
                learning_rate_init=1e-3,
                max_iter=800,
                early_stopping=False,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "mlp_128_64_32",
            MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                alpha=1e-2,
                learning_rate_init=1e-3,
                max_iter=1000,
                early_stopping=False,
                random_state=RANDOM_STATE,
            ),
        ),
    ]


def definicja_modeli_porownawczych() -> list[tuple[str, Any]]:
    """
    Modele porównawcze wg wytycznych projektu 2:
    - Random Forest × 3,
    - XGBoost × 3 (gdy biblioteka jest dostępna w środowisku).
    """
    lasy = [
        ("rf_100", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)),
        ("rf_200", RandomForestClassifier(n_estimators=200, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1)),
        ("rf_300", RandomForestClassifier(n_estimators=300, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1)),
    ]

    if not XGBOOST_DOSTEPNY:
        return lasy

    xgb = [
        (
            "xgb_d3",
            XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.9,
                colsample_bytree=0.9, eval_metric="mlogloss", tree_method="hist",
                random_state=RANDOM_STATE, n_jobs=-1,
            ),
        ),
        (
            "xgb_d6",
            XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9,
                colsample_bytree=0.9, eval_metric="mlogloss", tree_method="hist",
                random_state=RANDOM_STATE, n_jobs=-1,
            ),
        ),
        (
            "xgb_d8",
            XGBClassifier(
                n_estimators=400, max_depth=8, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, eval_metric="mlogloss", tree_method="hist",
                random_state=RANDOM_STATE, n_jobs=-1,
            ),
        ),
    ]
    return lasy + xgb


def definicja_wszystkich_modeli() -> list[tuple[str, Any]]:
    """Łączy model główny (MLP) z modelami porównawczymi (RF, XGBoost)."""
    return definicja_modeli_glownych() + definicja_modeli_porownawczych()


def rodzina_z_nazwy(nazwa: str) -> str:
    """Mapuje nazwę wariantu na rodzinę modelu (do tabel grupowych)."""
    if nazwa.startswith("mlp_"):
        return "MLP"
    if nazwa.startswith("rf_"):
        return "RandomForest"
    if nazwa.startswith("xgb_"):
        return "XGBoost"
    return "Inne"


# ---------------------------------------------------------------------------
# Walidacja krzyżowa i held-out test
# ---------------------------------------------------------------------------


def uruchom_walidacje(
    X: pd.DataFrame,
    y: np.ndarray,
    nazwy_cech: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Dla każdego modelu (MLP, RF, XGBoost) wykonuje stratyfikowaną 5-krotną CV.

    Metryki: accuracy, balanced_accuracy (średnia czułość po klasach) oraz f1_macro
    (uśrednione F1 po klasach — istotne przy niezbalansowanym rozkładzie ocen jakości).

    Zwraca:
        szczegoly — jeden wiersz na model (średnia i std z 5 foldów),
        grupy — agregaty (średnia, maksimum) w obrębie rodzin MLP / RandomForest / XGBoost.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    metryki = ["accuracy", "balanced_accuracy", "f1_macro"]

    wiersze: list[dict[str, Any]] = []
    for nazwa, clf in definicja_wszystkich_modeli():
        pipe = utworz_pipeline(clf, nazwy_cech)
        cv_wynik = cross_validate(
            pipe, X, y, cv=cv, scoring=metryki, n_jobs=-1, return_train_score=False
        )
        wiersze.append(
            {
                "model": nazwa,
                "rodzina": rodzina_z_nazwy(nazwa),
                "accuracy_mean": float(np.mean(cv_wynik["test_accuracy"])),
                "accuracy_std": float(np.std(cv_wynik["test_accuracy"])),
                "balanced_accuracy_mean": float(np.mean(cv_wynik["test_balanced_accuracy"])),
                "balanced_accuracy_std": float(np.std(cv_wynik["test_balanced_accuracy"])),
                "f1_macro_mean": float(np.mean(cv_wynik["test_f1_macro"])),
                "f1_macro_std": float(np.std(cv_wynik["test_f1_macro"])),
            }
        )

    szczegoly = pd.DataFrame(wiersze)

    grupy = szczegoly.groupby("rodzina", as_index=False).agg(
        accuracy_srednia=("accuracy_mean", "mean"),
        accuracy_max=("accuracy_mean", "max"),
        balanced_accuracy_srednia=("balanced_accuracy_mean", "mean"),
        balanced_accuracy_max=("balanced_accuracy_mean", "max"),
        f1_macro_srednia=("f1_macro_mean", "mean"),
        f1_macro_max=("f1_macro_mean", "max"),
    )

    return szczegoly, grupy


def wybierz_najlepszy_mlp(szczegoly: pd.DataFrame) -> str:
    """Zwraca nazwę wariantu MLP o najwyższej balanced accuracy w CV."""
    mlp = szczegoly[szczegoly["rodzina"] == "MLP"]
    if mlp.empty:
        raise ValueError("Brak wyników dla modeli MLP — sprawdź definicję modeli głównych.")
    return str(mlp.sort_values("balanced_accuracy_mean", ascending=False).iloc[0]["model"])


def ocena_na_tescie(
    X: pd.DataFrame,
    y: np.ndarray,
    nazwy_cech: list[str],
    nazwa_modelu: str,
) -> dict[str, Any]:
    """
    Trenuje wskazany model (najlepszy MLP) na zbiorze treningowym i ocenia na held-out test.

    Podział train/test jest stratyfikowany i wykonany jednym wspólnym seedem. Skaler
    dopasowuje się tylko na treningu (wewnątrz Pipeline) — bez przecieku do testu.

    Zwraca słownik: metryki na teście, macierz pomyłek, raport klasyfikacji, etykiety klas.
    """
    modele = dict(definicja_wszystkich_modeli())
    if nazwa_modelu not in modele:
        raise ValueError(f"Nieznany model: {nazwa_modelu}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipe = utworz_pipeline(modele[nazwa_modelu], nazwy_cech)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    etykiety = sorted(np.unique(y))
    cm = confusion_matrix(y_test, y_pred, labels=etykiety)
    raport = classification_report(
        y_test, y_pred, labels=etykiety, output_dict=True, zero_division=0
    )

    return {
        "model": nazwa_modelu,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": cm,
        "labels": etykiety,
        "classification_report": raport,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


# ---------------------------------------------------------------------------
# Eksport wyników
# ---------------------------------------------------------------------------


def zapisz_latex(szczegoly: pd.DataFrame, grupy: pd.DataFrame, katalog: Path) -> None:
    """Zapisuje tabele wyników w formacie LaTeX (booktabs) do katalogu results/."""
    katalog.mkdir(parents=True, exist_ok=True)
    fmt = "%.4f"

    szczegoly.to_latex(
        katalog / "wyniki_szczegolowe.tex",
        index=False,
        float_format=fmt,
        caption="Walidacja krzyzowa (5-fold) — klasyfikacja jakosci wina: MLP, Random Forest, XGBoost.",
        label="tab:szczegolowe",
        escape=True,
    )

    grupy.to_latex(
        katalog / "wyniki_agregaty_rodzin.tex",
        index=False,
        float_format=fmt,
        caption="Srednie i maksymalne metryki w obrebie rodziny modelu (MLP / RF / XGBoost).",
        label="tab:agregaty",
        escape=True,
    )


def zapisz_macierz_pomylek(wynik_test: dict[str, Any], katalog: Path) -> pd.DataFrame:
    """Zapisuje macierz pomyłek z held-out testu jako CSV i zwraca ją jako DataFrame."""
    katalog.mkdir(parents=True, exist_ok=True)
    cm_df = pd.DataFrame(
        wynik_test["confusion_matrix"],
        index=[f"prawda:{e}" for e in wynik_test["labels"]],
        columns=[f"pred:{e}" for e in wynik_test["labels"]],
    )
    cm_df.to_csv(katalog / "macierz_pomylek_test.csv")
    return cm_df


def uruchom_pelny_eksperyment(
    sciezka_danych: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Główny punkt wejścia: wczytanie danych, CV, held-out test najlepszego MLP, zapis wyników.

    Zwraca: (szczegoly, grupy, wynik_testu_najlepszego_MLP).
    """
    path = sciezka_danych or DATA_PATH
    df_pelny = wczytaj_pelna_ramke(path)
    X, y, nazwy = podziel_na_cechy_i_etykiete(df_pelny)

    szczegoly, grupy = uruchom_walidacje(X, y, nazwy)
    najlepszy_mlp = wybierz_najlepszy_mlp(szczegoly)
    wynik_test = ocena_na_tescie(X, y, nazwy, najlepszy_mlp)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    szczegoly.to_csv(RESULTS_DIR / "wyniki_szczegolowe.csv", index=False)
    grupy.to_csv(RESULTS_DIR / "wyniki_agregaty_rodzin.csv", index=False)
    zapisz_latex(szczegoly, grupy, RESULTS_DIR)
    cm_df = zapisz_macierz_pomylek(wynik_test, RESULTS_DIR)

    # Tabela metryk testowych najlepszego MLP
    pd.DataFrame(
        [
            {
                "model": wynik_test["model"],
                "accuracy_test": wynik_test["accuracy"],
                "balanced_accuracy_test": wynik_test["balanced_accuracy"],
                "f1_macro_test": wynik_test["f1_macro"],
                "n_train": wynik_test["n_train"],
                "n_test": wynik_test["n_test"],
            }
        ]
    ).to_csv(RESULTS_DIR / "wyniki_test_mlp.csv", index=False)

    zapisz_wykresy_html(df_pelny, szczegoly, grupy, wynik_test, RESULTS_DIR)

    meta = RESULTS_DIR / "wersje_bibliotek.txt"
    xgb_wersja = ""
    if XGBOOST_DOSTEPNY:
        import xgboost

        xgb_wersja = f"xgboost {xgboost.__version__}\n"
    xgb_info = xgb_wersja if XGBOOST_DOSTEPNY else (
        "xgboost: niedostepny (pominieto 3 warianty; porownanie MLP + Random Forest)\n"
    )
    meta.write_text(
        "Odtwarzalnosc: ten plik zapisuje wersje bibliotek przy generowaniu wynikow.\n"
        f"scikit-learn {sklearn.__version__}\n"
        f"numpy {np.__version__}\n"
        f"pandas {pd.__version__}\n"
        f"{xgb_info}",
        encoding="utf-8",
    )

    return szczegoly, grupy, wynik_test
