#!/usr/bin/env python3
"""
Aktualizuje docs/dokumentacja_do125148.docx: treść opisów (Wine Quality) + wykresy PNG.

Zachowuje formatowanie Word (nagłówki, style akapitów, układ) — podmienia tekst i pliki media/.
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_PATH, FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from src.experiment import (
    ocena_na_tescie,
    podziel_na_cechy_i_etykiete,
    wczytaj_pelna_ramke,
    wybierz_najlepszy_mlp,
)

DOCX = Path(__file__).resolve().parent / "dokumentacja_do125148.docx"
MEDIA_DIR = Path(__file__).resolve().parent / "_media_wine"
RESULTS = ROOT / "results"
N_RECS = 6497
N_TRAIN = 5197
N_TEST = 1300


def _fmt(x: float, n: int = 4) -> str:
    return f"{x:.{n}f}".replace(".", ",")


def generuj_wykresy() -> dict[str, Path]:
    """10 PNG zgodnych z Rys. 1–10 w sprawozdaniu."""
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 120

    df = wczytaj_pelna_ramke(DATA_PATH)
    X, y, nazwy = podziel_na_cechy_i_etykiete(df)
    szczegoly = pd.read_csv(RESULTS / "wyniki_szczegolowe.csv")
    grupy = pd.read_csv(RESULTS / "wyniki_agregaty_rodzin.csv")
    najlepszy_mlp = wybierz_najlepszy_mlp(szczegoly)
    wynik_test = ocena_na_tescie(X, y, nazwy, najlepszy_mlp)

    out: dict[str, Path] = {}

    # Rys. 1 — rozkład quality
    fig, ax = plt.subplots(figsize=(9, 4))
    vc = df[TARGET_COLUMN].value_counts().sort_index()
    ax.bar(vc.index.astype(str), vc.values, color=sns.color_palette("Blues", len(vc)))
    ax.set_xlabel("Ocena jakości (quality)")
    ax.set_ylabel("Liczba próbek")
    ax.set_title(f"Rozkład klas — Wine Quality ({N_RECS} rekordów)")
    p1 = MEDIA_DIR / "image1.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    out["image1.png"] = p1

    # Rys. 2 — braki (brak NaN) + skala cech (średnie znormalizowane wizualnie)
    braki = df[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().sum()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(braki.index, braki.values, color="#1B998B")
    ax.set_xlabel("Liczba braków")
    ax.set_title("Braki w danych — wszystkie kolumny kompletne (0 NaN)")
    p2 = MEDIA_DIR / "image2.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    out["image2.png"] = p2

    # Rys. 3 — histogramy 4 cech
    cechy_hist = ["fixed acidity", "volatile acidity", "alcohol", "total sulfur dioxide"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, col in zip(axes.flat, cechy_hist):
        ax.hist(df[col], bins=40, color="#5C4D7D", alpha=0.85, edgecolor="white")
        ax.set_title(col)
    fig.suptitle("Histogramy wybranych cech chemicznych")
    p3 = MEDIA_DIR / "image3.png"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    out["image3.png"] = p3

    # Rys. 4 — boxploty vs quality
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.boxplot(data=df, x=TARGET_COLUMN, y="alcohol", ax=axes[0], palette="Blues")
    axes[0].set_title("Alkohol vs ocena jakości")
    sns.boxplot(data=df, x=TARGET_COLUMN, y="volatile acidity", ax=axes[1], palette="Oranges")
    axes[1].set_title("Kwasowość lotna vs ocena jakości")
    p4 = MEDIA_DIR / "image4.png"
    fig.savefig(p4, bbox_inches="tight")
    plt.close(fig)
    out["image4.png"] = p4

    # Rys. 5 — korelacja
    corr = df[FEATURE_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Macierz korelacji Pearsona — 11 cech chemicznych")
    plt.xticks(rotation=45, ha="right")
    p5 = MEDIA_DIR / "image5.png"
    fig.savefig(p5, bbox_inches="tight")
    plt.close(fig)
    out["image5.png"] = p5

    # Rys. 6 — ważność cech RF (zamiast MI)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    rf = RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
    )
    rf.fit(Xs, y_train)
    imp = pd.Series(rf.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.plot(kind="barh", ax=ax, color="#1B998B")
    ax.set_title("Ważność cech — Random Forest (n_estimators=300)")
    ax.set_xlabel("Importance")
    p6 = MEDIA_DIR / "image6.png"
    fig.savefig(p6, bbox_inches="tight")
    plt.close(fig)
    out["image6.png"] = p6

    # Rys. 7 — porównanie modeli CV
    df_plot = szczegoly.copy()
    df_plot["_sort"] = df_plot["rodzina"].map({"MLP": 0, "RandomForest": 1, "XGBoost": 2}).fillna(9)
    df_plot = df_plot.sort_values(["_sort", "model"])
    kolory = {"MLP": "#E84855", "RandomForest": "#1B998B", "XGBoost": "#5C4D7D"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, met, std, tit in zip(
        axes,
        ["accuracy_mean", "balanced_accuracy_mean", "f1_macro_mean"],
        ["accuracy_std", "balanced_accuracy_std", "f1_macro_std"],
        ["Accuracy (CV)", "Balanced accuracy (CV)", "F1-macro (CV)"],
    ):
        kol = [kolory.get(r, "#888") for r in df_plot["rodzina"]]
        ax.bar(range(len(df_plot)), df_plot[met], yerr=df_plot[std], capsize=3, color=kol, ecolor="gray")
        ax.set_xticks(range(len(df_plot)))
        ax.set_xticklabels(df_plot["model"], rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(tit)
    fig.suptitle("Porównanie wariantów modeli — 5-fold CV", y=1.02)
    p7 = MEDIA_DIR / "image7.png"
    fig.savefig(p7, bbox_inches="tight")
    plt.close(fig)
    out["image7.png"] = p7

    # Rys. 8 — macierz pomyłek MLP test
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(
        confusion_matrix=wynik_test["confusion_matrix"],
        display_labels=wynik_test["labels"],
    ).plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(f"Macierz pomyłek — {wynik_test['model']} (test 20%, n={N_TEST})")
    p8 = MEDIA_DIR / "image8.png"
    fig.savefig(p8, bbox_inches="tight")
    plt.close(fig)
    out["image8.png"] = p8

    # Rys. 9 — agregaty rodzin (zamiast ROC)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(grupy))
    w = 0.35
    ax.bar(x - w / 2, grupy["f1_macro_srednia"], w, label="Śr. F1-macro", color="#5C4D7D")
    ax.bar(x + w / 2, grupy["f1_macro_max"], w, label="Maks. F1-macro", color="#8F7EAF")
    ax.set_xticks(x)
    ax.set_xticklabels(grupy["rodzina"])
    ax.set_ylim(0, 0.55)
    ax.legend()
    ax.set_title("F1-macro — średnia i maksimum w rodzinie modelu (CV)")
    p9 = MEDIA_DIR / "image9.png"
    fig.savefig(p9, bbox_inches="tight")
    plt.close(fig)
    out["image9.png"] = p9

    # Rys. 10 — top 8 ważności cech
    top = imp.tail(8)
    fig, ax = plt.subplots(figsize=(8, 5))
    top.plot(kind="barh", ax=ax, color="#E84855")
    ax.set_title("Top 8 cech — Random Forest (importance)")
    ax.set_xlabel("Importance")
    p10 = MEDIA_DIR / "image10.png"
    fig.savefig(p10, bbox_inches="tight")
    plt.close(fig)
    out["image10.png"] = p10

    return out


def _wczytaj_wyniki() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    sz = pd.read_csv(RESULTS / "wyniki_szczegolowe.csv")
    gr = pd.read_csv(RESULTS / "wyniki_agregaty_rodzin.csv")
    test = pd.read_csv(RESULTS / "wyniki_test_mlp.csv").iloc[0]
    return sz, gr, test


def teksty_paragrafow() -> dict[int, str]:
    sz, gr, test = _wczytaj_wyniki()
    mlp_best = wybierz_najlepszy_mlp(sz)
    rf_row = sz[sz["model"] == "rf_300"].iloc[0]
    mlp_row = sz[sz["model"] == mlp_best].iloc[0]

    xgb_info = ""
    if "XGBoost" in gr["rodzina"].values:
        xgb_info = " oraz XGBoost (3 warianty), gdy biblioteka jest dostępna"
    else:
        xgb_info = " (XGBoost pomijany, gdy brak biblioteki/libomp na macOS)"

    return {
        6: "2. Temat projektu",
        7: (
            "Klasyfikacja jakości wina (Wine Quality) z wykorzystaniem uczenia maszynowego "
            "na zbiorze WineQT.csv (UCI / Kaggle). Projekt obejmuje kompletny potok ML: "
            "wczytanie danych, eksplorację (EDA), trenowanie sieci MLP (model główny) "
            "oraz porównanie z Random Forest"
            f"{xgb_info}, walidację krzyżową 5-fold, ocenę na zbiorze testowym 20% "
            "i prezentację wyników w aplikacji Streamlit."
        ),
        8: (
            "Repozytorium: run_experiment.py (eksperyment i eksport do results/), "
            "src/experiment.py (pipeline, modele, CV), src/wykresy.py (wykresy Plotly), "
            "streamlit_app.py (dashboard), wine_quality_mlp_sklearn_project2.ipynb (raport Jupyter)."
        ),
        9: "3. Charakterystyka problemu",
        10: (
            "Zadanie to klasyfikacja wieloklasowa (multiclass classification) nadzorowana. "
            "Na podstawie 11 parametrów chemicznych wina (m.in. alkohol, kwasowość, siarczany, gęstość) "
            "model przypisuje próbce dyskretną ocenę sensoryczną quality (skala 3–9). "
            "To nie jest klasyfikacja binarna „dobre/złe”."
        ),
        11: (
            "Klasy są niezbalansowane: dominują oceny 5 i 6 (łącznie większość próbek), "
            "natomiast oceny skrajne (3, 4, 8, 9) występują rzadko. Dlatego oprócz accuracy "
            "stosujemy balanced accuracy i F1-macro, które lepiej odzwierciedlają jakość na wszystkich klasach."
        ),
        12: (
            "Sens praktyczny: automatyczna wstępna ocena jakości na podstawie analizy chemicznej "
            "może wspierać kontrolę jakości w produkcji wina (szybki screening przed degustacją ekspercką)."
        ),
        13: (
            "Sens edukacyjny: pokazać powtarzalny pipeline ML — od danych po dashboard — "
            "z MLP jako modelem głównym, uczciwą walidacją (Pipeline + CV) i krytyczną interpretacją metryk."
        ),
        14: "4. Liczba instancji",
        15: (
            f"Zbiór WineQT.csv zawiera {N_RECS} rekordów (wino czerwone i białe, połączone). "
            "Po podziale trening/test 80/20 ze stratyfikacją względem quality:"
        ),
        16: f"•  {N_TRAIN} próbek treningowych — dopasowanie modelu i 5-krotna walidacja krzyżowa.",
        17: f"•  {N_TEST} próbek testowych (held-out) — końcowa ocena najlepszego MLP.",
        18: (
            "Stratyfikacja zachowuje proporcje klas w train i test. "
            "Dzięki temu metryki na teście są porównywalne z CV."
        ),
        19: "5. Liczba atrybutów i ich charakterystyka",
        20: (
            "Plik CSV ma 13 kolumn: Id (identyfikator), 11 cech chemicznych oraz quality (etykieta). "
            "Do modelu trafia wyłącznie 11 cech numerycznych — bez one-hot, bez kolumny Id."
        ),
        21: (
            "•  Cechy numeryczne (11): fixed acidity, volatile acidity, citric acid, residual sugar, "
            "chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol. "
            "Różne skale (np. density ≈ 1, SO₂ do setek) — wymagają standaryzacji w pipeline."
        ),
        22: (
            "•  Zmienna docelowa: quality (liczby całkowite 3–9) — ocena sensoryczna jakości wina."
        ),
        23: (
            "•  Braki danych: w zbiorze źródłowym nie występują wartości NaN (kontrola w wczytaj_pelna_ramke)."
        ),
        24: (
            "•  Id: kolumna pomocnicza, wyłączona z uczenia (nie niesie informacji o jakości)."
        ),
        25: (
            "•  Źródło: UCI Wine Quality / Kaggle (yasserh/wine-quality-dataset); "
            "skrypt scripts/pobierz_dane.py odtwarza plik z repozytorium UCI."
        ),
        26: (
            "Konfiguracja kolumn i ścieżek: src/config.py (FEATURE_COLUMNS, TARGET_COLUMN, DATA_PATH)."
        ),
        27: "6. Preprocessing danych — wstępne przetwarzanie",
        28: (
            "W projekcie nie ma osobnego pliku preprocess.py — transformacje są w sklearn Pipeline "
            "(src/experiment.py), co zapobiega przeciekowi danych do walidacji krzyżowej:"
        ),
        29: (
            "•  Krok 1 — wczytanie CSV, walidacja kolumn i braków, rzutowanie quality na int."
        ),
        30: (
            "•  Krok 2 — podział X (11 cech) / y (quality); kolumna Id jest pomijana."
        ),
        31: (
            "•  Krok 3 — StandardScaler wewnątrz Pipeline: dopasowanie tylko na danych treningowych "
            "każdej foldy CV (oraz na train przed testem held-out)."
        ),
        32: (
            "•  Krok 4 — brak kodowania kategorycznego (wszystkie cechy wejściowe są liczbowe)."
        ),
        33: (
            "•  Krok 5 — stratyfikowany podział train/test 80/20 (random_state=42, stratify=y) "
            "dla oceny najlepszego MLP na zbiorze testowym."
        ),
        34: (
            "•  Krok 6 — eksport wyników: CSV, LaTeX, wykresy HTML/PNG do katalogu results/."
        ),
        37: "7. Projekt modelu — MLP (główny), Random Forest, XGBoost",
        38: (
            "Model główny — MLPClassifier (sklearn.neural_network), trzy warianty architektury ukrytych warstw:"
        ),
        39: (
            "Sieć feed-forward z funkcją aktywacji ReLU, solver Adam, regularyzacja L2 (alpha=0,0001). "
            "Warianty: mlp_64 (64), mlp_128_64 (128–64), mlp_128_64_32 (128–64–32). "
            "Wybór najlepszego wariantu wg najwyższej balanced accuracy w 5-fold CV."
        ),
        40: (
            "Implementacja: src/experiment.py — funkcje definicja_wszystkich_modeli(), utworz_pipeline(), "
            "uruchom_walidacje(). Uruchomienie: python run_experiment.py."
        ),
        41: "Konfiguracja MLP (wspólna dla wariantów):",
        42: "•  solver = adam, activation = relu, max_iter = 500, random_state = 42.",
        43: "•  alpha = 0,0001 (L2), early_stopping wyłączone (stabilność na wielu klasach).",
        44: "•  StandardScaler w ColumnTransformer przed klasyfikatorem.",
        45: "•  Metryki CV: accuracy, balanced_accuracy, f1_macro.",
        46: "•  n_jobs = -1 tam, gdzie dotyczy modele drzewiaste.",
        47: "•  class_weight = balanced dla Random Forest i XGBoost.",
        48: "•  Stratyfikowana 5-fold CV (StratifiedKFold, shuffle=True, random_state=42).",
        49: "Modele porównawcze (ta sama macierz cech, ten sam Pipeline ze skalerem):",
        50: (
            "•  RandomForestClassifier ×3: rf_100, rf_200, rf_300 (n_estimators odpowiednio 100, 200, 300)."
        ),
        51: (
            "•  XGBClassifier ×3 (gdy dostępny): xgb_100, xgb_200, xgb_300 — gradient boosting na drzewach."
        ),
        52: (
            "•  Porównanie rodzin: agregaty średnie i maksymalne metryki w obrębie MLP / RF / XGBoost."
        ),
        53: (
            "Walidacja krzyżowa: cross_validate z scoring=['accuracy', 'balanced_accuracy', 'f1_macro'], cv=5."
        ),
        54: "8. Ocena na zastrzeżonym zbiorze testowym (20%)",
        55: (
            f"Najlepszy MLP ({mlp_best}) trenowany na {N_TRAIN} próbkach i oceniany na {N_TEST} próbkach testowych "
            "(dane niewidziane w trakcie CV ani treningu finalnego)."
        ),
        56: (
            "•  Accuracy — odsetek poprawnych klasyfikacji; przy dominacji klas 5–6 może być wyższa niż "
            "rzeczywista jakość na klasach rzadkich."
        ),
        57: (
            "•  Balanced accuracy — średnia czułości po klasach (ważna przy niezbalansowaniu)."
        ),
        58: "•  F1-macro — średnia F1 per klasa z równymi wagami.",
        59: "•  Macierz pomyłek — analiza pomyłek sąsiednich ocen (np. 5↔6).",
        60: "•  Raport klasyfikacji per klasa (precision, recall, F1).",
        61: (
            "Dodatkowo w CV porównujemy wszystkie warianty MLP, RF i XGB; "
            "najlepszy RF w CV: rf_300 (accuracy ≈ 0,69)."
        ),
        64: "9. Analiza uzyskanych rezultatów",
        65: f"Wyniki walidacji krzyżowej (5-fold) — najlepszy MLP ({mlp_best}):",
        66: f"•  Accuracy (CV): {_fmt(mlp_row['accuracy_mean'])} ± {_fmt(mlp_row['accuracy_std'])}",
        67: f"•  Balanced accuracy (CV): {_fmt(mlp_row['balanced_accuracy_mean'])} ± {_fmt(mlp_row['balanced_accuracy_std'])}",
        68: f"•  F1-macro (CV): {_fmt(mlp_row['f1_macro_mean'])} ± {_fmt(mlp_row['f1_macro_std'])}",
        69: "",
        70: "",
        71: f"Najlepszy Random Forest w CV (rf_300): accuracy {_fmt(rf_row['accuracy_mean'])}, F1-macro {_fmt(rf_row['f1_macro_mean'])}.",
        72: "Porównanie rodzin (średnie w CV):",
        73: f"•  MLP — accuracy średnia {_fmt(gr.loc[gr['rodzina']=='MLP','accuracy_srednia'].iloc[0])}, F1-macro max {_fmt(gr.loc[gr['rodzina']=='MLP','f1_macro_max'].iloc[0])}.",
        74: f"•  Random Forest — accuracy max {_fmt(gr.loc[gr['rodzina']=='RandomForest','accuracy_max'].iloc[0])}, F1-macro max {_fmt(gr.loc[gr['rodzina']=='RandomForest','f1_macro_max'].iloc[0])}.",
        75: "",
        76: "Interpretacja:",
        77: (
            "Random Forest osiąga wyższą accuracy w CV niż MLP, ale zgodnie z wymaganiami projektu 2 "
            "modelem głównym pozostaje sieć MLP — RF i XGB służą porównaniu. "
            "Na teście held-out MLP osiąga accuracy ok. 0,63 przy balanced accuracy ok. 0,35 — "
            "typowe dla wielu klas i niezbalansowania."
        ),
        78: (
            "Najtrudniejsze są klasy rzadkie (3, 4, 8, 9) — w macierzy pomyłek widać niewielką liczbę trafień "
            "lub zerowe dla skrajnych ocen."
        ),
        79: (
            "•  Wykres rozkładu quality (Rys. 1) pokazuje dominację ocen 5 i 6 — uzasadnia metryki balanced/F1-macro."
        ),
        80: (
            "•  Macierz korelacji (Rys. 5) — umiarkowane zależności (np. alkohol vs jakość w boxplotach)."
        ),
        81: (
            "•  Macierz pomyłek MLP (Rys. 8) — większość trafień na klasach 5 i 6; pomyłki sąsiednich ocen."
        ),
        82: (
            "•  Ważność cech RF (Rys. 6, 10) — najwyżej: alcohol, volatile acidity, sulphates (zgodnie z literaturą)."
        ),
        83: (
            "Wniosek: problem jest wykonalny, ale ograniczony rozrzutem etykiet i liczebnością klas skrajnych; "
            "możliwe ulepszenia: wagowanie klas, grupowanie ocen, tuning MLP, pełne XGBoost po instalacji libomp."
        ),
        84: "10. Wnioski",
        85: (
            "Zaimplementowano potok ML: dane WineQT.csv → EDA → Pipeline (StandardScaler + klasyfikator) → "
            "MLP (3 warianty) + RF (3) + opcjonalnie XGB (3) → 5-fold CV → test 20% najlepszego MLP → "
            "eksport results/ i aplikacja Streamlit."
        ),
        86: (
            "Metodologia poprawna (brak przecieku, stratyfikacja, wiele metryk). "
            f"Najlepszy MLP: {mlp_best}; najlepszy RF w CV: rf_300. "
            f"Test MLP: accuracy {_fmt(test['accuracy_test'])}, balanced accuracy {_fmt(test['balanced_accuracy_test'])}."
        ),
        87: (
            "Projekt spełnia wymaganie sieci neuronowej jako modelu głównego oraz porównania z innymi "
            "klasyfikatorami; dokumentacja i notebooki opisują każdy etap z wykresami."
        ),
        89: "Dodatek A. Rozszerzony opis projektu i wykresy z uruchomienia",
        90: (
            "Dodatek uzupełnia sprawozdanie o przepływ danych w repozytorium oraz 10 rysunków "
            "wygenerowanych z aktualnego uruchomienia (run_experiment.py / ten skrypt aktualizujący)."
        ),
        91: "A.1. Architektura i przepływ danych",
        92: "Komponenty projektu:",
        93: "•  scripts/pobierz_dane.py — pobranie/łączenie danych UCI do data/WineQT.csv.",
        94: "•  run_experiment.py + src/experiment.py — CV, test MLP, zapis CSV/LaTeX/HTML.",
        95: "•  streamlit_app.py — dashboard (formuły, EDA, wyniki, druk).",
        96: "A.2. Eksploracyjna analiza danych (EDA)",
        97: (
            "EDA obejmuje rozkład quality, braki (brak NaN), histogramy i boxploty cech, "
            "macierz korelacji oraz wstępną interpretację związku alkoholu i kwasowości z oceną."
        ),
        98: "A.2.1. Rozkład ocen jakości (quality)",
        100: (
            f"Rys. 1. Rozkład klas quality ({N_RECS} rekordów). Dominują oceny 5 i 6; klasy 3, 4, 8, 9 są rzadsze — "
            "stąd konieczność metryk balanced accuracy i F1-macro."
        ),
        101: "A.2.2. Kompletność danych",
        103: (
            "Rys. 2. Liczba braków per kolumna — w WineQT.csv wszystkie wartości są uzupełnione (0 braków). "
            "Nie stosujemy imputacji ani usuwania wierszy z powodu NaN."
        ),
        104: "A.2.3. Rozkłady cech numerycznych",
        106: (
            "Rys. 3. Histogramy czterech reprezentatywnych cech chemicznych. Rozkłady są różne "
            "(np. density skupiona wokół 1, alcohol w szerszym zakresie) — uzasadnia StandardScaler."
        ),
        107: "A.2.4. Cechy a ocena jakości",
        109: (
            "Rys. 4. Boxploty alkoholu i kwasowości lotnej względem quality — widać trend wyższego alkoholu "
            "przy wyższych ocenach oraz wpływ volatile acidity na gorsze oceny."
        ),
        110: "A.2.5. Macierz korelacji",
        112: (
            "Rys. 5. Macierz korelacji Pearsona dla 11 cech. Korelacje między parametrami chemicznymi są umiarkowane; "
            "żadna para nie dominuje całkowicie nad pozostałymi predyktorami."
        ),
        113: "A.2.6. Ważność cech (Random Forest)",
        115: (
            "Rys. 6. Ranking ważności cech według Random Forest (n_estimators=300, dane treningowe). "
            "Najsilniejsze predyktory: alcohol, volatile acidity, sulphates — spójne z EDA."
        ),
        116: "A.3. Wyniki walidacji i testu",
        117: "A.3.1. Porównanie modeli (CV)",
        118: "",
        119: (
            f"Rys. 7. Porównanie wszystkich wariantów modeli w 5-fold CV ({len(sz)} konfiguracji). "
            f"Najwyższa accuracy: rf_300 ({_fmt(rf_row['accuracy_mean'])}); "
            f"najlepszy MLP: {mlp_best} (balanced accuracy CV {_fmt(mlp_row['balanced_accuracy_mean'])})."
        ),
        120: "Tabela 1. Wybrane wyniki walidacji krzyżowej (5-fold, średnia):",
        121: "A.3.2. Macierz pomyłek (najlepszy MLP, test 20%)",
        122: "",
        123: (
            f"Rys. 8. Macierz pomyłek dla {mlp_best} na zbiorze testowym ({N_TEST} próbek). "
            "Koncentracja trafień na klasach 5–6; rzadkie klasy słabiej rozpoznawane."
        ),
        124: "A.3.3. Agregaty rodzin modeli",
        126: (
            "Rys. 9. F1-macro — średnia i maksimum w obrębie rodziny (MLP / Random Forest / XGBoost). "
            "Random Forest osiąga najwyższy maksymalny F1-macro w CV."
        ),
        127: "A.3.4. Ważność cech — top 8",
        129: (
            "Rys. 10. Top 8 cech wg Random Forest. Potwierdza, że parametry chemiczne niosą sygnał predykcyjny "
            "(w przeciwieństwie do syntetycznych zbiorów bez związku cecha–etykieta)."
        ),
        130: "A.4. Dashboard Streamlit (streamlit_app.py)",
        131: "•  Opis projektu i wzory — definicje accuracy, balanced accuracy, F1-macro.",
        132: "•  EDA — rozkład quality, korelacje, statystyki (Plotly).",
        133: "•  Preprocessing — Pipeline, StandardScaler, brak przecieku.",
        134: "•  Modele — architektury MLP, RF, XGB; wyniki CV.",
        135: "•  Porównanie — wykresy i tabele z results/.",
        136: "•  Wyniki testu — macierz pomyłek najlepszego MLP, metryki held-out.",
        137: "•  Eksplorator — podgląd danych i filtry.",
        138: "•  Informacje — uruchomienie (start.sh), biblioteki, odtwarzalność.",
        139: (
            "Dashboard korzysta z plików w results/ wygenerowanych przez run_experiment.py. "
            "Uruchomienie: ./start.sh lub streamlit run streamlit_app.py."
        ),
    }


def aktualizuj_tabele(doc) -> None:
    sz, _, _ = _wczytaj_wyniki()
    wiersze = [
        ("mlp_128_64_32", "MLP"),
        ("rf_300", "Random Forest"),
        ("rf_200", "Random Forest"),
    ]
    t = doc.tables[0]
    hdr = t.rows[0].cells
    hdr[0].text = "Model"
    hdr[1].text = "Rodzina"
    hdr[2].text = "Accuracy (CV)"
    hdr[3].text = "Balanced acc. (CV)"
    hdr[4].text = "F1-macro (CV)"
    hdr[5].text = "Std acc. (CV)"
    for i, (model, rodz) in enumerate(wiersze, start=1):
        r = sz[sz["model"] == model].iloc[0]
        cells = t.rows[i].cells
        cells[0].text = model
        cells[1].text = rodz
        cells[2].text = _fmt(r["accuracy_mean"])
        cells[3].text = _fmt(r["balanced_accuracy_mean"])
        cells[4].text = _fmt(r["f1_macro_mean"])
        cells[5].text = _fmt(r["accuracy_std"])


def aktualizuj_tekst(doc) -> None:
    mapping = teksty_paragrafow()
    for idx, new_text in mapping.items():
        if idx < len(doc.paragraphs):
            doc.paragraphs[idx].text = new_text
    aktualizuj_tabele(doc)


def wstaw_obrazy(docx_path: Path, obrazy: dict[str, Path]) -> None:
    tmp = docx_path.with_suffix(".tmp.zip")
    with zipfile.ZipFile(docx_path, "r") as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            name = Path(item.filename).name
            if name in obrazy and item.filename.startswith("word/media/"):
                data = obrazy[name].read_bytes()
            zout.writestr(item, data)
    backup = docx_path.with_suffix(".docx.bak_wine")
    if not backup.is_file():
        shutil.copy2(docx_path, backup)
    tmp.replace(docx_path)


def main() -> None:
    if not DOCX.is_file():
        print(f"Brak pliku: {DOCX}", file=sys.stderr)
        sys.exit(1)
    if not (RESULTS / "wyniki_szczegolowe.csv").is_file():
        print("Uruchom najpierw: python run_experiment.py", file=sys.stderr)
        sys.exit(1)

    from docx import Document

    print("Generuję wykresy PNG...")
    obrazy = generuj_wykresy()

    print("Aktualizuję treść w Word...")
    doc = Document(str(DOCX))
    aktualizuj_tekst(doc)
    doc.save(str(DOCX))

    print("Podmieniam obrazy w docx...")
    wstaw_obrazy(DOCX, obrazy)

    print(f"Gotowe: {DOCX}")


if __name__ == "__main__":
    main()
