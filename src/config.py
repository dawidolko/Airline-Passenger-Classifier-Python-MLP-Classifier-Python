"""
Stałe konfiguracyjne — klasyfikacja jakości wina (Wine Quality, UCI / Kaggle).

Jeden wspólny seed dla wszystkich komponentów losowych zwiększa odtwarzalność wyników
zgodnie z dokumentacją scikit-learn (Common pitfalls / Getting reproducible results).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Kaggle: yasserh/wine-quality-dataset → WineQT.csv (red + white, łącznie ~6497 wierszy)
DATA_PATH = PROJECT_ROOT / "data" / "WineQT.csv"

RESULTS_DIR = PROJECT_ROOT / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Kolumna identyfikatora — pomijana w uczeniu
ID_COLUMN = "Id"

# Zmienna docelowa: ocena jakości wina (wieloklasowa, typowo 3–8)
TARGET_COLUMN = "quality"

# 11 cech chemicznych (numeryczne)
FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

# Opisy klas do raportów (klucz = wartość quality)
CLASS_LABELS_PL = {
    3: "Ocena 3 (bardzo niska)",
    4: "Ocena 4 (niska)",
    5: "Ocena 5 (średnia)",
    6: "Ocena 6 (dobra)",
    7: "Ocena 7 (wysoka)",
    8: "Ocena 8 (bardzo wysoka)",
    9: "Ocena 9 (wyjątkowa)",
}
