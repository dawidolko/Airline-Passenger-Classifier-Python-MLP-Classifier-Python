#!/usr/bin/env python3
"""Generuje rozbudowane notebooki projektu 2 (Wine Quality)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [ln + "\n" for ln in text.strip().split("\n")]}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [ln + "\n" for ln in text.strip().split("\n")],
    }


def nb(cells: list) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }


SKLEARN_CELLS = [
    md("""
# Projekt 2 — Klasyfikacja jakości wina (Wine Quality)

**Autor:** Olko Dawid · **Przedmiot:** Sztuczna inteligencja

## Cel notebooka

Ten notebook prowadzi krok po kroku przez:
1. wczytanie i **eksplorację** zbioru `WineQT.csv`,
2. uruchomienie eksperymentu z modułu `src/experiment.py` (MLP + Random Forest + opcjonalnie XGBoost),
3. **wizualizację wyników** walidacji krzyżowej i macierzy pomyłek,
4. **wnioski** merytoryczne pod sprawozdanie.

**Zbiór:** ~6497 próbek wina, **11 cech chemicznych**, etykieta **`quality`** (oceny 3–9) — klasyfikacja **wieloklasowa**, nie binarna.

> Uruchom komórki **od góry do dołu** (`Run All`). Pierwsze uruchomienie eksperymentu może potrwać 1–3 minuty.
"""),
    md("""
## 0. Konfiguracja środowiska

Importujemy biblioteki i ustawiamy ścieżkę do katalogu projektu (tam jest folder `src/`).
Wykresy wyświetlą się **w notebooku** dzięki `%matplotlib inline`.
"""),
    code("""
%matplotlib inline

import warnings
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["figure.dpi"] = 100

# Katalog projektu (notebook w root lub w docs/)
ROOT = Path.cwd()
if not (ROOT / "src").is_dir() and (ROOT.parent / "src").is_dir():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    DATA_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    ID_COLUMN,
    RESULTS_DIR,
    RANDOM_STATE,
)
from src.experiment import wczytaj_pelna_ramke, podziel_na_cechy_i_etykiete, uruchom_pelny_eksperyment

print("Katalog projektu:", ROOT.resolve())
print("Plik danych:", DATA_PATH.resolve())
print("Istnieje:", DATA_PATH.is_file())
"""),
    md("""
## 1. Wczytanie danych

Funkcja `wczytaj_pelna_ramke`:
- wczytuje CSV,
- sprawdza brakujące wartości,
- weryfikuje obecność 11 cech i kolumny `quality`.

Kolumna **`Id`** to tylko identyfikator wiersza — **nie** trafia do modelu.
"""),
    code("""
df = wczytaj_pelna_ramke(DATA_PATH)

print(f"Liczba rekordów: {len(df)}")
print(f"Liczba kolumn: {df.shape[1]}")
print(f"Zakres ocen quality: {df[TARGET_COLUMN].min()} – {df[TARGET_COLUMN].max()}")
print()
display(df[FEATURE_COLUMNS + [TARGET_COLUMN]].head(8))
"""),
    md("""
### 1.1 Rozkład klas (`quality`)

Wino ma **nierówny rozkład ocen** — najwięcej próbek z oceną **5** i **6**.  
Przy takim rozkładzie sama **accuracy** może być wysoka, jeśli model „zawsze” przewiduje klasę 5 lub 6. Dlatego w projekcie liczymy też **balanced accuracy** i **F1-macro**.
"""),
    code("""
vc = df[TARGET_COLUMN].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(vc.index.astype(str), vc.values, color=sns.color_palette("Blues", len(vc)))
ax.set_xlabel("Ocena jakości (quality)")
ax.set_ylabel("Liczba próbek")
ax.set_title("Rozkład klas — Wine Quality")
for b, v in zip(bars, vc.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 30, str(v), ha="center", fontsize=9)
plt.tight_layout()
plt.show()

print(vc)
print(f"\\nNajczęstsza klasa: {vc.idxmax()} ({vc.max()} próbek, {100*vc.max()/len(df):.1f}%)")
"""),
    md("""
### 1.2 Statystyki opisowe cech

Wszystkie cechy wejściowe są **numeryczne** (parametry chemiczne). Różne kolumny mają **różne skale** (np. `density` ~1, `total sulfur dioxide` do setek) — stąd w modelu MLP stosujemy **StandardScaler** wewnątrz pipeline.
"""),
    code("""
desc = df[FEATURE_COLUMNS].describe().T
desc["brak_NaN"] = df[FEATURE_COLUMNS].isna().sum()
display(desc.round(3))
"""),
    md("""
### 1.3 Macierz korelacji (Pearson)

Mapa korelacji pokazuje, które cechy są ze sobą powiązane. Wysoka |korelacja| (>0,7) sugeruje redundancję — w opcjonalnej selekcji cech można rozważyć usunięcie jednej z pary.
"""),
    code("""
corr = df[FEATURE_COLUMNS].corr()

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, annot=False)
ax.set_title("Macierz korelacji — 11 cech chemicznych")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Pary z |r| > 0.7 (informacyjnie)
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
pary = [(i, j, upper.loc[j, i]) for i in upper.index for j in upper.columns if pd.notna(upper.loc[j, i]) and abs(upper.loc[j, i]) > 0.7]
if pary:
    print("Pary cech z |korelacja| > 0,7:")
    for a, b, r in sorted(pary, key=lambda x: -abs(x[2])):
        print(f"  {a} — {b}: {r:.3f}")
else:
    print("Brak par z |korelacja| > 0,7.")
"""),
    md("""
### 1.4 Zależność cech od jakości (przykłady)

Sprawdzamy, czy **alkohol** i **kwasowość lotna** różnią się między ocenami — to typowe predyktory jakości wina w literaturze.
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.boxplot(data=df, x=TARGET_COLUMN, y="alcohol", ax=axes[0], palette="Blues")
axes[0].set_title("Alkohol vs ocena jakości")
axes[0].set_xlabel("quality")

sns.boxplot(data=df, x=TARGET_COLUMN, y="volatile acidity", ax=axes[1], palette="Oranges")
axes[1].set_title("Kwasowość lotna vs ocena jakości")
axes[1].set_xlabel("quality")

plt.tight_layout()
plt.show()
"""),
    md("""
## 2. Eksperyment — modele i walidacja krzyżowa

Logika w `src/experiment.py`:
| Krok | Co robi |
|------|---------|
| Pipeline | `StandardScaler` → klasyfikator (bez przecieku: skaler tylko na treningu foldy) |
| Modele | **MLP** ×3 (główny), **Random Forest** ×3, **XGBoost** ×3 jeśli dostępny |
| CV | Stratyfikowana **5-fold**, `random_state=42` |
| Metryki | accuracy, balanced accuracy, F1-macro |
| Test | Najlepszy MLP (wg balanced accuracy w CV) na **20%** hold-out |

Ustaw `URUCHOM_OD_ZERA = False`, aby wczytać gotowe wyniki z `results/` (szybciej).  
Ustaw `True`, aby przeliczyć wszystko od nowa (jak `python run_experiment.py`).
"""),
    code("""
URUCHOM_OD_ZERA = False  # True = przelicz CV od zera (1–3 min); False = wczytaj results/

csv_sz = RESULTS_DIR / "wyniki_szczegolowe.csv"
csv_gr = RESULTS_DIR / "wyniki_agregaty_rodzin.csv"

if not URUCHOM_OD_ZERA and csv_sz.is_file() and csv_gr.is_file():
    print("Wczytuję zapisane wyniki z results/")
    szczegoly = pd.read_csv(csv_sz)
    grupy = pd.read_csv(csv_gr)
    # Macierz pomyłek — szybka ocena najlepszego MLP
    from src.experiment import wybierz_najlepszy_mlp, ocena_na_tescie
    X, y, nazwy = podziel_na_cechy_i_etykiete(df)
    wynik_test = ocena_na_tescie(X, y, nazwy, wybierz_najlepszy_mlp(szczegoly))
else:
    print("Uruchamiam pełny eksperyment (CV + test MLP)...")
    szczegoly, grupy, wynik_test = uruchom_pelny_eksperyment()

print("\\n--- Najlepszy MLP (CV) ---")
from src.experiment import wybierz_najlepszy_mlp
print("Wariant:", wybierz_najlepszy_mlp(szczegoly))
print(f"Test accuracy: {wynik_test['accuracy']:.4f}")
print(f"Test balanced accuracy: {wynik_test['balanced_accuracy']:.4f}")
print(f"Test F1-macro: {wynik_test['f1_macro']:.4f}")
"""),
    md("""
### 2.1 Tabela wyników — wszystkie warianty modeli

Kolumny `*_mean` i `*_std` to średnia i odchylenie standardowe z **5 foldów** walidacji krzyżowej.
"""),
    code("""
display(szczegoly.round(4))
display(grupy.round(4))
"""),
    md("""
### 2.2 Wykres — porównanie accuracy (CV)

Słupki: średnia accuracy; **linie** (errorbar): odchylenie między foldami.
"""),
    code("""
df_plot = szczegoly.copy()
df_plot["_sort"] = df_plot["rodzina"].map({"MLP": 0, "RandomForest": 1, "XGBoost": 2}).fillna(9)
df_plot = df_plot.sort_values(["_sort", "model"])
kolory = {"MLP": "#E84855", "RandomForest": "#1B998B", "XGBoost": "#5C4D7D"}

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, metryka, std, tyt in zip(
    axes,
    ["accuracy_mean", "balanced_accuracy_mean", "f1_macro_mean"],
    ["accuracy_std", "balanced_accuracy_std", "f1_macro_std"],
    ["Accuracy", "Balanced accuracy", "F1-macro"],
):
    kol = [kolory.get(r, "#888") for r in df_plot["rodzina"]]
    ax.bar(range(len(df_plot)), df_plot[metryka], yerr=df_plot[std], capsize=3, color=kol, ecolor="gray")
    ax.set_xticks(range(len(df_plot)))
    ax.set_xticklabels(df_plot["model"], rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{tyt} (5-fold CV)")
    ax.set_ylabel("Średnia ± std")

plt.suptitle("Porównanie modeli — Wine Quality", y=1.02)
plt.tight_layout()
plt.show()
"""),
    md("""
### 2.3 Agregaty po rodzinie modelu

Dla każdej rodziny (MLP / RF / XGB) pokazujemy **średnią** i **maksymalną** wartość metryki spośród 3 wariantów hiperparametrów.
"""),
    code("""
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(grupy))
w = 0.35
ax.bar(x - w/2, grupy["accuracy_srednia"], w, label="Śr. accuracy", color="#5C4D7D")
ax.bar(x + w/2, grupy["accuracy_max"], w, label="Maks. accuracy", color="#8F7EAF")
ax.set_xticks(x)
ax.set_xticklabels(grupy["rodzina"])
ax.set_ylim(0, 1)
ax.legend()
ax.set_title("Accuracy — średnia vs maksimum w rodzinie modelu")
plt.tight_layout()
plt.show()
"""),
    md("""
## 3. Macierz pomyłek — najlepszy MLP na zbiorze testowym (20%)

Wiersze = **prawdziwa** ocena, kolumny = **predykcja**.  
Diagonalne komórki to trafne klasyfikacje. Błędy „obok” (np. 5↔6) są typowe przy podobnych klasach sensorycznych.
"""),
    code("""
from sklearn.metrics import ConfusionMatrixDisplay

cm = wynik_test["confusion_matrix"]
etykiety = wynik_test["labels"]

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=etykiety).plot(
    ax=ax, cmap="Blues", colorbar=True
)
ax.set_title(f"Macierz pomyłek — {wynik_test['model']} (held-out test)")
plt.tight_layout()
plt.show()

cm_df = pd.DataFrame(cm, index=[f"prawda:{e}" for e in etykiety], columns=[f"pred:{e}" for e in etykiety])
display(cm_df)
"""),
    md("""
## 4. Wnioski (do sprawozdania)

### Wyniki
- **Random Forest** (`rf_300`) osiąga zwykle **najlepszą accuracy** w CV (~0,69) i **F1-macro** (~0,40).
- **MLP** (`mlp_128_64_32` lub `mlp_128_64`) daje ~**0,61–0,62** accuracy w CV i ~**0,63** na teście.
- **Balanced accuracy** jest **niższa** (~0,35) niż accuracy — efekt **niezbalansowania** klas 5 i 6.

### Metodologia
- Pipeline ze **skalowaniem** jest poprawny (brak przecieku do CV).
- Porównanie **wielu wariantów** hiperparametrów w ramach MLP i RF jest zgodne z wytycznymi projektu 2.

### Ograniczenia i dalsze kroki
- Dokładne przewidywanie skrajnych ocen (3, 4, 8, 9) jest trudne przy małej liczbie próbek tych klas.
- Możliwe ulepszenia: **wagowanie klas**, grupowanie ocen (np. niska/średnia/wysoka), tuning MLP, **XGBoost** po `brew install libomp` (macOS).

### Powiązane pliki
- `run_experiment.py`, `streamlit run streamlit_app.py`, `results/wyniki_szczegolowe.csv`
"""),
]

PYTORCH_CELLS = [
    md("""
# Projekt 2 — MLP w PyTorch (Wine Quality)

Uzupełnienie do notebooka scikit-learn: ta sama baza **`WineQT.csv`**, ale sieć neuronowa zaimplementowana w **PyTorch** (warstwy `Linear`, `ReLU`, `CrossEntropyLoss`).

**Wymaganie:** `pip install torch` (w aktywnym venv projektu).

## Plan
1. Przygotowanie danych (ten sam podział 80/20, stratyfikacja).
2. Definicja i trening MLP.
3. Wykres funkcji straty w epokach.
4. Macierz pomyłek i metryki na zbiorze testowym.
5. Porównanie z wynikami sklearn z `results/`.
"""),
    code("""
# Jeśli brak PyTorch: odkomentuj następną linię w terminalu i uruchom komórkę ponownie
# !pip install torch

%matplotlib inline

import warnings
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:
    raise ImportError("Zainstaluj PyTorch: pip install torch") from e

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
    classification_report,
)

ROOT = Path.cwd()
if not (ROOT / "src").is_dir() and (ROOT.parent / "src").is_dir():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN, RANDOM_STATE, TEST_SIZE, RESULTS_DIR
from src.experiment import wczytaj_pelna_ramke, podziel_na_cechy_i_etykiete

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch:", torch.__version__)
print("Urządzenie:", DEVICE)
"""),
    md("""
## 1. Dane treningowe i testowe

- **X** — 11 cech chemicznych (bez `Id`).
- **y** — ocena `quality`; w PyTorch mapujemy na indeksy 0…K−1.
- **StandardScaler** dopasowany **tylko na train** (jak w pipeline sklearn).
"""),
    code("""
df = wczytaj_pelna_ramke(DATA_PATH)
X, y, _ = podziel_na_cechy_i_etykiete(df)

classes = sorted(np.unique(y))
class_to_idx = {c: i for i, c in enumerate(classes)}
y_idx = np.array([class_to_idx[v] for v in y])

X_train, X_test, y_train, y_test = train_test_split(
    X, y_idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_idx
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

n_features = X_train_s.shape[1]
n_classes = len(classes)
print(f"Train: {len(y_train)}, Test: {len(y_test)}, Cechy: {n_features}, Klasy: {classes}")
"""),
    md("""
## 2. Architektura sieci MLP

Prosta sieć feed-forward: **128 → 64 → K klas**, aktywacja ReLU, dropout 0,2 po pierwszej warstwie ukrytej (ograniczenie przeuczenia).
"""),
    code("""
class WineMLP(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_out),
        )

    def forward(self, x):
        return self.net(x)


model = WineMLP(n_features, n_classes).to(DEVICE)
print(model)
"""),
    md("""
## 3. Trening (40 epok)

Używamy **CrossEntropyLoss** (wieloklasowa klasyfikacja) i optymizatora **Adam**.  
Zapisujemy stratę w każdej epoce — wykres pokaże, czy model się stabilizuje.
"""),
    code("""
BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-3

train_ds = TensorDataset(
    torch.tensor(X_train_s, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long),
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

historia_loss = []

model.train()
for ep in range(EPOCHS):
    ep_loss = 0.0
    n = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        ep_loss += loss.item() * len(yb)
        n += len(yb)
    historia_loss.append(ep_loss / n)
    if (ep + 1) % 10 == 0:
        print(f"Epoka {ep+1}/{EPOCHS}, loss={historia_loss[-1]:.4f}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, EPOCHS + 1), historia_loss, marker="o", markersize=3)
ax.set_xlabel("Epoka")
ax.set_ylabel("Cross-entropy (średnia na batch)")
ax.set_title("Krzywa uczenia — MLP PyTorch")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""),
    md("""
## 4. Ewaluacja na zbiorze testowym (20%)
"""),
    code("""
model.eval()
with torch.no_grad():
    logits = model(torch.tensor(X_test_s, dtype=torch.float32).to(DEVICE))
    y_pred = logits.argmax(1).cpu().numpy()

acc = accuracy_score(y_test, y_pred)
ba = balanced_accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

print(f"Accuracy:          {acc:.4f}")
print(f"Balanced accuracy: {ba:.4f}")
print(f"F1-macro:          {f1:.4f}")
print()
print(classification_report(y_test, y_pred, target_names=[str(c) for c in classes], zero_division=0))
"""),
    code("""
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=[str(c) for c in classes]
).plot(ax=ax, cmap="Purples", colorbar=True)
ax.set_title("Macierz pomyłek — MLP PyTorch (test 20%)")
plt.tight_layout()
plt.show()
"""),
    md("""
## 5. Porównanie z MLP (scikit-learn) z pliku wyników

Jeśli wcześniej uruchomiono `python run_experiment.py`, można porównać PyTorch z najlepszym wariantem **sklearn MLP** na tym samym typie podziału (oba: 80/20, seed 42, skalowanie).
"""),
    code("""
test_csv = RESULTS_DIR / "wyniki_test_mlp.csv"
if test_csv.is_file():
    sk = pd.read_csv(test_csv).iloc[0]
    print("sklearn MLP (z results/wyniki_test_mlp.csv):")
    print(f"  model: {sk['model']}")
    print(f"  accuracy_test: {sk['accuracy_test']:.4f}")
    print(f"  balanced_accuracy_test: {sk['balanced_accuracy_test']:.4f}")
    print(f"  f1_macro_test: {sk['f1_macro_test']:.4f}")
    print()
    print("PyTorch MLP (ten notebook):")
    print(f"  accuracy_test: {acc:.4f}")
    print(f"  balanced_accuracy_test: {ba:.4f}")
    print(f"  f1_macro_test: {f1:.4f}")
else:
    print("Brak results/wyniki_test_mlp.csv — uruchom: python run_experiment.py")
"""),
    md("""
## 6. Wnioski (PyTorch)

- Implementacja **PyTorch** pozwala śledzić **krzywą straty** i swobodnie modyfikować architekturę.
- Wyniki są **porównywalne rzędu wielkości** ze sklearn MLP — różnice wynikają z innej inicjalizacji, braku early stopping i innej implementacji optimizerów.
- Przy **niezbalansowanych** klasach warto rozważyć `weight` w `CrossEntropyLoss` lub metryki z wagami klas.
- Do sprawozdania główne wyniki bierz z notebooka **scikit-learn** i `src/experiment.py` (oficjalny pipeline projektu).
"""),
]


def main() -> None:
    paths = [
        ROOT / "wine_quality_mlp_sklearn_project2.ipynb",
        ROOT / "docs" / "wine_quality_mlp_sklearn_project2.ipynb",
        ROOT / "docs" / "wine_quality_mlp_pytorch_project2.ipynb",
    ]
    payloads = [
        nb(SKLEARN_CELLS),
        nb(SKLEARN_CELLS),
        nb(PYTORCH_CELLS),
    ]
    for path, payload in zip(paths, payloads):
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
        print("Zapisano:", path, f"({len(payload['cells'])} komórek)")


if __name__ == "__main__":
    main()
