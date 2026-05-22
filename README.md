# Klasyfikacja jakości wina — MLP vs Random Forest / XGBoost

## Krótki opis

- **Zbiór:** [Wine Quality](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset) (`data/WineQT.csv`) — **~6497** próbek, **11** cech chemicznych, **wieloklasowa** etykieta `quality` (oceny **3–9**).
- **Model główny:** sieć **MLP** (3 warianty architektury).
- **Modele porównawcze:** **Random Forest** × 3 oraz **XGBoost** × 3 (gdy biblioteka jest dostępna).
- **Walidacja:** stratyfikowana **5-fold** CV; **held-out test 20%** dla najlepszego MLP.
- **Metryki:** `accuracy`, `balanced_accuracy`, `f1_macro`.
- **Stack:** scikit-learn, pandas, Plotly, Streamlit, Jupyter.

## Czym jest zbiór?

Dane z badań chemicznych **wina czerwonego i białego** (UCI Wine Quality). Każdy wiersz to próbka opisana parametrami takimi jak kwasowość, zawartość alkoholu, siarczany czy gęstość. Zmienna **`quality`** to sensoryczna ocena jakości (skala dyskretna) — **nie jest to klasyfikacja binarna**.

## Wymagania

- Python 3.10+
- `pip install -r requirements.txt`
- **macOS (opcjonalnie):** `brew install libomp` — aby włączyć XGBoost w porównaniu.

## Dane

Plik `data/WineQT.csv` jest w repozytorium. Odtworzenie z UCI:

```bash
python scripts/pobierz_dane.py
```

## Eksperyment

```bash
python run_experiment.py
```

Zapisuje: `results/wyniki_*.csv`, `.tex`, `wykresy/*.html`, `macierz_pomylek_test.csv`, `wyniki_test_mlp.csv`.

## Streamlit i start

```bash
chmod +x start.sh   # jednorazowo
./start.sh          # macOS/Linux
start.bat           # Windows
```

Jeśli pojawi się `No module named pip`, skrypt **usuwa i tworzy na nowo** `.venv`. Możesz też ręcznie:

```bash
rm -rf .venv && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

→ `http://localhost:8501` — formuły, EDA, wyniki, druk (`Ctrl+P` / `Cmd+P`).

## Notebooki i dokumentacja

| Plik | Opis |
|------|------|
| `wine_quality_mlp_sklearn_project2.ipynb` | Raport: EDA, sklearn MLP, wnioski |
| `docs/wine_quality_mlp_pytorch_project2.ipynb` | Wariant MLP w PyTorch |
| `docs/dokumentacja_do125148.docx` | Sprawozdanie Word (aktualizacja treści i wykresów: `python docs/aktualizuj_dokumentacja.py`) |

## Brak przecieku

`StandardScaler` w `Pipeline` — dopasowanie **tylko na treningu** każdej foldy CV.

## Struktura

| Ścieżka | Opis |
|---------|------|
| `data/WineQT.csv` | Zbiór źródłowy |
| `src/config.py` | Kolumny, ścieżki, seed |
| `src/experiment.py` | Pipeline, modele, CV, eksport |
| `src/wykresy.py` | Wykresy Plotly |
| `streamlit_app.py` | Aplikacja WWW |
| `results/` | Wyniki generowane |

## Licencja

Zobacz `LICENSE`.
