# Airline Passenger Satisfaction — MLP (PyTorch)

Projekt klasyfikacji wieloklasowej `Class` (Business / Eco / Eco Plus) na zbiorze
**Airline Passenger Satisfaction** z Kaggle. Model główny: sieć neuronowa MLP
zaimplementowana od zera w PyTorch. Model referencyjny: Random Forest (sklearn).

---

## Wymagania

- Python 3.10+
- Pakiety: `requirements.txt`

```bash
pip install -r requirements.txt
```

---

## Szybkie uruchomienie

```bash
# sam eksperyment (trening + ewaluacja + wykresy):
python3 run_experiment.py

# dashboard Streamlit:
streamlit run streamlit_app.py
```

Skrypty jednolinijkowe:
- macOS/Linux: `./start.sh`
- Windows: `start.bat`

---

## Dane

| Element | Wartość |
|---------|--------|
| Źródło | [Kaggle — Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) |
| Pliki lokalne | `data/train.csv`, `data/test.csv` |
| Rekordy | ~129 880 (train + test Kaggle) |
| Target | `Class` (Business ~48%, Eco ~45%, **Eco Plus ~7%**) |

### Usuwane kolumny
- `Unnamed: 0` — sztuczny indeks CSV
- `id` — identyfikator pasażera
- `satisfaction` — oryginalny binarny target (nie używamy go)

### Cechy (18 numerycznych + 3 kategoryczne)
- **Numeryczne:** Age, Flight Distance, 14 ocen usług (skala 0–5), Departure Delay, Arrival Delay
- **Kategoryczne:** Gender, Customer Type, Type of Travel

---

## Preprocessing (`airline_project/preprocessing.py`)

1. **Usuwanie wartości nierealnych** (`remove_unrealistic_values`):
   - Age < 0 lub > 100 → NaN
   - Flight Distance ≤ 0 → NaN
   - Opóźnienia < 0 lub > 1440 min → NaN
   - Oceny usług < 0 lub > 5 → NaN
2. **Imputacja braków:**
   - Mediana (numeryczne), moda (kategoryczne)
   - Fit **wyłącznie na train** — bez data leakage
3. **OneHotEncoder** — kategoryczne → kolumny binarne
4. **StandardScaler** — numeryczne → z-score (fit na train)
5. **Podział:** 70% train / 15% val / 15% test, stratified po target

---

## Model MLP (`airline_project/model.py`)

### Architektura warstwy ukrytej

```
Linear → BatchNorm1d → ReLU → Dropout
```

- **Linear** — transformacja liniowa (uczone wagi + bias)
- **BatchNorm1d** — normalizacja w batchu (stabilizacja treningu)
- **ReLU** — aktywacja nieliniowa max(0, x)
- **Dropout** — regularyzacja (losowe zerowanie neuronów)

### Elementy uczenia

| Element | Opis |
|---------|------|
| Optimizer | AdamW (weight_decay=1e-4) |
| Loss | CrossEntropyLoss z **class weights** |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early stopping | Walidacyjne macro F1 (patience=5, min_delta=1e-4) |
| Batch size | 1024 |
| GPU | Automatyczna detekcja CUDA / MPS / CPU |

### Grid search (18 konfiguracji)

| Parametr | Wartości |
|----------|----------|
| Architektura | (128,64), (256,128,64), (512,256,128) |
| Dropout | 0.1, 0.2, 0.3 |
| Learning rate | 1e-3, 5e-4 |

Baseline: (128,64), dropout=0.2, lr=1e-3. Tuned: najlepsza z siatki.

---

## Wyniki na zbiorze testowym (19 483 próbki)

| Model | Accuracy | Precision macro | Recall macro | F1-macro |
|-------|----------|-----------------|--------------|----------|
| MLP baseline | 0.7731 | 0.6581 | 0.6822 | 0.6504 |
| **MLP tuned** | 0.7825 | 0.6571 | 0.6773 | **0.6529** |
| Random Forest | **0.8644** | 0.6791 | 0.6248 | 0.6058 |

**Najlepsza konfiguracja:** MLP 512×256×128, dropout=0.3, lr=1e-3

---

## Wnioski

1. **MLP tuned wygrywa F1-macro** — lepiej balansuje klasy dzięki class weights.
2. **RF wygrywa accuracy**, ale niemal ignoruje Eco Plus (recall ~1%).
3. **Eco Plus** to klasa problematyczna (~7%) — MLP osiąga recall ~41%.
4. **Class weights w loss** są kluczowe dla wykrywania mniejszościowej klasy.
5. **Ograniczenia:** mała klasa Eco Plus, brak feature engineering, grid search 18 konfiguracji.

---

## Struktura projektu

```
├── airline_project/
│   ├── __init__.py
│   ├── config.py          # konfiguracja (ścieżki, hiperparametry, cechy)
│   ├── model.py           # klasa AirlineMLP + wybór urządzenia
│   ├── preprocessing.py   # czyszczenie, podział, imputacja, skalowanie
│   └── experiment.py      # grid search, trening, ewaluacja, wykresy
├── data/archive (2)/
│   ├── train.csv
│   └── test.csv
├── results/airline/
│   ├── model_comparison.csv
│   ├── grid_search_results.csv
│   ├── classification_report_*.txt
│   ├── run_metadata.json
│   └── plots/
│       ├── confusion_matrices_side_by_side.png
│       ├── loss_and_f1_history.png
│       └── scheduler_lr.png
├── docs/
│   ├── airline_passenger_satisfaction_mlp_project2.ipynb
│   ├── dokumentacja_do125148.docx
│   └── aktualizuj_dokumentacja.py
├── run_experiment.py      # punkt wejścia
├── streamlit_app.py       # dashboard Streamlit
├── requirements.txt
├── start.sh / start.bat
└── README.md
```

---

## Dashboard Streamlit

```bash
streamlit run streamlit_app.py
```

Widoki: Podsumowanie wyników, Raporty klasyfikacji, Wykresy, Lista plików.
Przycisk „Uruchom pełny eksperyment" pozwala uruchomić trening z poziomu UI.
