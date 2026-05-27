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
| Plik `data/train.csv` (Kaggle) | **103 904** rekordów |
| Plik `data/test.csv` (Kaggle) | **25 976** rekordów |
| **Razem (używane w projekcie)** | **129 880** rekordów |
| Po podziale 70/15/15 | train **90 915** / val **19 482** / test **19 483** |
| Target | `Class` (Business ~48%, Eco ~45%, **Eco Plus ~7%**) |

> **Uwaga:** „~105 tys.” to zwykle sam plik `train.csv` (~104k), a „~130 tys.” to **train + test** połączone. To nie są sprzeczne liczby — to **różne etapy** (pliki Kaggle vs podział w eksperymencie).

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

## FAQ — najważniejsze pytania (prosto)

> **Co to za czarny pasek w podglądzie README?**  
> W pliku masz linie z trzech myślników: `---`. W Markdown to **separator sekcji** (pozioma linia), nie wykres i nie wynik modelu. Np. między „train/val/test” a „wyciek danych”.

### Skąd liczby 129k, ~105k i 19 483? (żeby nie było chaosu)

| Co widzisz | Liczba | Co to znaczy |
|------------|--------|--------------|
| `data/train.csv` | **103 904** | Osobny plik z Kaggle (~104–105 tys.) — **nie** to samo co „train 70%” w modelu |
| `data/test.csv` | **25 976** | Drugi plik z Kaggle |
| **Całość w projekcie** | **129 880** | `train.csv` + `test.csv` **połączone**, potem jeden podział 70/15/15 |
| Train w modelu (70%) | **90 915** | Tu model **się uczy** |
| Val (15%) | **19 482** | Kontrola w trakcie treningu |
| Test końcowy (15%) | **19 483** | **Egzamin końcowy** — stąd `19483` w raportach |

**Nie ma błędu:** ~105k to plik Kaggle train, ~130k to suma obu plików, ~91k to train **wewnątrz** eksperymentu, ~19.5k to końcowy test.

### Raport na screenie (`satisfied` / `neutral or dissatisfied`) — dlaczego to myli?

Jeśli w dokumentacji widzisz klasy **`satisfied`** i **`neutral or dissatisfied`** zamiast **Business / Eco / Eco Plus**, to **stary wykres/screen** (oryginalny target Kaggle `satisfaction`), **nie** wynik tego projektu.

W tym projekcie raporty są dla **`Class`** — patrz `results/airline/classification_report_*.txt` lub sekcję „Raporty klasyfikacji” poniżej.

### Jak działa train / val / test? (krok po kroku)

**Tak — ogólnie dobrze myślisz**, tylko val to nie jest „test końcowy”, tylko **kontrola w trakcie nauki**.

```
Całość danych (po połączeniu train.csv + test.csv z Kaggle)
        │
        ├── 70% TRAIN   → model UCZY SIĘ (zmienia wagi)
        ├── 15% VAL     → kontrola PODCZAS uczenia (nie zmienia wag)
        └── 15% TEST    → egzamin KOŃCOWY (dopiero na końcu, raz)
```

**Kolejność w projekcie:**

1. **Preprocessing** — parametry (średnia, mediana, one-hot) liczone **tylko na train**, potem to samo stosowane na val i test.
2. **Grid search (18 wariantów MLP)** — każdy wariant:
   - uczy się na **train (70%)**,
   - co epokę sprawdzany na **val (15%)**,
   - wybieramy najlepszy wariant po **F1-macro na val**.
3. **Trening baseline i tuned** — znowu train + val (early stopping, scheduler LR).
4. **Dopiero na końcu** — ocena **baseline, tuned i Random Forest** na **test (15%)** → raporty, tabela w README, wykresy.

**Val (15%) służy do:**
- early stopping (kiedy przestać uczyć),
- zmniejszania learning rate (wykres `scheduler_lr.png`),
- wyboru najlepszej konfiguracji z grid search.

**Test (15%)** — model **nie był** na nim używany przy wyborze hiperparametrów. To uczciwy wynik końcowy (np. accuracy 0.76 / 0.77 w raporcie).

**Ważne:** val **nie zastępuje** testu. Val = wiele razy w trakcie treningu. Test = **jeden raz** na końcu.

---

### Co to jest wyciek danych (data leakage)? — rozpisane krok po kroku

**Wyciek danych** = model lub preprocessing **dostaje podpowiedź z testu albo z odpowiedzi** zanim zrobimy końcową ocenę. Wtedy wynik na teście jest **za wysoki** i **nie wiarygodny**.

**Analogia:** na egzaminie widziałeś pytania wcześniej — ocena nie oddaje realnej wiedzy.

---

#### Przykład na żywo: skalowanie wieku (`Age`)

Załóżmy, że na **train** średni wiek = 40, na **test** średni wiek = 60.

**Źle (wyciek):**
1. Liczysz średnią z **train + test** razem → np. 45.
2. Skalujesz wszystkie wiersze tą średnią.
3. Model „wie” pośrednio, że w teście są starsi pasażerowie (bo średnia 45 zawiera test).

**Dobrze (bez wycieku — tak jest u nas):**
1. **`fit` na train:** średnia = 40, odchylenie z train.
2. **`transform` na test:** wiek 60 → `(60 - 40) / odchylenie_train` — test nie zmienia średniej, tylko używa parametrów z train.

To samo dotyczy: mediany przy brakach, OneHot (jakie kategorie istnieją), grid search (który model wybrać).

---

#### Jak dane mogą wyciekać? (tabela)

| Błąd | Co robisz źle | Skutek |
|------|----------------|--------|
| Preprocessing na całości | `fit` na train+test+val razem | Test „wchodzi” do średniej, mediany, one-hot |
| Uczenie na teście | Wagi uczą się na 15% test | Test już nie jest niewidziany |
| Wybór modelu po teście | Grid search: „najlepszy” = najwyższy wynik na test | Test użyty do strojenia |
| `satisfaction` w cechach | Model widzi zadowolenie przy przewidywaniu `Class` | Podpowiedź (ściąganie) |
| `Class` w preprocessingu cech | Np. skalowanie z użyciem etykiety | Bezpośrednia odpowiedź w cechach |

---

#### Co robimy w tym projekcie (kolejność — bez wycieku)

```
1. Łączymy train.csv + test.csv Kaggle  →  129 880 wierszy
2. Dzielimy 70% / 15% / 15%             →  train / val / test
3. fit preprocessingu TYLKO na train   →  mediana, scaler, one-hot
4. transform na val i test             →  te same parametry, bez fit
5. Uczenie MLP na train                →  wagi z train
6. Kontrola na val                      →  early stopping, LR, grid search
7. JEDEN RAZ ocena na test              →  raporty (0.76, 0.77, 19483 próbek)
```

| Krok | Zbiór | Czy może być wyciek? | U nas |
|------|-------|----------------------|-------|
| Średnia wieku, mediana opóźnień | train only przy `fit` | Tak, jeśli liczysz z testem | **Nie** — tylko train |
| Skalowanie val/test | transform | Tak, jeśli znowu `fit` | **Nie** — tylko transform |
| Który MLP wybrać (grid) | val | Tak, jeśli po test | **Val** |
| Raport accuracy / F1 | test | Tak, jeśli wcześniej stroisz na test | **Test tylko na końcu** |
| Kolumna `satisfaction` | — | Tak, jako cecha | **Usunięta** |
| Target `Class` | tylko jako y | Tak, w cechach X | **Tylko y, nie w X** |

---

#### `fit` vs `transform` — najprościej

| Słowo | Po polsku | Kiedy |
|-------|-----------|--------|
| **fit** | „naucz parametry” | Tylko na **train (70%)** — np. mediana Age, średnia do skalera |
| **transform** | „zastosuj te parametry” | Na **val** i **test** — bez ponownego liczenia |

**Wyciek byłby:** `fit` na całym zbiorze albo na train+test.  
**U nas:** `fit_preprocessor(train_df)` w `preprocessing.py`, potem `transformuj_cechy(val/test, ...)`.

---

#### Gdzie w kodzie?

```text
preprocessing.py
  fit_preprocessor(train_df)     ← fit TYLKO train
  transformuj_cechy(val_df)      ← transform
  transformuj_cechy(test_df)    ← transform

experiment.py
  trenuj_mlp(..., x_train, y_train, x_val, y_val)   ← uczenie + val
  przewidz_mlp(..., x_test)                         ← test na końcu
```

**Na obronę (1 zdanie):**  
„Wyciek to gdy test lub odpowiedź wpływa na uczenie. U nas preprocessing fitujemy na train, model stroimy na val, a test używamy wyłącznie raz do końcowego raportu.”

---

### Dlaczego usuwamy `Unnamed: 0`, `id`, `satisfaction`?
- `Unnamed: 0` to techniczny numer wiersza z CSV, nie cecha.
- `id` to identyfikator pasażera, model nie powinien uczyć się numerów.
- `satisfaction` to inny target (zadowolony/niezadowolony). W tym projekcie target to `Class`.

### `satisfaction` (satisfied / neutral or dissatisfied) — czego nie używamy?

W CSV są **dwie osobne kolumny**:

| Kolumna | Znaczenie |
|---------|-----------|
| `satisfaction` | Czy pasażer zadowolony: `satisfied` lub `neutral or dissatisfied` |
| `Class` | Klasa lotu: `Business`, `Eco`, `Eco Plus` |

To **nie to samo** (można być zadowolonym w Eco albo niezadowolonym w Business).

- **Usuwamy** kolumnę `satisfaction` z cech wejściowych — model jej nie widzi.
- **Zostawiamy** `Class` jako odpowiedź (target) do nauki.
- Wiersze z `satisfied` i `neutral or dissatisfied` **zostają** — używamy ich pozostałych kolumn.
- `Class` **nie zastępuje** `satisfaction` — od początku przewidujemy klasę podróży, nie zadowolenie.

### Jak kodujemy cechy kategoryczne (3 → 6 kolumn)?

3 kolumny tekstowe: `Gender`, `Customer Type`, `Type of Travel`.

**OneHotEncoder** — każda wartość dostaje osobną kolumnę 0/1, np. `Gender_Male=1`, `Gender_Female=0`.

| Źródło | Wartości | Kolumn po kodowaniu |
|--------|----------|---------------------|
| Gender | 2 | 2 |
| Customer Type | 2 | 2 |
| Type of Travel | 2 | 2 |
| **Razem** | **3** | **6** |

Razem z 18 cechami numerycznymi → **24 cechy** wejściowe do modelu.

### Cechy numeryczne — czy są „zamieniane”?

**Nie** na kategorie. 18 kolumn zostaje liczbami: czyszczenie → uzupełnienie braków (mediana) → **StandardScaler** (skalowanie). One-hot dotyczy **tylko** kategorycznych.

### Dlaczego w tabeli accuracy 0.7633, a w raporcie 0.76?

To **ta sama wartość**, tylko inne zaokrąglenie:

- `model_comparison.csv` — pełna liczba (np. 0.7633),
- `classification_report_*.txt` — zaokrąglenie do 2 miejsc (0.76).

Nie ma sprzeczności — raport jest bardziej „płaski” wizualnie.

### Co znaczy `accuracy` z liczbą 19483 w raporcie?

- **19483** = liczba wszystkich próbek na **teście** (support łącznie),
- **accuracy** (np. 0.76) = **jedna** metryka: jaki % wszystkich trafień na 3 klasach łącznie.

Per klasa są osobne wiersze (Business, Eco, Eco Plus). To klasyfikacja **3-klasowa**, nie binarna (2 wartości).

### Co pokazuje wykres `scheduler_lr.png`?

- **LR (learning rate)** = wielkość kroku przy aktualizacji wag.
- Na początku LR większy (szybsza nauka).
- Gdy **F1 na val** przestaje rosnąć → `ReduceLROnPlateau` obniża LR (schodki w dół na wykresie).
- Sens: najpierw szybko, potem dokładniej „dostrajać” model.

### Co znaczy „fit na train, transform na val/test”?
- **fit** = policz parametry preprocessingu (średnia, odchylenie, mediana, słownik kategorii) na train.
- **transform** = zastosuj te same parametry do val/test.
- Dzięki temu nie ma przecieku danych (model nie podgląda testu).

### Po co `handle_unknown='ignore'`?
Gdy w val/test pojawi się kategoria, której nie było na train, kod się nie wysypie.

### Dlaczego 70/15/15 zamiast 80/20?
- 80/20 jest OK, gdy tylko trenujesz i testujesz.
- Tu stroimy model (grid search + early stopping), więc potrzebny jest osobny **validation set**.
- Dlatego: 70% train, 15% validation, 15% test.

### Czemu w plikach widzę `test.csv` ~25k, a w wynikach test ~19.5k?
Bo łączymy `data/train.csv` + `data/test.csv` Kaggle, a potem robimy własny podział 70/15/15.
Finalny test to 15% całości, czyli ok. 19 483.

### Co to znaczy `output_dim=3`?
Mamy 3 klasy targetu `Class`: `Business`, `Eco`, `Eco Plus`. Dlatego ostatnia warstwa ma 3 neurony.

### Co robią bloki `Linear -> BatchNorm -> ReLU -> Dropout`?
- `Linear`: liczy ważone sumy cech.
- `BatchNorm`: stabilizuje wartości między warstwami.
- `ReLU`: dodaje nieliniowość (model staje się „mądrzejszy” niż prosta linia).
- `Dropout`: losowo wyłącza część neuronów podczas treningu, żeby model się nie przeuczał.

### Co daje więcej warstw i neuronów?
Większa sieć może nauczyć się trudniejszych zależności, ale łatwiej się przeucza. Dlatego testujemy kilka wariantów i wybieramy najlepszy.

### Co to jest grid search?
Automatyczne sprawdzanie wielu kombinacji parametrów (architektura, dropout, learning rate) i wybór najlepszej po wyniku walidacyjnym.

### Co to jest CrossEntropyLoss?
Funkcja straty dla klasyfikacji wieloklasowej. Karze model, gdy nisko ocenia prawdziwą klasę.

### Co to są class weights?
Rzadka klasa (`Eco Plus`) dostaje większą karę za błąd, więc model bardziej się na niej skupia.

### Co to jest F1-macro?
F1 liczone osobno dla każdej klasy, potem średnia. Każda klasa ma taką samą wagę.

### Co to jest SMOTE / oversampling?
- **Oversampling**: zwiększanie liczby próbek klasy mniejszościowej.
- **SMOTE**: tworzenie nowych, sztucznych próbek tej klasy.
W tym projekcie nie używamy tego (zgodnie z wymaganiem) — używamy class weights.

### Czym różnią się metryki?
- `accuracy`: procent wszystkich trafień.
- `precision_macro`: jak często model ma rację, gdy wskazuje klasę.
- `recall_macro`: ile prawdziwych przypadków klasy wykrył.
- `f1_macro`: kompromis precision/recall, średnio po klasach.

### Co znaczy `model.eval()`?
Tryb oceny modelu: dropout jest wyłączony, a batchnorm działa stabilnie.

### Co to jest AdamW, ReduceLROnPlateau, early stopping i epoka?
- **AdamW**: algorytm aktualizacji wag.
- **ReduceLROnPlateau**: zmniejsza learning rate, gdy wynik stoi w miejscu.
- **Early stopping**: kończy trening, gdy brak poprawy przez kilka epok.
- **Epoka**: jedno pełne przejście przez cały zbiór treningowy.

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
| MLP baseline | 0.7633 | 0.6462 | 0.6666 | 0.6390 |
| **MLP tuned** | 0.7750 | 0.6497 | 0.6705 | **0.6465** |
| Random Forest | **0.8644** | 0.6791 | 0.6248 | 0.6058 |

**Najlepsza konfiguracja:** MLP 512×256×128, dropout=0.3, lr=1e-3

---

## Raporty Klasyfikacji (dokładne)

Poniżej są dokładne raporty per klasa dla 3 modeli.

### MLP baseline

```text
              precision    recall  f1-score   support

    Business       0.94      0.85      0.89      9325
         Eco       0.81      0.73      0.77      8747
    Eco Plus       0.18      0.42      0.26      1411

    accuracy                           0.76     19483
   macro avg       0.65      0.67      0.64     19483
weighted avg       0.83      0.76      0.79     19483
```

### MLP tuned

```text
              precision    recall  f1-score   support

    Business       0.94      0.85      0.89      9325
         Eco       0.81      0.75      0.78      8747
    Eco Plus       0.19      0.41      0.26      1411

    accuracy                           0.77     19483
   macro avg       0.65      0.67      0.65     19483
weighted avg       0.83      0.77      0.80     19483
```

### Random Forest

```text
              precision    recall  f1-score   support

    Business       0.94      0.91      0.93      9325
         Eco       0.80      0.95      0.87      8747
    Eco Plus       0.30      0.01      0.02      1411

    accuracy                           0.86     19483
   macro avg       0.68      0.62      0.61     19483
weighted avg       0.83      0.86      0.84     19483
```

### Co te wyniki mówią (prosto)?
- `Business`: wszystkie modele działają bardzo dobrze.
- `Eco`: Random Forest ma bardzo wysokie `recall` (0.95), ale odbywa się to kosztem klasy `Eco Plus`.
- `Eco Plus`: MLP (recall ~0.41-0.42) wykrywa tę klasę dużo lepiej niż RF (recall 0.01).
- Dlatego RF ma wyższą `accuracy`, ale gorszy `F1-macro` od MLP tuned.

---

## Jak czytać wykres LR (scheduler_lr.png)?

- LR to `learning rate`, czyli wielkość kroku aktualizacji wag.
- Na początku LR jest większy, żeby szybciej uczyć model.
- Gdy walidacyjne `F1-macro` przestaje rosnąć, `ReduceLROnPlateau` zmniejsza LR (zwykle o połowę).
- Na wykresie widać „schodki” w dół — to normalne i pożądane.
- Sens: duży krok na początku, mniejszy krok na końcu = stabilniejsze „dostrajanie” modelu.

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
├── data/
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
