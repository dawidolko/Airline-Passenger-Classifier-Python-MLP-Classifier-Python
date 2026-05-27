#!/usr/bin/env python3
"""
Skrypt tworzy/aktualizuje docs/dokumentacja_do125148.docx pod projekt
Airline Passenger Satisfaction. Wymaga: python-docx + wyniki w results/airline/.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.shared import Pt

ROOT = Path(__file__).resolve().parent.parent
DOCX_PATH = ROOT / "docs" / "dokumentacja_do125148.docx"
RESULTS_DIR = ROOT / "results" / "airline"


def pobierz_dane_raportu() -> dict:
    comparison = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    grid = pd.read_csv(RESULTS_DIR / "grid_search_results.csv").sort_values("best_val_f1_macro", ascending=False)
    meta = json.loads((RESULTS_DIR / "run_metadata.json").read_text(encoding="utf-8"))
    return {
        "comparison": comparison,
        "best_test": comparison.sort_values("f1_macro", ascending=False).iloc[0].to_dict(),
        "best_grid": grid.iloc[0].to_dict(),
        "meta": meta,
    }


def zaktualizuj_docx() -> None:
    data = pobierz_dane_raportu()
    comparison = data["comparison"]
    best_test = data["best_test"]
    best_grid = data["best_grid"]
    meta = data["meta"]

    doc = Document()
    style = doc.styles["Normal"]
    style.font.size = Pt(11)

    doc.add_heading("Sztuczna Inteligencja — Projekt 2", level=0)
    doc.add_paragraph(
        "Temat: Klasyfikacja klasy podróży pasażera linii lotniczych "
        "(Airline Passenger Satisfaction — multiclass)"
    )
    doc.add_paragraph("Autor: Dawid Olko, 125148")
    doc.add_paragraph("Informatyka II. Stopnia, I rok")
    doc.add_paragraph("Prowadzący: Dr inż. Jacek Bartman")
    doc.add_paragraph("Rok akademicki 2025/2026")

    doc.add_heading("1. Charakterystyka problemu", level=1)
    doc.add_paragraph(
        "Projekt dotyczy klasyfikacji wieloklasowej — na podstawie parametrów lotu "
        "i ocen usług pokładowych model ma przypisać pasażera do jednej z trzech klas "
        "podróży: Business, Eco lub Eco Plus."
    )
    doc.add_paragraph(
        "Zbiór danych pochodzi z Kaggle (Airline Passenger Satisfaction) i zawiera "
        "dane ankietowe pasażerów linii lotniczych. Problem jest niezbalansowany: "
        "klasa Eco Plus stanowi jedynie ~7% wszystkich rekordów."
    )
    doc.add_paragraph(
        "Źródło: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction"
    )

    doc.add_heading("2. Liczba instancji i atrybutów", level=1)
    doc.add_paragraph(
        f"Zbiór zawiera łącznie {meta['n_records']} rekordów "
        f"(po połączeniu train i test z Kaggle) i 25 kolumn."
    )
    doc.add_paragraph(
        f"Po preprocessingu i podziale stratyfikowanym: "
        f"train = {meta['n_train']}, validation = {meta['n_val']}, test = {meta['n_test']} "
        f"(70% / 15% / 15%)."
    )
    doc.add_paragraph(f"Liczba cech po transformacji: {meta['n_features']}.")
    doc.add_paragraph(
        "Cechy wejściowe (18 numerycznych + 3 kategoryczne):\n"
        "• Numeryczne: Age, Flight Distance, 14 ocen usług (0–5), "
        "Departure Delay in Minutes, Arrival Delay in Minutes.\n"
        "• Kategoryczne: Gender, Customer Type, Type of Travel.\n"
        "• Zmienna docelowa: Class (Business / Eco / Eco Plus)."
    )
    doc.add_paragraph(
        "Usunięte kolumny: Unnamed: 0 (indeks CSV), id (identyfikator), "
        "satisfaction (oryginalny binarny target — nie używamy)."
    )

    doc.add_heading("3. Preprocessing danych", level=1)
    doc.add_paragraph(
        "Etapy preprocessingu (moduł airline_project/preprocessing.py):"
    )
    doc.add_paragraph(
        "• Usunięcie wartości nierealnych (remove_unrealistic_values): "
        "Age poza 0–100, Flight Distance <= 0, opóźnienia < 0 lub > 1440 min, "
        "oceny usług poza 0–5 → zamiana na NaN."
    )
    doc.add_paragraph(
        "• Imputacja braków: mediana dla numerycznych, moda dla kategorycznych — "
        "fit wyłącznie na zbiorze treningowym (brak data leakage)."
    )
    doc.add_paragraph(
        "• OneHotEncoder dla cech kategorycznych (handle_unknown='ignore')."
    )
    doc.add_paragraph(
        "• StandardScaler dla cech numerycznych (z = (x - μ) / σ), "
        "dopasowany tylko na train."
    )

    doc.add_heading("4. Projekt modelu — MLP (PyTorch)", level=1)
    doc.add_paragraph(
        "Model główny to sieć neuronowa MLP (Multi-Layer Perceptron) zaimplementowana "
        "w PyTorch (klasa AirlineMLP w airline_project/model.py)."
    )
    doc.add_paragraph(
        "Architektura warstwy ukrytej: Linear → BatchNorm1d → ReLU → Dropout.\n"
        "• Linear — transformacja liniowa y = xW^T + b (uczone wagi i bias).\n"
        "• BatchNorm1d — normalizacja w obrębie batcha (stabilizacja treningu).\n"
        "• ReLU — aktywacja nieliniowa max(0, x).\n"
        "• Dropout — losowe zerowanie neuronów (regularyzacja)."
    )
    doc.add_paragraph(
        "Mechanizmy treningowe:\n"
        "• Optymalizator: AdamW (Adam z poprawioną regularyzacją L2).\n"
        "• Funkcja straty: CrossEntropyLoss z class weights "
        "(wyższy koszt błędu na klasie mniejszościowej Eco Plus).\n"
        "• Scheduler: ReduceLROnPlateau (zmniejsza LR gdy F1 stagnuje).\n"
        "• Early stopping: zatrzymanie po 5 epokach bez poprawy val macro F1."
    )
    doc.add_paragraph(
        f"Baseline: MLP 128→64, dropout=0.2, lr=1e-3.\n"
        f"Tuned (najlepsza z siatki): {meta['tuned_config']['name']}."
    )

    doc.add_heading("5. Grid search hiperparametrów", level=1)
    doc.add_paragraph(
        "Siatka: 3 architektury × 3 wartości dropout × 2 learning rate = 18 konfiguracji.\n"
        "Każda konfiguracja trenowana z early stopping; kryterium wyboru: "
        "walidacyjne macro F1."
    )
    doc.add_paragraph(
        f"Najlepsza konfiguracja: {best_grid['name']} "
        f"(val macro F1 = {best_grid['best_val_f1_macro']:.4f})."
    )

    doc.add_heading("6. Model referencyjny — Random Forest", level=1)
    doc.add_paragraph(
        "Random Forest z domyślnymi parametrami sklearn (100 drzew, bez class weights)."
    )

    doc.add_heading("7. Wyniki na zbiorze testowym", level=1)
    table = doc.add_table(rows=1, cols=5)
    table.style = "Table Grid"
    header = table.rows[0].cells
    header[0].text = "Model"
    header[1].text = "Accuracy"
    header[2].text = "Precision macro"
    header[3].text = "Recall macro"
    header[4].text = "F1 macro"
    for _, row in comparison.iterrows():
        cells = table.add_row().cells
        cells[0].text = str(row["model"])
        cells[1].text = f"{row['accuracy']:.4f}"
        cells[2].text = f"{row['precision_macro']:.4f}"
        cells[3].text = f"{row['recall_macro']:.4f}"
        cells[4].text = f"{row['f1_macro']:.4f}"

    doc.add_paragraph("")
    doc.add_paragraph(
        f"Najlepszy model pod kątem F1-macro: {best_test['model']} "
        f"(F1-macro = {best_test['f1_macro']:.4f}, accuracy = {best_test['accuracy']:.4f})."
    )

    doc.add_heading("8. Wnioski", level=1)
    doc.add_paragraph(
        "1) MLP tuned (512→256→128, dropout=0.3) osiąga najlepsze F1-macro (~0.653) — "
        "lepsze niż Random Forest (~0.606) i MLP baseline (~0.650)."
    )
    doc.add_paragraph(
        "2) Random Forest ma wyższą accuracy (~0.864), ale prawie całkowicie ignoruje "
        "klasę Eco Plus (recall ~1%). To pokazuje, że sama accuracy jest niewystarczającą "
        "metryką przy niezbalansowaniu klas."
    )
    doc.add_paragraph(
        "3) Class weights w CrossEntropyLoss skutecznie zwiększają recall na Eco Plus "
        "(~41% w MLP vs ~1% w RF), kosztem precision na dominujących klasach."
    )
    doc.add_paragraph(
        "4) Macierz pomyłek: Business i Eco rozpoznawane dobrze (precision > 80%); "
        "Eco Plus mylona głównie z Eco (podobne parametry lotu)."
    )
    doc.add_paragraph(
        "5) Ograniczenia projektu: mała klasa Eco Plus (~7%), brak zaawansowanej "
        "inżynierii cech, siatka ograniczona do 18 konfiguracji, dane ankietowe "
        "mogą nie odzwierciedlać rzeczywistych wzorców rezerwacji."
    )
    doc.add_paragraph(
        "6) Możliwe ulepszenia: RF z class_weight='balanced', SMOTE/oversampling, "
        "bayesowska optymalizacja hiperparametrów, feature engineering, ensemble."
    )

    doc.add_heading("9. Artefakty", level=1)
    doc.add_paragraph(
        "Wyniki i wykresy zapisano w results/airline/ oraz results/airline/plots/.\n"
        "Notebook: docs/airline_passenger_satisfaction_mlp_project2.ipynb.\n"
        "Dashboard Streamlit: streamlit run streamlit_app.py."
    )

    DOCX_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(DOCX_PATH)
    print(f"Zaktualizowano dokumentację: {DOCX_PATH}")


if __name__ == "__main__":
    zaktualizuj_docx()
