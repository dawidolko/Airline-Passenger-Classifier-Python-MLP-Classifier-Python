#!/usr/bin/env python3
"""
Dashboard Streamlit dla projektu Airline Passenger Satisfaction.
Wyświetla wyniki eksperymentu MLP (PyTorch) vs Random Forest.
Tekst interfejsu: polski.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from airline_project.experiment import uruchom_pelny_eksperyment

RESULTS_DIR = ROOT / "results" / "airline"
PLOTS_DIR = RESULTS_DIR / "plots"
MODEL_CSV = RESULTS_DIR / "model_comparison.csv"
GRID_CSV = RESULTS_DIR / "grid_search_results.csv"
META_JSON = RESULTS_DIR / "run_metadata.json"

REPORTS = {
    "MLP baseline": RESULTS_DIR / "classification_report_mlp_baseline.txt",
    "MLP tuned": RESULTS_DIR / "classification_report_mlp_tuned.txt",
    "Random Forest": RESULTS_DIR / "classification_report_random_forest.txt",
}

PLOTS = {
    "Macierze pomyłek": PLOTS_DIR / "confusion_matrices_side_by_side.png",
    "Loss i F1 (historia)": PLOTS_DIR / "loss_and_f1_history.png",
    "Scheduler LR": PLOTS_DIR / "scheduler_lr.png",
}


def wczytaj_json(path: Path) -> dict:
    """Wczytuje JSON; zwraca pusty dict gdy brak pliku."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


st.set_page_config(page_title="Airline MLP Project", layout="wide")
st.title("Airline Passenger Satisfaction — MLP (PyTorch) vs Random Forest")
st.caption("Target: `Class` (Business / Eco / Eco Plus) | Dane: `data/train.csv` + `data/test.csv`")

with st.sidebar:
    st.header("Akcje")
    if st.button("Uruchom pełny eksperyment", type="primary", use_container_width=True):
        with st.spinner("Trwa trening i ewaluacja. To może potrwać kilka minut..."):
            uruchom_pelny_eksperyment()
        st.success("Gotowe. Wyniki zapisane w `results/airline`.")
        st.rerun()
    st.markdown("---")
    st.markdown("**CLI:** `python3 run_experiment.py`")
    st.markdown("**Notebook:** `docs/airline_passenger_satisfaction_mlp_project2.ipynb`")


tab_summary, tab_reports, tab_plots, tab_conclusions, tab_files = st.tabs(
    ["Podsumowanie", "Raporty", "Wykresy", "Wnioski", "Pliki"]
)

with tab_summary:
    if not MODEL_CSV.exists():
        st.warning("Brak wyników. Kliknij przycisk uruchomienia eksperymentu w pasku bocznym.")
    else:
        comparison = pd.read_csv(MODEL_CSV)
        meta = wczytaj_json(META_JSON)
        best = comparison.sort_values("f1_macro", ascending=False).iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Najlepszy model", str(best["model"]))
        c2.metric("F1-macro", f"{best['f1_macro']:.4f}")
        c3.metric("Accuracy", f"{best['accuracy']:.4f}")
        c4.metric("Urządzenie", str(meta.get("device", "brak")))

        d1, d2, d3 = st.columns(3)
        d1.metric("Train", f"{int(meta.get('n_train', 0)):,}")
        d2.metric("Validation", f"{int(meta.get('n_val', 0)):,}")
        d3.metric("Test", f"{int(meta.get('n_test', 0)):,}")

        st.subheader("Tabela porównawcza modeli")
        st.dataframe(comparison.round(4), use_container_width=True)

        if GRID_CSV.exists():
            st.subheader("Top 10 konfiguracji (walidacyjne macro F1)")
            grid = pd.read_csv(GRID_CSV).sort_values("best_val_f1_macro", ascending=False).head(10)
            st.dataframe(
                grid[["name", "best_val_f1_macro", "dropout", "learning_rate"]].round(4),
                use_container_width=True,
            )

with tab_reports:
    for name, report_path in REPORTS.items():
        st.subheader(name)
        if report_path.exists():
            st.code(report_path.read_text(encoding="utf-8"), language="text")
        else:
            st.info(f"Brak pliku: `{report_path.name}`")

with tab_plots:
    for title, path in PLOTS.items():
        st.subheader(title)
        if path.exists():
            st.image(str(path), use_column_width=True)
        else:
            st.info(f"Brak pliku: `{path.name}`")

with tab_conclusions:
    st.subheader("Wnioski końcowe")
    if MODEL_CSV.exists():
        comparison = pd.read_csv(MODEL_CSV)
        best = comparison.sort_values("f1_macro", ascending=False).iloc[0]

        st.markdown(f"""
**1. Najlepszy model:** {best['model']} (F1-macro = {best['f1_macro']:.4f}, accuracy = {best['accuracy']:.4f})

**2. MLP vs Random Forest:**
- MLP tuned wygrywa **F1-macro** — lepiej balansuje klasy dzięki class weights w CrossEntropyLoss.
- Random Forest wygrywa **accuracy**, ale prawie ignoruje klasę Eco Plus (recall ~1%).

**3. Klasa Eco Plus (~7%):**
- To klasa problematyczna — mniejszościowa.
- MLP z class weights osiąga recall ~41% (łapie ok. 2 na 5 przypadków).
- RF bez balansowania klasyfikuje prawie wszystko jako Business lub Eco.

**4. Dlaczego F1-macro jest ważniejsze niż accuracy:**
- Accuracy jest zdominowana przez duże klasy (Business 48%, Eco 45%).
- F1-macro waży każdą klasę równo — wymusza dobre wyniki na WSZYSTKICH klasach.

**5. Ograniczenia:**
- Eco Plus ma za mało przykładów dla idealnej klasyfikacji.
- Brak feature engineering (agregaty ocen, interakcje).
- Grid search 18 konfiguracji — kompromis czasowy.
- RF bez class_weight='balanced' (celowo — domyślne parametry jako baseline).

**6. Możliwe ulepszenia:**
- RF z `class_weight='balanced'`
- SMOTE / oversampling Eco Plus
- Bayesowska optymalizacja (Optuna)
- Feature engineering
- Ensemble (voting/stacking)
        """)
    else:
        st.info("Brak wyników — uruchom eksperyment.")

with tab_files:
    if RESULTS_DIR.exists():
        files = sorted([f for f in RESULTS_DIR.rglob("*") if f.is_file()])
        for file_path in files:
            st.caption(str(file_path.relative_to(ROOT)))
    else:
        st.info("Katalog wyników jeszcze nie istnieje.")
