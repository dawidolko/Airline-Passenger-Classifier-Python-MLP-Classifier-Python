#!/usr/bin/env python3
"""
Streamlit: klasyfikacja jakości wina (Wine Quality) — MLP vs RF / XGBoost.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.config import CLASS_LABELS_PL, DATA_PATH, FEATURE_COLUMNS, ID_COLUMN, RESULTS_DIR, TARGET_COLUMN
from src.experiment import uruchom_pelny_eksperyment, wczytaj_pelna_ramke
from src.wykresy import (
    wykres_agregaty_rodzin,
    wykres_macierz_korelacji,
    wykres_macierz_pomylek,
    wykres_porownanie_metryk,
    wykres_rozkald_jakosci,
)

TEX_SZCZEGOLY = RESULTS_DIR / "wyniki_szczegolowe.tex"
TEX_AGREGATY = RESULTS_DIR / "wyniki_agregaty_rodzin.tex"
CSV_TEST = RESULTS_DIR / "wyniki_test_mlp.csv"
CSV_CM = RESULTS_DIR / "macierz_pomylek_test.csv"


def wyswietl_plik_tex(sciezka: Path) -> None:
    if not sciezka.is_file():
        st.caption(f"Brak pliku: `{sciezka.name}` — uruchom ewaluację.")
        return
    st.code(sciezka.read_text(encoding="utf-8"), language="latex", line_numbers=True)


st.set_page_config(
    page_title="Wine Quality — MLP vs RF/XGBoost",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@media print {
  [data-testid="stSidebar"], [data-testid="stToolbar"], footer { display: none !important; }
  .block-container { max-width: 100% !important; padding: 0.5cm !important; }
}
.block-container { padding-top: 1.2rem; }
div[data-testid="stMetricValue"] { font-size: 1.35rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Klasyfikacja jakości wina (Wine Quality)")
st.caption(
    "Zbiór **WineQT** — ok. 6497 próbek, 11 cech chemicznych, **wieloklasowa** etykieta `quality` (3–9)."
)

csv_szczegol = RESULTS_DIR / "wyniki_szczegolowe.csv"
csv_agregat = RESULTS_DIR / "wyniki_agregaty_rodzin.csv"
folder_wykresy = RESULTS_DIR / "wykresy"

with st.sidebar:
    st.header("Akcje")
    if st.button("Uruchom pełną ewaluację", type="primary", use_container_width=True):
        with st.spinner("5-fold CV, test MLP, zapis wyników…"):
            uruchom_pelny_eksperyment()
        st.success("Zapisano w `results/`.")
        st.rerun()
    st.markdown("**CLI:** `python run_experiment.py`  \n**Start:** `./start.sh` / `start.bat`")
    st.markdown("---")
    st.markdown("**Druk:** `Ctrl+P` / `Cmd+P` (ukrywa panel boczny).")

tab_opis, tab_wymogi, tab_dane, tab_wyniki = st.tabs(
    ["Opis projektu", "Wymogi i metodologia", "Zbiór danych (EDA)", "Wyniki klasyfikacji"]
)

with tab_opis:
    st.info(
        "Projekt 2: **MLP** (główny) vs **Random Forest** i **XGBoost**. "
        "Klasyfikacja **wieloklasowa** — nie binarna."
    )

    st.subheader("Zbiór danych")
    st.markdown(
        """
**Wine Quality** (UCI / [Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)) —
próbki wina czerwonego i białego z **11 parametrami chemicznymi** i oceną **`quality`** (3–9).

| Parametr | Wartość |
|----------|---------|
| Rekordy | ~6497 |
| Cechy | 11 (numeryczne) |
| Klasa decyzyjna | `quality` (jedna kolumna, 7 klas w praktyce) |
| Braki | brak |
"""
    )

    st.subheader("Cel porównania")
    st.markdown(
        """
1. **MLP** — 3 architektury warstw ukrytych.  
2. **Random Forest** — 3 warianty.  
3. **XGBoost** — 3 warianty (jeśli dostępny).  

**5-fold CV** + **test 20%** dla najlepszego MLP (wybór po `balanced_accuracy`).
"""
    )

    st.subheader("Formuły")
    st.markdown(
        r"""
**Standaryzacja:** \(z_j = (x_j - \mu_j) / \sigma_j\) (tylko na treningu foldy).

**Accuracy:** \(\mathrm{Acc} = \frac{1}{N}\sum \mathbb{1}(\hat{y}_i = y_i)\)

**Balanced accuracy:** średnia czułości po klasach (ważna przy dominacji ocen 5 i 6).

**F1-macro:** uśrednione F1 po wszystkich klasach `quality`.
"""
    )

    st.subheader("Przebieg")
    st.markdown(
        """
1. Wczytanie `data/WineQT.csv`, pominięcie kolumny `Id`.  
2. `Pipeline(StandardScaler → klasyfikator)` + stratyfikowana CV.  
3. Wybór najlepszego MLP → ocena na 20% testu, macierz pomyłek.  
4. Eksport CSV, LaTeX, HTML.
"""
    )

    st.subheader("Klasy (quality)")
    for k, opis in sorted(CLASS_LABELS_PL.items()):
        st.markdown(f"- **{k}** — {opis}")

    st.subheader("LaTeX — podgląd")
    wyswietl_plik_tex(TEX_SZCZEGOLY)

with tab_wymogi:
    st.markdown(
        """
| Wymaganie | Realizacja |
|-----------|------------|
| Zbiór ≠ Iris, wieloklasowy | Wine Quality, `quality` 3–9 |
| Nie binarna | tak — kilka klas ocen |
| Jedna kolumna decyzyjna | `quality` |
| MLP główny | 3× `MLPClassifier` |
| RF / XGBoost | 3+3 warianty |
| Skalowanie bez przecieku | `StandardScaler` w Pipeline |
| 5-fold CV stratyfikowana | `StratifiedKFold(42)` |
| Metryki | accuracy, balanced accuracy, F1-macro |
| Streamlit + wykresy | `streamlit_app.py`, `results/wykresy/` |
"""
    )

with tab_dane:
    try:
        df_raw = wczytaj_pelna_ramke(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Brak `{DATA_PATH}`. Uruchom: `python scripts/pobierz_dane.py`")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rekordy", len(df_raw))
    c2.metric("Cechy", len(FEATURE_COLUMNS))
    c3.metric("Klasy quality", f"{df_raw[TARGET_COLUMN].min()}–{df_raw[TARGET_COLUMN].max()}")
    c4.metric("Braki", "0" if not df_raw[FEATURE_COLUMNS].isna().any().any() else "tak")

    st.plotly_chart(wykres_rozkald_jakosci(df_raw), use_container_width=True)
    st.plotly_chart(wykres_macierz_korelacji(df_raw), use_container_width=True)

    with st.expander("Podgląd danych"):
        pokaz = [c for c in df_raw.columns if c != ID_COLUMN]
        st.dataframe(df_raw[pokaz].head(15), use_container_width=True)
    with st.expander("Statystyki opisowe"):
        st.dataframe(df_raw[FEATURE_COLUMNS].describe().round(3), use_container_width=True)

with tab_wyniki:
    if not csv_szczegol.is_file():
        st.warning("Brak wyników — uruchom ewaluację z panelu bocznego.")
    else:
        df = pd.read_csv(csv_szczegol)
        agg = pd.read_csv(csv_agregat)

        st.subheader("Walidacja krzyżowa (5-fold)")
        st.plotly_chart(wykres_porownanie_metryk(df), use_container_width=True)
        st.plotly_chart(wykres_agregaty_rodzin(agg), use_container_width=True)

        if CSV_TEST.is_file():
            st.subheader("Test 20% — najlepszy MLP")
            tr = pd.read_csv(CSV_TEST).iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Model", tr["model"])
            m2.metric("Accuracy", f"{tr['accuracy_test']:.4f}")
            m3.metric("Balanced acc.", f"{tr['balanced_accuracy_test']:.4f}")
            m4.metric("F1-macro", f"{tr['f1_macro_test']:.4f}")

        if CSV_CM.is_file():
            st.markdown("**Macierz pomyłek (test)**")
            st.dataframe(pd.read_csv(CSV_CM, index_col=0), use_container_width=True)

        st.subheader("Tabele")
        st.dataframe(df.round(4), use_container_width=True)
        st.dataframe(agg.round(4), use_container_width=True)

        st.subheader("Pobieranie")
        cols = st.columns(3)
        for i, (p, n) in enumerate(
            [
                (TEX_SZCZEGOLY, "wyniki_szczegolowe.tex"),
                (TEX_AGREGATY, "wyniki_agregaty_rodzin.tex"),
                (RESULTS_DIR / "wersje_bibliotek.txt", "wersje_bibliotek.txt"),
            ]
        ):
            if p.is_file():
                cols[i].download_button(f"Pobierz {n}", p.read_text(encoding="utf-8"), n, key=f"dl{i}")

        if folder_wykresy.is_dir():
            st.markdown("**HTML:**")
            for h in sorted(folder_wykresy.glob("*.html")):
                st.caption(str(h.relative_to(ROOT)))
