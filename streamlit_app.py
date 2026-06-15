#!/usr/bin/env python3
"""
Streamlit dashboard for the Airline Passenger Satisfaction project.
Displays the results of the MLP (PyTorch) vs Random Forest experiment.
Interface language: English.
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
    "Confusion matrices": PLOTS_DIR / "confusion_matrices_side_by_side.png",
    "Loss and F1 (history)": PLOTS_DIR / "loss_and_f1_history.png",
    "Scheduler LR": PLOTS_DIR / "scheduler_lr.png",
}


def wczytaj_json(path: Path) -> dict:
    """Loads JSON; returns an empty dict when the file is missing."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


st.set_page_config(page_title="Airline MLP Project", layout="wide")
st.title("Airline Passenger Satisfaction — MLP (PyTorch) vs Random Forest")
st.caption("Target: `Class` (Business / Eco / Eco Plus) | Data: `data/train.csv` + `data/test.csv`")

with st.sidebar:
    st.header("Actions")
    if st.button("Run the full experiment", type="primary", use_container_width=True):
        with st.spinner("Training and evaluation in progress. This may take a few minutes..."):
            uruchom_pelny_eksperyment()
        st.success("Done. Results saved in `results/airline`.")
        st.rerun()
    st.markdown("---")
    st.markdown("**CLI:** `python3 run_experiment.py`")
    st.markdown("**Notebook:** `docs/airline_passenger_satisfaction_mlp_project2.ipynb`")


tab_summary, tab_reports, tab_plots, tab_conclusions, tab_files = st.tabs(
    ["Summary", "Reports", "Charts", "Conclusions", "Files"]
)

with tab_summary:
    if not MODEL_CSV.exists():
        st.warning("No results. Click the run-experiment button in the sidebar.")
    else:
        comparison = pd.read_csv(MODEL_CSV)
        meta = wczytaj_json(META_JSON)
        best = comparison.sort_values("f1_macro", ascending=False).iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best model", str(best["model"]))
        c2.metric("F1-macro", f"{best['f1_macro']:.4f}")
        c3.metric("Accuracy", f"{best['accuracy']:.4f}")
        c4.metric("Device", str(meta.get("device", "none")))

        d1, d2, d3 = st.columns(3)
        d1.metric("Train", f"{int(meta.get('n_train', 0)):,}")
        d2.metric("Validation", f"{int(meta.get('n_val', 0)):,}")
        d3.metric("Test", f"{int(meta.get('n_test', 0)):,}")

        st.subheader("Model comparison table")
        st.dataframe(comparison.round(4), use_container_width=True)

        if GRID_CSV.exists():
            st.subheader("Top 10 configurations (validation macro F1)")
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
            st.info(f"Missing file: `{report_path.name}`")

with tab_plots:
    for title, path in PLOTS.items():
        st.subheader(title)
        if path.exists():
            st.image(str(path), use_column_width=True)
        else:
            st.info(f"Missing file: `{path.name}`")

with tab_conclusions:
    st.subheader("Final conclusions")
    if MODEL_CSV.exists():
        comparison = pd.read_csv(MODEL_CSV)
        best = comparison.sort_values("f1_macro", ascending=False).iloc[0]

        st.markdown(f"""
**1. Best model:** {best['model']} (F1-macro = {best['f1_macro']:.4f}, accuracy = {best['accuracy']:.4f})

**2. MLP vs Random Forest:**
- MLP tuned wins on **F1-macro** — it balances the classes better thanks to class weights in CrossEntropyLoss.
- Random Forest wins on **accuracy**, but almost ignores the Eco Plus class (recall ~1%).

**3. The Eco Plus class (~7%):**
- This is a problematic, minority class.
- The MLP with class weights reaches recall ~41% (catches about 2 out of 5 cases).
- RF without balancing classifies almost everything as Business or Eco.

**4. Why F1-macro matters more than accuracy:**
- Accuracy is dominated by the large classes (Business 48%, Eco 45%).
- F1-macro weights every class equally — it forces good results on ALL classes.

**5. Limitations:**
- Eco Plus has too few examples for perfect classification.
- No feature engineering (score aggregates, interactions).
- Grid search over 18 configurations — a time trade-off.
- RF without class_weight='balanced' (deliberately — default parameters as a baseline).

**6. Possible improvements:**
- RF with `class_weight='balanced'`
- SMOTE / oversampling of Eco Plus
- Bayesian optimization (Optuna)
- Feature engineering
- Ensemble (voting/stacking)
        """)
    else:
        st.info("No results — run the experiment.")

with tab_files:
    if RESULTS_DIR.exists():
        files = sorted([f for f in RESULTS_DIR.rglob("*") if f.is_file()])
        for file_path in files:
            st.caption(str(file_path.relative_to(ROOT)))
    else:
        st.info("The results directory does not exist yet.")
