"""
Wykresy Plotly — zbiór Wine Quality oraz wyniki eksperymentu (MLP / RF / XGBoost).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import FEATURE_COLUMNS, TARGET_COLUMN


def wykres_rozkald_jakosci(df: pd.DataFrame, kolumna_klasy: str = TARGET_COLUMN) -> go.Figure:
    """Słupki liczebności ocen jakości wina (klasyfikacja wieloklasowa)."""
    vc = df[kolumna_klasy].value_counts().sort_index()
    fig = px.bar(
        x=vc.index.astype(str),
        y=vc.values,
        labels={"x": "Ocena jakości (quality)", "y": "Liczba próbek"},
        title="Rozkład ocen jakości wina",
        color=vc.values,
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        xaxis_title="Klasa (quality)",
        yaxis_title="Liczba obserwacji",
        coloraxis_showscale=False,
        template="plotly_white",
        height=440,
    )
    return fig


# Alias dla kompatybilności wewnętrznej
wykres_rozkald_klas = wykres_rozkald_jakosci


def wykres_macierz_korelacji(df: pd.DataFrame, kolumny_numeryczne: list[str] | None = None) -> go.Figure:
    """Mapa ciepła korelacji Pearsona między cechami chemicznymi."""
    kolumny = kolumny_numeryczne or list(FEATURE_COLUMNS)
    sub = df[kolumny].select_dtypes(include=["number"])
    corr = sub.corr(numeric_only=True)
    fig = px.imshow(
        corr,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Macierz korelacji cech chemicznych (Pearson)",
    )
    fig.update_layout(height=640, template="plotly_white")
    fig.update_xaxes(tickangle=-45)
    return fig


def _etykieta_wykresu(wiersz: pd.Series) -> str:
    return f"{wiersz.get('rodzina', '')}: {wiersz.get('model', '')}"


def _sortuj_modele(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["etykieta"] = df.apply(_etykieta_wykresu, axis=1)
    df["_sort"] = df["rodzina"].map({"MLP": 0, "RandomForest": 1, "XGBoost": 2}).fillna(9)
    return df.sort_values(["_sort", "model"])


def wykres_porownanie_metryk(szczegoly: pd.DataFrame) -> go.Figure:
    """Trzy panele metryk z 5-fold CV dla wszystkich wariantów modeli."""
    df = _sortuj_modele(szczegoly)
    kolory = {"MLP": "#E84855", "RandomForest": "#1B998B", "XGBoost": "#5C4D7D"}
    barwy = [kolory.get(r, "#888") for r in df["rodzina"]]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Accuracy (±std)", "Balanced accuracy (±std)", "F1-macro (±std)"),
        horizontal_spacing=0.06,
    )
    x = list(range(len(df)))
    paramy = [
        ("accuracy_mean", "accuracy_std"),
        ("balanced_accuracy_mean", "balanced_accuracy_std"),
        ("f1_macro_mean", "f1_macro_std"),
    ]
    for col, (mean_kol, std_kol) in enumerate(paramy, start=1):
        fig.add_trace(
            go.Bar(
                x=x,
                y=df[mean_kol],
                error_y=dict(type="data", array=df[std_kol], visible=True),
                marker_color=barwy,
                showlegend=False,
                hovertext=df["etykieta"],
                hoverinfo="text+y",
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(tickmode="array", tickvals=x, ticktext=df["model"], tickangle=-45, row=1, col=col)
        fig.update_yaxes(range=[0, 1.05], row=1, col=col)

    fig.update_layout(
        title_text="Porównanie modeli — jakość wina (MLP vs RF vs XGBoost, 5-fold CV)",
        template="plotly_white",
        height=520,
        margin=dict(b=130),
    )
    return fig


def wykres_agregaty_rodzin(agg: pd.DataFrame) -> go.Figure:
    """Średnia vs maksimum metryk w obrębie rodzin modeli."""
    fig = go.Figure()
    serie = [
        ("Średnia accuracy", "accuracy_srednia", "#5C4D7D"),
        ("Maks. accuracy", "accuracy_max", "#8F7EAF"),
        ("Średnia balanced acc.", "balanced_accuracy_srednia", "#C17C74"),
        ("Maks. balanced acc.", "balanced_accuracy_max", "#E8A09A"),
        ("Średnia F1-macro", "f1_macro_srednia", "#1B998B"),
        ("Maks. F1-macro", "f1_macro_max", "#7FD1C2"),
    ]
    for nazwa, kol, kolor in serie:
        if kol in agg.columns:
            fig.add_trace(go.Bar(name=nazwa, x=agg["rodzina"], y=agg[kol], marker_color=kolor))
    fig.update_layout(
        barmode="group",
        title="Agregacja po rodzinie modelu — średnie i maksima metryk",
        yaxis_title="Wartość metryki",
        template="plotly_white",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def wykres_macierz_pomylek(wynik_test: dict[str, Any]) -> go.Figure:
    """Macierz pomyłek najlepszego MLP na zbiorze testowym (20%)."""
    cm = wynik_test["confusion_matrix"]
    etykiety = [str(e) for e in wynik_test["labels"]]
    fig = px.imshow(
        cm,
        x=etykiety,
        y=etykiety,
        text_auto=True,
        color_continuous_scale="Blues",
        labels={"x": "Predykcja (quality)", "y": "Rzeczywista (quality)", "color": "Liczba"},
        title=f"Macierz pomyłek — najlepszy MLP ({wynik_test['model']})",
    )
    fig.update_layout(height=560, template="plotly_white")
    return fig


def zapisz_wykresy_html(
    df_zbior: pd.DataFrame,
    szczegoly: pd.DataFrame,
    grupy: pd.DataFrame,
    wynik_test: dict[str, Any],
    katalog: Path,
) -> None:
    """Zapisuje wykresy HTML do results/wykresy/."""
    out = katalog / "wykresy"
    out.mkdir(parents=True, exist_ok=True)

    wykres_rozkald_jakosci(df_zbior).write_html(out / "01_rozkald_jakosci.html", include_plotlyjs="cdn")
    wykres_macierz_korelacji(df_zbior).write_html(out / "02_macierz_korelacji.html", include_plotlyjs="cdn")
    wykres_porownanie_metryk(szczegoly).write_html(out / "03_porownanie_modeli.html", include_plotlyjs="cdn")
    wykres_agregaty_rodzin(grupy).write_html(out / "04_agregaty_rodzin.html", include_plotlyjs="cdn")
    wykres_macierz_pomylek(wynik_test).write_html(out / "05_macierz_pomylek.html", include_plotlyjs="cdn")
