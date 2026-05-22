#!/usr/bin/env python3
"""
Skrypt uruchamiający pełny eksperyment z katalogu głównego repozytorium.

Użycie (z katalogu projektu):
    python run_experiment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Umożliwia import pakietu `src` przy uruchomieniu pliku bez instalacji jako pakietu
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment import uruchom_pelny_eksperyment  # noqa: E402


def main() -> None:
    print("Ewaluacja modeli (moze potrwac kilka minut)...")
    szczegoly, grupy, wynik_test = uruchom_pelny_eksperyment()
    print("\n--- Podsumowanie (fragment) ---")
    print(szczegoly.to_string(index=False))
    print("\n--- Agregaty rodzin ---")
    print(grupy.to_string(index=False))
    print("\n--- Najlepszy MLP — held-out test (20%) ---")
    print(
        f"Model: {wynik_test['model']}, accuracy={wynik_test['accuracy']:.4f}, "
        f"balanced_accuracy={wynik_test['balanced_accuracy']:.4f}, "
        f"f1_macro={wynik_test['f1_macro']:.4f}"
    )
    print(
        "\nZapisano w results/: CSV, LaTeX (.tex), wykresy HTML (wykresy/), "
        "macierz_pomylek_test.csv, wyniki_test_mlp.csv, wersje_bibliotek.txt"
    )


if __name__ == "__main__":
    main()
