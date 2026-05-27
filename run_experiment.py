#!/usr/bin/env python3
"""
Punkt wejścia do eksperymentu Airline Passenger Satisfaction.
Uruchamia pełny pipeline: preprocessing, grid search MLP, ewaluacja, wykresy.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from airline_project.experiment import uruchom_pelny_eksperyment


def main() -> None:
    """Uruchamia pełen pipeline eksperymentu."""
    print("Airline Passenger Satisfaction — eksperyment MLP (PyTorch)")
    uruchom_pelny_eksperyment()


if __name__ == "__main__":
    main()
