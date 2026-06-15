#!/usr/bin/env python3
"""
Entry point for the Airline Passenger Satisfaction experiment.
Runs the full pipeline: preprocessing, MLP grid search, evaluation, charts.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from airline_project.experiment import uruchom_pelny_eksperyment


def main() -> None:
    """Runs the full experiment pipeline."""
    print("Airline Passenger Satisfaction — MLP experiment (PyTorch)")
    uruchom_pelny_eksperyment()


if __name__ == "__main__":
    main()
