#!/usr/bin/env python3
"""Pobiera Wine Quality z UCI (red + white) i zapisuje data/WineQT.csv."""

from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import DATA_PATH, FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMN

UCI_ZIP = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"


def main() -> None:
    data_dir = DATA_PATH.parent
    data_dir.mkdir(exist_ok=True)
    zip_path = data_dir / "wine+quality.zip"

    print(f"Pobieranie: {UCI_ZIP}")
    urllib.request.urlretrieve(UCI_ZIP, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)

    cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    red = pd.read_csv(data_dir / "winequality-red.csv", sep=";")
    white = pd.read_csv(data_dir / "winequality-white.csv", sep=";")
    red.columns = cols
    white.columns = cols
    df = pd.concat([red, white], ignore_index=True)
    df.insert(0, ID_COLUMN, range(1, len(df) + 1))
    df.to_csv(DATA_PATH, index=False)
    print(f"Zapisano {DATA_PATH} — shape {df.shape}, klasy: {sorted(df[TARGET_COLUMN].unique())}")


if __name__ == "__main__":
    main()
