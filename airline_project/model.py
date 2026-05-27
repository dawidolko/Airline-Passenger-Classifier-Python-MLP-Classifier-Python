"""
Moduł definiujący architekturę sieci MLP oraz funkcję wyboru urządzenia (GPU/CPU).
Sieć budowana jest dynamicznie na podstawie listy rozmiarów warstw ukrytych.
"""
from __future__ import annotations

import torch
from torch import nn


class AirlineMLP(nn.Module):
    """
    Konfigurowalna sieć MLP do klasyfikacji wieloklasowej.

    Każda warstwa ukryta ma układ: Linear → BatchNorm1d → ReLU → Dropout.
    Ostatnia warstwa to Linear bez aktywacji (logity wejściowe do CrossEntropyLoss).

    Parametry konstruktora:
        input_dim   — liczba cech wejściowych po preprocessingu
        hidden_sizes — krotka z rozmiarami kolejnych warstw ukrytych, np. (512, 256, 128)
        output_dim  — liczba klas (3 w tym projekcie)
        dropout     — prawdopodobieństwo zerowania neuronu (regularyzacja)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: tuple[int, ...],
        output_dim: int,
        dropout: float,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Przepuszcza tensor cech przez sieć i zwraca logity (raw scores)."""
        return self.network(features)


def wybierz_urzadzenie() -> torch.device:
    """
    Wybiera najlepsze dostępne urządzenie obliczeniowe dla PyTorch.
    Priorytet: CUDA (GPU NVIDIA) > MPS (GPU Apple Silicon) > CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
