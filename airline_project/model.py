"""
Module defining the MLP network architecture and the device-selection function (GPU/CPU).
The network is built dynamically from a list of hidden-layer sizes.
"""
from __future__ import annotations

import torch
from torch import nn


class AirlineMLP(nn.Module):
    """
    A configurable MLP network for multiclass classification.

    Each hidden layer follows the layout: Linear → BatchNorm1d → ReLU → Dropout.
    The final layer is a Linear without activation (logits fed into CrossEntropyLoss).

    Constructor parameters:
        input_dim   — number of input features after preprocessing
        hidden_sizes — tuple with the sizes of successive hidden layers, e.g. (512, 256, 128)
        output_dim  — number of classes (3 in this project)
        dropout     — probability of zeroing a neuron (regularization)
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
        """Passes the feature tensor through the network and returns logits (raw scores)."""
        return self.network(features)


def wybierz_urzadzenie() -> torch.device:
    """
    Selects the best available compute device for PyTorch.
    Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon GPU) > CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
