"""LSTM classifier for 48-month payment trajectories.

Uses nn.LSTM (built-in). Swap for a from-scratch CustomLSTMCell if the
assignment enforces that constraint — the training loop is unchanged.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class SequenceLSTM(nn.Module):
    def __init__(self, input_size: int, hidden: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, bidirectional: bool = True,
                 num_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 48, input_size)
        out, (h_n, _) = self.lstm(x)
        # Mean-pool over time for a stable summary; last-step alone is noisier
        pooled = out.mean(dim=1)
        return self.head(pooled)
