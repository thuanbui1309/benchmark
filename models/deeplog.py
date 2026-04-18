"""DeepLog: LSTM-based log anomaly detection (Lu et al., CCS 2017)."""

import torch
import torch.nn as nn


class DeepLog(nn.Module):
    """LSTM for next log event prediction."""

    def __init__(self, input_size: int = 300, hidden_size: int = 128,
                 num_layers: int = 2, vocab_size: int = 300):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # last timestep
        return out
