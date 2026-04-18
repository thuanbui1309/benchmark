"""LogAnomaly: Dual LSTM for sequential + semantic patterns (Meng et al., IJCAI 2019)."""

import torch
import torch.nn as nn


class LogAnomaly(nn.Module):
    """Two parallel LSTMs: one for sequences, one for semantics."""

    def __init__(self, input_size: int = 300, hidden_size: int = 128,
                 num_layers: int = 2, vocab_size: int = 300):
        super().__init__()
        self.lstm_seq = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.lstm_sem = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(2 * hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size) - used for both branches
        out_seq, _ = self.lstm_seq(x)
        out_sem, _ = self.lstm_sem(x)
        combined = torch.cat([out_seq[:, -1, :], out_sem[:, -1, :]], dim=-1)
        return self.fc(combined)
