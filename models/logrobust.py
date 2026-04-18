"""LogRobust: BiLSTM with attention (Zhang et al., FSE 2019)."""

import torch
import torch.nn as nn


class LogRobust(nn.Module):
    """Bidirectional LSTM with attention mechanism."""

    def __init__(self, input_size: int = 300, hidden_size: int = 128,
                 num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.5,
        )
        # Attention params
        self.attn_w = nn.Linear(hidden_size * 2, hidden_size * 2, bias=False)
        self.attn_u = nn.Parameter(torch.randn(hidden_size * 2))
        # Output
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        # Attention
        attn_scores = torch.tanh(self.attn_w(out))  # (batch, seq_len, hidden*2)
        attn_scores = torch.matmul(attn_scores, self.attn_u)  # (batch, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=-1).unsqueeze(-1)
        context = (out * attn_weights).sum(dim=1)  # (batch, hidden*2)
        # Classify
        out = torch.relu(self.fc1(context))
        return self.fc2(out)
