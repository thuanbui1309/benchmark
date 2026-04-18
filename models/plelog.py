"""PLELog: Attention-based GRU (Yang et al., ICSE 2021)."""

import torch
import torch.nn as nn


class PLELog(nn.Module):
    """Bidirectional GRU with attention mechanism."""

    def __init__(self, input_size: int = 300, hidden_size: int = 128,
                 num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        # Attention
        self.attn_guide = nn.Parameter(torch.randn(hidden_size * 2))
        self.attn_linear = nn.Linear(hidden_size * 2, hidden_size * 2)
        # Output
        self.fc = nn.Linear(hidden_size * 2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.gru(x)  # (batch, seq_len, hidden*2)
        # Attention
        attn_scores = torch.tanh(self.attn_linear(out))
        attn_scores = (attn_scores * self.attn_guide).sum(dim=-1)  # (batch, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=-1).unsqueeze(-1)
        context = (out * attn_weights).sum(dim=1)  # (batch, hidden*2)
        return self.fc(context)
