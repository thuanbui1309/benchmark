"""NeuralLog: Transformer encoder for log anomaly detection (Le & Zhang, ICSE 2021)."""

import torch
import torch.nn as nn


class NeuralLog(nn.Module):
    """Transformer encoder with classification head."""

    def __init__(self, dim_model: int = 300, num_heads: int = 6,
                 num_layers: int = 1, dim_feedforward: int = 1200,
                 dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(dim_model, 32)
        self.fc2 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, dim_model)
        x = self.encoder(x)
        x = x.sum(dim=1)  # aggregate over sequence
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)
