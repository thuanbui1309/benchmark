"""LogCNN: TextCNN for log anomaly detection (Lu et al., ISSRE 2018)."""

import torch
import torch.nn as nn


class LogCNN(nn.Module):
    """TextCNN with multiple kernel sizes."""

    def __init__(self, embed_dim: int = 300, seq_len: int = 100,
                 out_channels: int = 128, kernel_sizes: list = None):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 4, 5]
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, out_channels, (k, embed_dim)),
                nn.ReLU(),
                nn.MaxPool2d((seq_len - k + 1, 1)),
            )
            for k in kernel_sizes
        ])
        self.fc = nn.Linear(len(kernel_sizes) * out_channels, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, embed_dim)
        x = x.unsqueeze(1)  # (batch, 1, seq_len, embed_dim)
        conv_outs = [conv(x).squeeze(-1).squeeze(-1) for conv in self.convs]
        x = torch.cat(conv_outs, dim=1)  # (batch, n_kernels * out_channels)
        return self.fc(x)
