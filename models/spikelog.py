"""SpikeLog: Original SNN for log anomaly detection (Qi et al., TKDE 2024)."""

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, functional


class SpikeNet(nn.Module):
    """Recurrent LIF network (from original SpikeLog)."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lif2 = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.lif_out = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        batch, seq_len, _ = x.shape
        mem_out = []
        for t in range(seq_len):
            h = self.fc1(x[:, t, :])
            h = self.lif1(h)
            h = self.fc2(h)
            h = self.lif2(h)
            h = self.fc_out(h)
            h = self.lif_out(h)
            mem_out.append(h)
        return torch.stack(mem_out, dim=1)  # (batch, seq_len, output_size)


class SpikeLog(nn.Module):
    """SpikeLog with LSTM decoder (original architecture)."""

    def __init__(self, input_size: int = 300, hidden_size: int = 128,
                 num_out: int = 64):
        super().__init__()
        self.spike_net = SpikeNet(input_size, hidden_size, num_out)
        self.lstm = nn.LSTM(num_out, num_out, num_layers=2, batch_first=True)
        self.fc = nn.Linear(num_out * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        functional.reset_net(self)
        spike_out = self.spike_net(x)
        lstm_out, _ = self.lstm(spike_out)
        # Use last timestep from both branches
        combined = torch.cat([spike_out[:, -1, :], lstm_out[:, -1, :]], dim=-1)
        return self.fc(combined)
