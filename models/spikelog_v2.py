"""SpikeLog-v2: SDSA + BSPN + pruning (ours)."""

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional


class BSPN(nn.Module):
    """Bit-Shift Power Normalization (neuromorphic-compatible)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # L1 normalization with bit-shift approximation
        norm = x.abs().mean(dim=-1, keepdim=True) + self.eps
        return x / norm * self.scale


class SDSABlock(nn.Module):
    """Spike-Driven Self-Attention Block."""

    def __init__(self, dim: int, num_heads: int = 4, tau: float = 2.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Q, K, V projections
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)
        self.w_o = nn.Linear(dim, dim, bias=False)

        # Normalization (BSPN for neuromorphic)
        self.norm_q = BSPN(dim)
        self.norm_k = BSPN(dim)
        self.norm_v = BSPN(dim)
        self.norm_ffn = BSPN(dim)

        # LIF neurons
        self.lif_q = neuron.LIFNode(tau=tau, surrogate_function=neuron.surrogate.ATan())
        self.lif_k = neuron.LIFNode(tau=tau, surrogate_function=neuron.surrogate.ATan())
        self.lif_v = neuron.LIFNode(tau=tau, surrogate_function=neuron.surrogate.ATan())
        self.lif_ffn = neuron.LIFNode(tau=tau, surrogate_function=neuron.surrogate.ATan())

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            neuron.LIFNode(tau=tau, surrogate_function=neuron.surrogate.ATan()),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, dim)
        B, L, D = x.shape

        # QKV with spike conversion
        q = self.lif_q(self.norm_q(self.w_q(x)))
        k = self.lif_k(self.norm_k(self.w_k(x)))
        v = self.lif_v(self.norm_v(self.w_v(x)))

        # Reshape for multi-head
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Spike-driven attention (AND-accumulate)
        attn = (q @ k.transpose(-2, -1)) / self.head_dim
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.w_o(out)

        # Residual + FFN
        x = x + out
        x = x + self.ffn(self.norm_ffn(x))
        return x


class SpikeLogV2(nn.Module):
    """SpikeLog-v2: SDSA + BSPN + token pruning."""

    def __init__(self, input_dim: int = 300, hidden: int = 128,
                 layers: int = 2, attn_heads: int = 4,
                 keep_ratio: float = 0.6, tau: float = 2.0):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.input_proj = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList([
            SDSABlock(hidden, attn_heads, tau) for _ in range(layers)
        ])
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        functional.reset_net(self)
        x = self.input_proj(x)

        # Process through blocks with pruning
        for block in self.blocks:
            x = block(x)
            # Token pruning
            if self.keep_ratio < 1.0:
                scores = x.abs().sum(dim=-1)  # (batch, seq_len)
                k = int(x.size(1) * self.keep_ratio)
                _, indices = scores.topk(k, dim=1)
                x = torch.gather(x, 1, indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))

        # Mean pool + dual encoding simulation
        pooled = x.mean(dim=1)
        combined = torch.cat([pooled, pooled], dim=-1)  # simplified dual
        return self.fc(combined)
