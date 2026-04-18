"""PyTorch inference profiling: latency, memory, FLOPs estimation."""

import time
import torch
import torch.nn as nn
from tqdm import tqdm


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model: nn.Module, input_shape: tuple) -> int:
    """Rough FLOPs estimation based on layer types."""
    total_flops = 0
    batch, seq_len, dim = input_shape

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # FLOPs = 2 * in * out (multiply + add)
            total_flops += 2 * module.in_features * module.out_features * seq_len
        elif isinstance(module, nn.LSTM):
            # LSTM: 4 gates, each is linear + activation
            hidden = module.hidden_size
            inp = module.input_size
            layers = module.num_layers
            directions = 2 if module.bidirectional else 1
            # Per timestep: 4 * (input_size + hidden_size) * hidden_size * 2
            flops_per_step = 4 * (inp + hidden) * hidden * 2 * directions
            total_flops += flops_per_step * seq_len * layers
        elif isinstance(module, nn.GRU):
            hidden = module.hidden_size
            inp = module.input_size
            layers = module.num_layers
            directions = 2 if module.bidirectional else 1
            # GRU: 3 gates
            flops_per_step = 3 * (inp + hidden) * hidden * 2 * directions
            total_flops += flops_per_step * seq_len * layers
        elif isinstance(module, nn.Conv2d):
            # Simplified: kernel_h * kernel_w * in_channels * out_channels * output_size
            k_h, k_w = module.kernel_size
            out_h = seq_len - k_h + 1
            total_flops += k_h * k_w * module.in_channels * module.out_channels * out_h * 2
        elif isinstance(module, nn.TransformerEncoderLayer):
            # Attention: 4 * seq_len^2 * dim + FFN: 2 * seq_len * dim * 4 * dim
            total_flops += 4 * seq_len * seq_len * dim + 2 * seq_len * dim * 4 * dim

    return total_flops


@torch.no_grad()
def profile_pytorch(
    model: nn.Module,
    input_tensor: torch.Tensor,
    n_warmup: int = 10,
    n_runs: int = 100,
    device: str = "cpu",
) -> dict:
    """Profile PyTorch model inference."""
    model = model.to(device)
    model.eval()
    x = input_tensor.to(device)

    # Warmup
    for _ in range(n_warmup):
        _ = model(x)

    # Synchronize if CUDA
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # Measure latency
    times = []
    for _ in tqdm(range(n_runs), desc="  Inference", leave=False):
        start = time.perf_counter()
        _ = model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)  # ms

    # Stats
    latency_mean = sum(times) / len(times)
    latency_std = (sum((t - latency_mean) ** 2 for t in times) / len(times)) ** 0.5

    # Memory (rough estimate from parameters)
    n_params = count_parameters(model)
    mem_mb = n_params * 4 / (1024 * 1024)  # float32

    # FLOPs
    flops = estimate_flops(model, tuple(x.shape))

    return {
        "latency_ms": round(latency_mean, 3),
        "latency_std_ms": round(latency_std, 3),
        "params": n_params,
        "memory_mb": round(mem_mb, 2),
        "flops": flops,
        "throughput": round(1000 / latency_mean, 1),  # samples/sec
    }
