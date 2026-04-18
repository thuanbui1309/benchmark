"""Spike counting profiler for SNN energy estimation."""

import torch
import torch.nn as nn
from collections import defaultdict

try:
    from spikingjelly.activation_based import neuron
    HAS_SPIKINGJELLY = True
except ImportError:
    HAS_SPIKINGJELLY = False


@torch.no_grad()
def profile_spikes(
    model: nn.Module,
    input_tensor: torch.Tensor,
    n_runs: int = 10,
    device: str = "cpu",
    energy_config: dict = None,
) -> dict:
    """Profile SNN model by counting spikes and estimating energy."""
    if energy_config is None:
        energy_config = {
            "loihi2": {"sop": 0.052, "neuron": 0.081, "routing": 0.025},
            "cmos_45nm": {"fma": 4.6},
        }

    model = model.to(device)
    model.eval()
    x = input_tensor.to(device)

    # Hook to capture spike outputs
    spike_counts = defaultdict(list)
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                # Count spikes (assuming binary or near-binary)
                spikes = (out > 0.5).float().sum().item()
                total = out.numel()
                spike_counts[name].append((spikes, total))
        return hook

    # Register hooks on LIF neurons
    lif_idx = 0
    for name, module in model.named_modules():
        if HAS_SPIKINGJELLY and isinstance(module, neuron.BaseNode):
            h = module.register_forward_hook(make_hook(f"lif_{lif_idx}"))
            hooks.append(h)
            lif_idx += 1

    # Run inference
    for _ in range(n_runs):
        spike_counts.clear()
        try:
            from spikingjelly.activation_based import functional
            functional.reset_net(model)
        except:
            pass
        _ = model(x)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Aggregate stats
    total_spikes = 0
    total_neurons = 0
    layer_stats = {}

    for name, counts in spike_counts.items():
        spikes = sum(c[0] for c in counts) / len(counts)
        neurons = sum(c[1] for c in counts) / len(counts)
        firing_rate = spikes / neurons if neurons > 0 else 0
        total_spikes += spikes
        total_neurons += neurons
        layer_stats[name] = {"spikes": int(spikes), "firing_rate": round(firing_rate, 4)}

    avg_firing_rate = total_spikes / total_neurons if total_neurons > 0 else 0

    # Energy estimation (Loihi 2)
    loihi = energy_config["loihi2"]
    hidden_dim = 128  # approximate fan-out
    total_sops = int(total_spikes * hidden_dim)
    energy_pj = (
        total_sops * loihi["sop"] +
        total_neurons * loihi["neuron"] +
        total_spikes * loihi["routing"]
    )

    # Compare to ANN baseline (FLOPs * FMA energy)
    cmos = energy_config["cmos_45nm"]
    # Rough ANN equivalent: assume similar param count
    batch, seq_len, dim = x.shape
    ann_flops = seq_len * dim * hidden_dim * 2 * 2  # simplified estimate
    ann_energy_pj = ann_flops * cmos["fma"]

    energy_ratio = ann_energy_pj / energy_pj if energy_pj > 0 else 0

    return {
        "total_spikes": int(total_spikes),
        "total_neurons": int(total_neurons),
        "avg_firing_rate": round(avg_firing_rate, 4),
        "total_sops": total_sops,
        "energy_pj": round(energy_pj, 2),
        "energy_uj": round(energy_pj / 1e6, 6),
        "ann_energy_pj": round(ann_energy_pj, 2),
        "energy_ratio": round(energy_ratio, 1),
        "layer_stats": layer_stats,
    }
