#!/usr/bin/env python3
"""Energy benchmark for SpikeLog-v2 vs baselines.

Usage:
    uv run python benchmark.py                  # automated profiling (all models)
    uv run python benchmark.py --usb            # interactive USB power meter mode
    uv run python benchmark.py --usb --test     # USB mode, 5 inferences (test behavior)
    uv run python benchmark.py --usb --resume   # skip already-measured models
    uv run python benchmark.py --usb --only spikelog_v2  # single model
    uv run python benchmark.py --models deeplog neurallog  # specific models
    uv run python benchmark.py --device cuda    # use GPU
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Disable SDPA to prevent segfault on ARM (Pi 4)
os.environ.setdefault("PYTORCH_ENABLE_FLASH_SDP", "0")
os.environ.setdefault("PYTORCH_ENABLE_MEM_EFFICIENT_SDP", "0")

import torch
import yaml
from tabulate import tabulate
from tqdm import tqdm

from models import (
    DeepLog, LogAnomaly, LogRobust, LogCNN,
    NeuralLog, PLELog,
)
from models import SpikeLog, SpikeLogV2  # May be None if spikingjelly not installed
from profilers import profile_pytorch, profile_spikes


MODEL_REGISTRY = {
    "deeplog": (DeepLog, "ANN", "LSTM"),
    "loganomaly": (LogAnomaly, "ANN", "2×LSTM"),
    "logrobust": (LogRobust, "ANN", "BiLSTM+Attn"),
    "logcnn": (LogCNN, "ANN", "TextCNN"),
    "neurallog": (NeuralLog, "ANN", "Transformer"),
    "plelog": (PLELog, "ANN", "BiGRU+Attn"),
}

# Add SNN models if spikingjelly is available
if SpikeLog is not None:
    MODEL_REGISTRY["spikelog"] = (SpikeLog, "SNN", "RLeaky+LSTM")
if SpikeLogV2 is not None:
    MODEL_REGISTRY["spikelog_v2"] = (SpikeLogV2, "SNN", "SDSA+BSPN")

USB_RESULTS_FILE = "results/usb_benchmark.json"


def load_config(config_path: str = "config.yaml") -> dict:
    """Load benchmark configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_model(name: str, config: dict) -> torch.nn.Module:
    """Instantiate model from config."""
    model_cls, _, _ = MODEL_REGISTRY[name]
    model_cfg = config["models"].get(name, {})
    input_cfg = config["input"]

    if name == "deeplog":
        return model_cls(
            input_size=input_cfg["embed_dim"],
            hidden_size=model_cfg.get("hidden_size", 128),
            num_layers=model_cfg.get("num_layers", 2),
            vocab_size=model_cfg.get("vocab_size", 300),
        )
    elif name == "loganomaly":
        return model_cls(
            input_size=input_cfg["embed_dim"],
            hidden_size=model_cfg.get("hidden_size", 128),
            num_layers=model_cfg.get("num_layers", 2),
            vocab_size=model_cfg.get("vocab_size", 300),
        )
    elif name == "logrobust":
        return model_cls(
            input_size=input_cfg["embed_dim"],
            hidden_size=model_cfg.get("hidden_size", 128),
            num_layers=model_cfg.get("num_layers", 2),
        )
    elif name == "logcnn":
        return model_cls(
            embed_dim=input_cfg["embed_dim"],
            seq_len=input_cfg["seq_len"],
            out_channels=model_cfg.get("out_channels", 128),
            kernel_sizes=model_cfg.get("kernel_sizes", [3, 4, 5]),
        )
    elif name == "neurallog":
        return model_cls(
            dim_model=model_cfg.get("dim_model", input_cfg["embed_dim"]),
            num_heads=model_cfg.get("num_heads", 6),
            num_layers=model_cfg.get("num_layers", 1),
            dim_feedforward=model_cfg.get("dim_feedforward", 1200),
        )
    elif name == "plelog":
        return model_cls(
            input_size=input_cfg["embed_dim"],
            hidden_size=model_cfg.get("hidden_size", 128),
            num_layers=model_cfg.get("num_layers", 2),
        )
    elif name == "spikelog":
        return model_cls(
            input_size=input_cfg["embed_dim"],
            hidden_size=model_cfg.get("hidden_size", 128),
            num_out=model_cfg.get("num_out", 64),
        )
    elif name == "spikelog_v2":
        return model_cls(
            input_dim=input_cfg["embed_dim"],
            hidden=model_cfg.get("hidden", 128),
            layers=model_cfg.get("layers", 2),
            attn_heads=model_cfg.get("attn_heads", 4),
            keep_ratio=model_cfg.get("keep_ratio", 0.6),
        )
    else:
        raise ValueError(f"Unknown model: {name}")


def run_benchmark(
    models: list[str],
    config: dict,
    device: str = "cpu",
) -> list[dict]:
    """Run benchmark on selected models."""
    input_cfg = config["input"]
    bench_cfg = config["benchmark"]
    energy_cfg = config["energy"]

    # Create dummy input
    x = torch.randn(
        input_cfg["batch_size"],
        input_cfg["seq_len"],
        input_cfg["embed_dim"],
    )

    results = []
    for name in tqdm(models, desc="Benchmarking models"):
        if name not in MODEL_REGISTRY:
            print(f"[warn] Unknown model: {name}, skipping")
            continue

        model_cls, model_type, arch = MODEL_REGISTRY[name]
        print(f"[{name}] Benchmarking ({model_type}: {arch})...")

        try:
            model = create_model(name, config)

            # PyTorch profiling (latency, memory, FLOPs)
            pytorch_stats = profile_pytorch(
                model, x,
                n_warmup=bench_cfg["n_warmup"],
                n_runs=bench_cfg["n_runs"],
                device=device,
            )

            # Spike profiling for SNNs
            spike_stats = {}
            if model_type == "SNN":
                spike_stats = profile_spikes(
                    model, x,
                    n_runs=10,
                    device=device,
                    energy_config=energy_cfg,
                )

            # Compute energy estimate
            if model_type == "SNN" and spike_stats:
                energy_uj = spike_stats["energy_uj"]
                energy_ratio = spike_stats["energy_ratio"]
            else:
                # ANN energy from FLOPs
                fma_pj = energy_cfg["cmos_45nm"]["fma"]
                energy_pj = pytorch_stats["flops"] * fma_pj
                energy_uj = energy_pj / 1e6
                energy_ratio = 1.0  # baseline

            results.append({
                "model": name,
                "type": model_type,
                "arch": arch,
                "params": pytorch_stats["params"],
                "flops": pytorch_stats["flops"],
                "latency_ms": pytorch_stats["latency_ms"],
                "throughput": pytorch_stats["throughput"],
                "memory_mb": pytorch_stats["memory_mb"],
                "energy_uj": round(energy_uj, 4),
                "energy_ratio": round(energy_ratio, 1),
                "firing_rate": spike_stats.get("avg_firing_rate", None),
            })
            print(f"    Latency: {pytorch_stats['latency_ms']:.2f} ms, Energy: {energy_uj:.4f} µJ")

        except Exception as e:
            print(f"    [error] {e}")
            results.append({
                "model": name,
                "type": model_type,
                "arch": arch,
                "error": str(e),
            })

    return results


def print_results(results: list[dict]):
    """Print results as table."""
    headers = ["Model", "Type", "Arch", "Params", "Latency(ms)", "Energy(µJ)", "Ratio"]
    rows = []
    for r in results:
        if "error" in r:
            rows.append([r["model"], r["type"], r["arch"], "ERROR", "-", "-", "-"])
        else:
            rows.append([
                r["model"],
                r["type"],
                r["arch"],
                f"{r['params']:,}",
                f"{r['latency_ms']:.2f}",
                f"{r['energy_uj']:.4f}",
                f"{r['energy_ratio']:.1f}x",
            ])
    print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))


# ─── USB Power Meter Interactive Mode ────────────────────────────────────────


def _load_usb_results() -> dict:
    """Load existing USB benchmark results."""
    path = Path(__file__).parent / USB_RESULTS_FILE
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_usb_results(results: dict):
    """Save USB benchmark results incrementally."""
    path = Path(__file__).parent / USB_RESULTS_FILE
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def _prompt_usb_metrics() -> dict:
    """Prompt user to input USB power meter readings."""
    print("\n" + "-" * 40)
    print("  USB POWER METER READINGS")
    print("-" * 40)

    metrics = {}
    # Required fields
    for field, label, example in [
        ("wh", "Wh (energy)", "0.001"),
        ("w", "W  (power)", "2.35"),
        ("usb_time_s", "Time from USB (seconds)", "45"),
    ]:
        while True:
            raw = input(f"  {label} (e.g. {example}): ").strip()
            if not raw:
                print(f"    ⚠ {label} is required!")
                continue
            try:
                metrics[field] = float(raw)
                break
            except ValueError:
                print("    ⚠ Invalid number, try again")

    # Optional fields
    for field, label in [("v", "V+ (voltage)"), ("a", "A (current)"),
                         ("temp_c", "Temp (°C)")]:
        raw = input(f"  {label} [optional, Enter to skip]: ").strip()
        if raw:
            try:
                metrics[field] = float(raw)
            except ValueError:
                pass

    return metrics


def _run_usb_inference(name: str, model: torch.nn.Module,
                       x: torch.Tensor, n_inferences: int) -> dict:
    """Run timed inference loop for USB measurement."""
    model.eval()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Warmup (2 runs)
    with torch.no_grad():
        for _ in range(2):
            model(x)

    print(f"  ▶ Running {n_inferences} inferences...")
    print(f"    (Watch tqdm — prepare to note USB readings near the end)\n")

    start = time.perf_counter()
    with torch.no_grad():
        for _ in tqdm(range(n_inferences), desc=f"  {name}", unit="inf",
                      ncols=70):
            model(x)
    duration = time.perf_counter() - start

    ms_per_inf = (duration / n_inferences) * 1000
    print(f"\n  ✓ Done in {duration:.2f}s")
    print(f"    Latency: {ms_per_inf:.2f} ms/inference")
    print(f"    Throughput: {n_inferences / duration:.1f} inf/s")

    return {
        "params": params,
        "n_inferences": n_inferences,
        "duration_s": round(duration, 3),
        "ms_per_inference": round(ms_per_inf, 3),
    }


def _print_usb_summary(results: dict):
    """Print USB benchmark summary table."""
    if not results:
        return
    print(f"\n  {'Model':<15} {'Arch':<15} {'µJ/inf':>10} {'ms/inf':>10} {'W':>8}")
    print(f"  {'-' * 58}")
    for name, r in results.items():
        uj = r.get("derived", {}).get("energy_per_inference_uj", "?")
        ms = r.get("timing", {}).get("ms_per_inference", "?")
        w = r.get("usb_metrics", {}).get("w", "?")
        arch = r.get("arch", "?")
        uj_s = f"{uj:.3f}" if isinstance(uj, (int, float)) else str(uj)
        ms_s = f"{ms:.2f}" if isinstance(ms, (int, float)) else str(ms)
        w_s = f"{w:.2f}" if isinstance(w, (int, float)) else str(w)
        print(f"  {name:<15} {arch:<15} {uj_s:>10} {ms_s:>10} {w_s:>8}")
    print(f"\n  Results: {Path(__file__).parent / USB_RESULTS_FILE}")


def run_usb_benchmark(args, config: dict):
    """Interactive USB power meter benchmark."""
    device = "cpu"  # Pi 4 = CPU only
    n_inf = args.n_inferences
    results = _load_usb_results() if args.resume else {}

    # Select models
    if args.only:
        if args.only not in MODEL_REGISTRY:
            print(f"Error: '{args.only}' not found. Available: {', '.join(MODEL_REGISTRY.keys())}")
            return
        model_names = [args.only]
    else:
        model_names = list(MODEL_REGISTRY.keys())

    total = len(model_names)
    input_cfg = config["input"]
    x = torch.randn(input_cfg["batch_size"], input_cfg["seq_len"],
                     input_cfg["embed_dim"])

    print(f"\n{'=' * 60}")
    print(f"  USB POWER METER BENCHMARK")
    print(f"  Models: {total} | Inferences: {n_inf} | Mode: {'TEST' if args.test else 'FULL'}")
    print(f"  Input: ({input_cfg['batch_size']}, {input_cfg['seq_len']}, {input_cfg['embed_dim']})")
    print(f"{'=' * 60}", flush=True)

    for idx, name in enumerate(model_names, 1):
        model_cls, model_type, arch = MODEL_REGISTRY[name]

        # Skip if already measured
        if args.resume and name in results:
            print(f"\n  [{idx}/{total}] {name} — already measured, skipping")
            continue

        # Wait for user to prepare USB meter
        print(f"\n{'*' * 60}")
        print(f"  NEXT: [{idx}/{total}] {name} ({arch})")
        print(f"{'*' * 60}")
        print(f"\n  ➤ Reset USB meter (triple-press to zero Wh)")
        print(f"  ➤ Press Enter when ready to start...", flush=True)
        input()

        # Create model and run
        try:
            model = create_model(name, config).to(device)
        except Exception as e:
            print(f"\n  ✗ Failed to create {name}: {e}")
            print(f"  Skipping to next model...")
            continue

        try:
            timing = _run_usb_inference(name, model, x, n_inf)
        except Exception as e:
            print(f"\n  ✗ Inference failed for {name}: {e}")
            import traceback
            traceback.print_exc()
            print(f"  Skipping to next model...")
            continue

        # Collect USB readings
        print(f"\n  ➤ Now read your USB meter values:")
        usb_metrics = _prompt_usb_metrics()

        # Compute derived metrics
        wh = usb_metrics["wh"]
        energy_j = wh * 3600
        energy_per_inf_uj = (energy_j / n_inf) * 1e6

        derived = {
            "energy_total_j": round(energy_j, 6),
            "energy_per_inference_j": round(energy_j / n_inf, 9),
            "energy_per_inference_uj": round(energy_per_inf_uj, 3),
        }

        # Save
        entry = {
            "model": name, "type": model_type, "arch": arch,
            "timing": timing, "usb_metrics": usb_metrics, "derived": derived,
            "timestamp": datetime.now().isoformat(),
        }
        results[name] = entry
        _save_usb_results(results)

        print(f"\n  ┌─────────────────────────────────────")
        print(f"  │ {name}: {energy_per_inf_uj:.3f} µJ/inference")
        print(f"  │ Power: {usb_metrics['w']}W | Latency: {timing['ms_per_inference']:.2f}ms")
        print(f"  └─────────────────────────────────────")
        print(f"  ✓ Saved to {USB_RESULTS_FILE}")

        # Retry / next / quit
        while True:
            choice = input(
                f"\n  [n] Next model  [r] Retry {name}  [q] Quit\n  > "
            ).strip().lower()

            if choice == "r":
                print(f"\n  ↺ Retrying {name}...")
                print(f"  ➤ Reset USB meter, press Enter when ready...")
                input()
                timing = _run_usb_inference(name, model, x, n_inf)
                print(f"\n  ➤ Read USB meter values:")
                usb_metrics = _prompt_usb_metrics()
                wh = usb_metrics["wh"]
                energy_j = wh * 3600
                energy_per_inf_uj = (energy_j / n_inf) * 1e6
                derived = {
                    "energy_total_j": round(energy_j, 6),
                    "energy_per_inference_j": round(energy_j / n_inf, 9),
                    "energy_per_inference_uj": round(energy_per_inf_uj, 3),
                }
                entry = {
                    "model": name, "type": model_type, "arch": arch,
                    "timing": timing, "usb_metrics": usb_metrics, "derived": derived,
                    "timestamp": datetime.now().isoformat(),
                }
                results[name] = entry
                _save_usb_results(results)
                print(f"\n  ✓ Updated: {energy_per_inf_uj:.3f} µJ/inference")
                continue
            elif choice == "q":
                print("\n  Benchmark stopped. Progress saved.")
                _print_usb_summary(results)
                return
            else:
                break

    print(f"\n{'=' * 60}")
    print("  ALL MODELS COMPLETE!")
    print(f"{'=' * 60}")
    _print_usb_summary(results)


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Energy benchmark for log anomaly detection models")
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()),
                        help="Models to benchmark (automated mode)")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--output", help="Output JSON file")

    # USB interactive mode
    parser.add_argument("--usb", action="store_true",
                        help="Interactive USB power meter mode")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: 5 inferences per model (use with --usb)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-measured models (use with --usb)")
    parser.add_argument("--only", type=str,
                        help="Run only this model (use with --usb)")
    parser.add_argument("--n-inferences", type=int, default=1000,
                        help="Inferences per model for --usb mode (default: 1000)")
    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)
    config = load_config(str(config_path))

    # USB interactive mode
    if args.usb:
        if args.test:
            args.n_inferences = 5
            print("\n  ⚡ TEST MODE — 5 inferences per model\n")
        run_usb_benchmark(args, config)
        return

    # Automated profiling mode (original behavior)
    config["benchmark"]["device"] = args.device
    print(f"Device: {args.device}")
    print(f"Input shape: ({config['input']['batch_size']}, {config['input']['seq_len']}, {config['input']['embed_dim']})")
    print(f"Models: {', '.join(args.models)}\n")

    results = run_benchmark(args.models, config, device=args.device)
    print_results(results)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = args.output or str(results_dir / f"benchmark-{timestamp}.json")

    with open(output_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "device": args.device,
            "input_shape": [config["input"]["batch_size"], config["input"]["seq_len"], config["input"]["embed_dim"]],
            "n_runs": config["benchmark"]["n_runs"],
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
