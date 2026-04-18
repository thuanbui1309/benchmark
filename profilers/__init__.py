"""Profiling utilities for energy benchmarking."""

from .pytorch_profiler import profile_pytorch
from .spike_profiler import profile_spikes

__all__ = ["profile_pytorch", "profile_spikes"]
