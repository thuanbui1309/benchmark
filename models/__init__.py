"""Baseline model stubs for energy benchmarking."""

from .deeplog import DeepLog
from .loganomaly import LogAnomaly
from .logrobust import LogRobust
from .logcnn import LogCNN
from .neurallog import NeuralLog
from .plelog import PLELog

__all__ = [
    "DeepLog",
    "LogAnomaly",
    "LogRobust",
    "LogCNN",
    "NeuralLog",
    "PLELog",
]

# Optional SNN models (require spikingjelly)
try:
    from .spikelog import SpikeLog
    from .spikelog_v2 import SpikeLogV2
    __all__.extend(["SpikeLog", "SpikeLogV2"])
except ImportError:
    SpikeLog = None
    SpikeLogV2 = None
