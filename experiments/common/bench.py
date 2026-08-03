import gc
import torch

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator


@dataclass
class PeakMemoryMeasurement:
    """Peak additional CUDA memory measured by ``measure_peak_memory``."""

    peak_bytes: int = 0


@contextmanager
def measure_peak_memory(device: torch.device) -> Generator[PeakMemoryMeasurement, None, None]:
    """Measure peak additional CUDA memory used inside the context."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    measurement = PeakMemoryMeasurement()

    try:
        yield measurement
    finally:
        torch.cuda.synchronize(device)
        peak_memory = torch.cuda.max_memory_allocated(device) - baseline
        measurement.peak_bytes = max(0, peak_memory)
