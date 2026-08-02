import torch

import argparse
import gc
import statistics
import time


def measure_peak_memory(fn, device: torch.device) -> int:
    """Measure peak additional CUDA memory used by one invocation."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    output = fn()
    torch.cuda.synchronize(device)
    peak_memory = torch.cuda.max_memory_allocated(device) - baseline
    del output
    return max(0, peak_memory)
