import torch
import torch.distributed as dist


def ceil_div(a: int, b: int):
    return (a + b - 1) // b


def iter_batches(total_size: int, batch_size: int):
    for start in range(0, total_size, batch_size):
        yield start, min(total_size, start + batch_size)
