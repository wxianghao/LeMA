import cupynumeric as np
import torch

from legate.core import TaskContext, VariantCode, VariantOptions, get_legate_runtime, broadcast
from legate.core.task import task, OutputStore, InputStore

_PENDING: dict[int, tuple] = {}
_counter: int = 0


def _next_key(payload: tuple) -> int:
    global _counter
    key = _counter
    _counter += 1
    _PENDING[key] = payload
    return key


@task(variants=(VariantCode.GPU,))
def _gather_1d_task(ctx: TaskContext, dst: OutputStore, key: int):
    src, local_start, local_end = _PENDING.pop(key)
    t_dst = torch.from_dlpack(dst)
    t_dst[local_start:local_end] = src


@task(variants=(VariantCode.GPU,), constraints=(broadcast("dst", (1,)),))
def _gather_2d_row_task(ctx: TaskContext, dst: OutputStore, key: int):
    src, local_start, local_end = _PENDING.pop(key)
    t_dst = torch.from_dlpack(dst)
    t_dst[local_start:local_end, :] = src


def gather_interop_1d(src: torch.Tensor, dst: np.ndarray, local_start: int, local_end: int) -> None:
    _gather_1d_task(dst, _next_key((src, local_start, local_end)))


def gather_interop_2d_row(src: torch.Tensor, dst: np.ndarray, local_start: int, local_end: int) -> None:
    _gather_2d_row_task(dst, _next_key((src, local_start, local_end)))
