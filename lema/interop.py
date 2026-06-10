import cupynumeric as np
import torch

from legate.core import TaskContext, VariantCode, VariantOptions, get_legate_runtime, broadcast
from legate.core.task import task, OutputStore


def gather_interop_1d(src: torch.Tensor, dst: np.ndarray, local_start: int, local_end: int) -> None:
    @task(variants=(VariantCode.GPU,))
    def gather(ctx: TaskContext, dst: OutputStore):
        t_dst = torch.from_dlpack(dst)
        t_dst[local_start:local_end] = src

    gather(dst)


def gather_interop_2d_row(src: torch.Tensor, dst: np.ndarray, local_start: int, local_end: int) -> None:
    @task(variants=(VariantCode.GPU,), constraints=(broadcast("dst", (1,)),))
    def gather(ctx: TaskContext, dst: OutputStore):
        t_dst = torch.from_dlpack(dst)
        t_dst[local_start:local_end, :] = src

    gather(dst)
