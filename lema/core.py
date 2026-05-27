import cupynumeric as np
import torch
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector
import os

from typing import Tuple, Callable
from torch import nn
from legate.core import (
    TaskContext,
    VariantCode,
    VariantOptions,
    get_legate_runtime,
)

MASTER_ADDR = os.environ.get("MASTER_ADDR", "127.0.0.1")
TORCH_MASTER_PORT = int(os.environ.get("MASTER_PORT", "29500"))


class LeMA:
    @classmethod
    def begin_distribution(cls) -> Tuple[int, int]:
        rt = get_legate_runtime()

        # Get process information
        rank = rt.node_id
        world_size = rt.node_count

        # Initialize process gropu
        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://{MASTER_ADDR}:{TORCH_MASTER_PORT}",
        )

        return rank, world_size

    @classmethod
    def end_distribution(cls) -> None:
        dist.destroy_process_group()

    def __init__(
        self,
        model: nn.Module,
        residual_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        model_dtype: torch.dtype | None = None,
        optim_dtype: np.dtype = np.float32,
    ) -> None:
        self._model: nn.Module = model

        # Set up the residual and the loss functions
        self._residual_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
            lambda y_hat, y: torch.flatten(residual_callable(y_hat, y))
        )
        self._loss_fn: Callable[[torch.Tensor, torch.Tensor], float] = (
            lambda y_hat, y: float(self._residual_fn(y_hat, y).square().mean())
        )

        # Determine the model's data type
        if model_dtype is None:
            model_dtype = next(model.parameters()).dtype

        # Extract parameters into a flat tensor
        self._flat: torch.Tensor = parameters_to_vector(model.parameters()).to(
            dtype=model_dtype
        )
        self._flat.detach_()

        # Bind parameters to the flat one
        offset = 0
        for p in model.parameters():
            size = p.numel()
            offset_next = offset + size
            p.data = self._flat[offset:offset_next].view_as(p.data)
            offset = offset_next

        # Determine the optimizer's data type
        self._optim_dtype: np.dtype = optim_dtype

        # Backup the flat parameter tensor
        self._backup = self._flat.clone()

    @torch.no_grad()
    def step(self, x: torch.Tensor, y: torch.Tensor):

        # Determine the Jacobian's shape
        batch_size = y.shape[0]
        model_size = self._flat.shape[0]

        # Calculate the output and the residual
        y_hat = self._model(y)
        r = self._residual_fn(y_hat, y)

        # Compute the Jacobian matrix
        def compute_residual(flat, xx, yy):
            params_list = torch.split(
                flat, [p.numel() for p in self._model.parameters()]
            )
            params = {
                name: tensor.view_as(param)
                for (name, param), tensor in zip(
                    dict(self._model.named_parameters()).items(), params_list
                )
            }
            yy_hat = torch.func.functional_call(self._model, params, xx)
            return self._residual_fn(yy, yy_hat)

        jac_fn = torch.func.jacrev(lambda p: compute_residual(p, x, y))
        J = jac_fn(self._flat)

        # Build equation
        overdetermined = batch_size > model_size
        if overdetermined:
            pass
        else:
            pass
