import cupynumeric as np
import torch
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
from legate.core.task import task, OutputStore


class LeMA:
    def __init__(
        self,
        model: nn.Module,
        residual_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        model_dtype: torch.dtype | None = None,
        optim_dtype: np.dtype = np.float32,
    ) -> None:
        # Annotate members' types
        self._model: nn.Module
        self._flat: torch.Tensor
        self._backup: torch.Tensor
        self._residual_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        self._loss_fn: Callable[[torch.Tensor, torch.Tensor], float]
        self._optim_dtype: np.dtype
        self._rank: int
        self._world_size: int

        # Setup process info
        rt = get_legate_runtime()
        self._rank: int = rt.node_id
        self._world_size: int = rt.node_count

        # Setup the residual and the loss functions
        self._residual_fn = lambda y_hat, y: torch.flatten(residual_callable(y_hat, y))
        self._loss_fn = lambda y_hat, y: float(self._residual_fn(y_hat, y).square().mean())

        # Setup and flatten the model
        self._model = model
        model_dtype = next(model.parameters()).dtype if model_dtype is None else model_dtype
        # Extract parameters
        self._flat = parameters_to_vector(model.parameters()).to(dtype=model_dtype)
        self._flat.detach_()
        # Bind parameters to the flat one
        offset = 0
        for p in model.parameters():
            size = p.numel()
            offset_next = offset + size
            p.data = self._flat[offset:offset_next].view_as(p.data)
            offset = offset_next
        # Backup the flat parameter tensor
        self._backup = self._flat.clone()

        # Determine the optimizer's data type
        self._optim_dtype = optim_dtype

    @torch.no_grad()
    def step(self, x: torch.Tensor, y: torch.Tensor, slice_size: int = 64):
        # Initialize Jacobian matrix
        batch_size, model_size = x.shape[0], self._flat.shape[0]
        J = np.empty((batch_size, model_size), dtype=self._optim_dtype)

        def handle_slice(x_slice: torch.Tensor, y_slice: torch.Tensor, offset: int, cnt: int):
            # Compute Jacobian slice
            J_slice = self._compute_jacobian_slice(x_slice, y_slice)

            # Gather Jacobian slice
            # J[offset:offset+cnt, :] = J_slice
            @task(
                variants=(VariantCode.GPU,),
                options=VariantOptions(concurrent=True),
            )
            def gather_slice(ctx: TaskContext, dst_store: OutputStore) -> None:
                t_dst = torch.from_dlpack(dst_store)
                t_dst.copy_(J_slice)

            gather_slice(J)

        # Handle slices in a round-robin way
        offset, cur_rank = 0, 0
        while offset < batch_size:
            if self._rank == cur_rank:
                end = min(batch_size, offset + slice_size)
                handle_slice(x[offset:end], y[offset:end], offset, end - offset)
            cur_rank = (cur_rank + 1) % self._world_size
            offset += slice_size

        # # Split the input
        # samples_per_rank = (x.shape[0] + self._world_size - 1) // self._world_size
        # rank_start_idx = self._rank * samples_per_rank
        # rank_end_idx = min(x.shape[0], rank_start_idx + samples_per_rank)
        # x = x[rank_start_idx:rank_end_idx]
        # y = y[rank_start_idx:rank_end_idx]

        # # Determine the Jacobian's shape
        # batch_size = y.shape[0]
        # model_size = self._flat.shape[0]

        # # Calculate the output and the residual
        # y_hat = self._model(y)
        # r = self._residual_fn(y_hat, y)

        # # Compute the Jacobian matrix
        # J = self._compute_jacobian(x, y, slice_size=slice_size)

        # # Build equation
        # overdetermined = batch_size > model_size
        # if overdetermined:
        #     pass
        # else:
        #     pass

    # @torch.no_grad()
    # def _compute_jacobian(self, x: torch.Tensor, y: torch.Tensor, slice_size: int = 0) -> np.ndarray:
    #     # Determine the Jacobian size
    #     nsamples, nparams = x.shape[0], self._flat.shape[0]
    #     J = np.empty((nsamples, nparams), dtype=self._optim_dtype)

    #     # Determine the slice size
    #     slice_size = slice_size if slice_size > 0 else x.shape[0]

    #     # Compute the Jacobian matrix slice by slice
    #     slice_start_idx = 0
    #     while slice_start_idx < slice_size:
    #         slice_end_idx = min(x.shape[0], slice_start_idx + slice_size)
    #         xx = x[slice_start_idx:slice_end_idx]
    #         yy = y[slice_start_idx:slice_end_idx]

    #         # Compute slice
    #         J_slice = self._compute_jacobian_slice(xx, yy)

    #         # Send slice
    #         @task(
    #             variants=(VariantCode.GPU,),
    #             options=VariantOptions(concurrent=True),
    #         )
    #         def gather_to_legate(ctx: TaskContext, dst_store: OutputStore) -> None:
    #             pass

    #         slice_start_idx = slice_end_idx

    @torch.no_grad()
    def _compute_jacobian_slice(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Define stateless residual function
        def compute_residual(flat, xx, yy):
            params_list = torch.split(flat, [p.numel() for p in self._model.parameters()])
            params = {
                name: tensor.view_as(param)
                for (name, param), tensor in zip(dict(self._model.named_parameters()).items(), params_list)
            }
            yy_hat = torch.func.functional_call(self._model, params, xx)
            return self._residual_fn(yy, yy_hat)

        # Determine the Jacobian's evaluation function
        m, n = x.shape[0], self._flat.shape[0]
        jac_fn = torch.func.jacrev if m < n else torch.func.jacfwd
        return jac_fn(lambda p: compute_residual(p, x, y))(self._flat)
