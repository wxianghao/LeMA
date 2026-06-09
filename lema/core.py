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
        max_iters: int = 10,
        damp_start: float = 1e-3,
        damp_ratio: float = 10.0,
        damp_min: float = 1e-9,
        damp_max: float = 1e9,
    ) -> None:
        # Annotate members' types
        self._model: nn.Module
        self._flat: torch.Tensor
        self._backup: torch.Tensor
        self._device: torch.device
        self._residual_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        # self._loss_fn: Callable[[torch.Tensor, torch.Tensor], float]
        self._optim_dtype: np.dtype
        self._max_iters: int
        self._damp_start: float
        self._damp_ratio: float
        self._damp_min: float
        self._damp_max: float
        self._damp_cur: float
        self._rank: int
        self._world_size: int

        # Setup process info
        rt = get_legate_runtime()
        self._rank = rt.node_id
        self._world_size = rt.node_count
        self._device = next(model.parameters()).device

        # Setup the residual and the loss functions
        self._residual_fn = lambda y_hat, y: torch.flatten(residual_callable(y_hat, y))
        # self._loss_fn = lambda y_hat, y: float(self._residual_fn(y_hat, y).square().mean())

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

        # Configure the optimizer options
        self._optim_dtype = optim_dtype
        self._max_iters = max_iters
        self._damp_start = damp_start
        self._damp_ratio = damp_ratio
        self._damp_min = damp_min
        self._damp_max = damp_max
        self._damp_cur = damp_start

    @torch.no_grad()
    def step(self, x: torch.Tensor, y: torch.Tensor, slice_size: int = 64):
        batch_size, model_size = x.shape[0], self._flat.shape[0]

        # Split the input
        samples_per_rank = (batch_size + self._world_size - 1) // self._world_size
        rank_start_idx = self._rank * samples_per_rank
        rank_end_idx = min(batch_size, rank_start_idx + samples_per_rank)
        x_slice = x[rank_start_idx:rank_end_idx]
        y_slice = y[rank_start_idx:rank_end_idx]

        # Compute the Jacobian matrix
        J = np.empty((batch_size, model_size), dtype=self._optim_dtype)
        for offset in range(rank_start_idx, rank_end_idx, slice_size):
            cnt = min(slice_size, rank_end_idx - offset)
            J_slice = self._compute_jacobian_slice(x[offset : offset + cnt], y[offset : offset + cnt])
            J[offset : offset + cnt, :] = np.asarray(J_slice)

        # Compute the residual
        r = np.empty(batch_size, dtype=self._optim_dtype)
        r_slice = np.array(self._residual_fn(self._model(x_slice), y_slice))
        r[rank_start_idx:rank_end_idx] = r_slice

        # # Build the LM equation
        # transformed = batch_size < model_size
        # if transformed:
        #     JJ = J @ J.T
        #     rhs = r
        # else:
        #     JJ = J.T @ J
        #     rhs = J.T @ J

        # terminating = False
        # loss = np.mean(np.square(r))
        # print(loss)
        # for i in range(self._max_iters):
        #     solved = False
        #     JJ_damped = JJ + self._damp_cur * np.eye(JJ.shape[0])
        #     try:
        #         delta = np.linalg.solve(JJ_damped, rhs)
        #         solved = True
        #     except Exception as e:
        #         pass

        #     if transformed:
        #         delta = J.T @ delta

        #     if solved:
        #         # Update
        #         self._flat.add_(torch.from_dlpack(delta, device=self._device))
        #         # Calculate the new loss
        #         r_slice_new = np.array(self._residual_fn(self._model(x_slice), y_slice))
        #         r[rank_start_idx:rank_end_idx] = r_slice_new
        #         new_loss = np.mean(np.square(r))
        #         print(new_loss)

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
        # TODO: jacrev may produce NaN
        jac_fn = torch.func.jacrev if m < n else torch.func.jacfwd
        return jac_fn(lambda p: compute_residual(p, x, y))(self._flat)
