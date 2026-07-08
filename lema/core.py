import cupynumeric as np
import torch
import torch.distributed as dist
import torch.cuda.nvtx as nvtx

from torch.nn.utils import parameters_to_vector
from typing import Callable, Any
from torch import nn
from .comm import gather_interop_1d, gather_interop_2d_row, torch_reduce_scalar
from legate.core import get_legate_runtime


class LeMAResults:
    def __init__(self, iterations: int, loss: float, damp_factor: float) -> None:
        self.iterations: int = iterations
        self.loss: float = loss
        self.damp_factor: float = damp_factor


class LeMA:
    def __init__(
        self,
        model: nn.Module,
        squared_residual_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        model_dtype: torch.dtype | None = None,
        optim_dtype: np.dtype = np.float32,
        max_iters: int = 10,
        damp_start: float = 1e-3,
        damp_ratio: float = 10.0,
        damp_min: float = 1e-9,
        damp_max: float = 1e9,
    ) -> None:
        ################################################################################
        # Type annotations
        ################################################################################
        self._model: nn.Module
        self._flat: torch.Tensor
        self._backup: torch.Tensor
        self._device: torch.device
        self._residual_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        self._loss_fn: Callable[[torch.Tensor, torch.Tensor], float | torch.Tensor]
        self._optim_dtype: np.dtype
        self._max_iters: int
        self._damp_start: float
        self._damp_ratio: float
        self._damp_min: float
        self._damp_max: float
        self._damp_cur: float
        self._rank: int
        self._world_size: int

        ################################################################################
        # Get process and device information
        ################################################################################
        rt = get_legate_runtime()
        self._rank = rt.node_id
        self._world_size = rt.node_count
        self._device = next(model.parameters()).device

        ################################################################################
        # Model setup
        ################################################################################
        self._model = model
        params = list(model.parameters())
        # Determine the data type of the model
        if model_dtype is None:
            model_dtype = params[0].dtype
        # Flatten the parameters
        self._flat = parameters_to_vector(params).to(dtype=model_dtype)
        self._flat.detach_()
        # Bind the parameters to the flat one
        offset = 0
        for p in model.parameters():
            size = p.numel()
            p.data = self._flat[offset : offset + size].view_as(p.data)
            offset += size
        # Broadcast rank 0's parameters
        dist.broadcast(self._flat, 0)
        # Backup the parameters
        self._backup = self._flat.clone()

        ################################################################################
        # Residual and loss functions
        ################################################################################
        eps = 1e-8
        self._residual_fn = lambda y_hat, y: torch.sqrt(torch.flatten(squared_residual_callable(y_hat, y)) + eps)
        self._loss_fn = lambda y_hat, y: torch.sum(squared_residual_callable(y_hat, y))

        ################################################################################
        # Optimizer configuration
        ################################################################################
        self._optim_dtype = optim_dtype
        self._max_iters = max_iters
        self._damp_start = damp_start
        self._damp_ratio = damp_ratio if damp_ratio >= 1.0 else 1.0 / damp_ratio
        self._damp_min = damp_min
        self._damp_max = damp_max
        self._damp_cur = damp_start

    @torch.no_grad()
    def step(self, x: torch.Tensor, y: torch.Tensor, slice_size: int = 64):
        nvtx.mark("Start step")
        local_batch_size, model_size = x.shape[0], self._flat.shape[0]
        batch_size = torch_reduce_scalar(local_batch_size, device=self._device)

        ################################################################################
        # Compute the Jacobian matrix
        ################################################################################
        with nvtx.range("Init Jacobian"):
            J = np.empty((batch_size, model_size), dtype=self._optim_dtype)
        # Compute the Jacobian slices
        for start in range(0, local_batch_size, slice_size):
            end = min(start + slice_size, local_batch_size)
            with nvtx.range("Compute Jacobian slice"):
                J_slice = self._compute_jacobian_slice(x[start:end], y[start:end])
            # Gather slices
            with nvtx.range("Gather Jacobian slice"):
                gather_interop_2d_row(J_slice, J, start, end)

        ################################################################################
        # Compute the residual
        ################################################################################
        with nvtx.range("Compute residual"):
            r = np.empty(batch_size, dtype=self._optim_dtype)
            r_slice = self._residual_fn(self._model(x), y)
        with nvtx.range("Gather residual"):
            gather_interop_1d(r_slice, r, 0, local_batch_size)

        ################################################################################
        # Build LMA equation
        ################################################################################
        with nvtx.range("Build equation"):
            JJ, rhs = self._build_equation(J, r)

        ################################################################################
        # Compute the initial loss
        ################################################################################
        with nvtx.range("Compute loss initial"):
            loss = self._compute_loss(x, y)

        ################################################################################
        # LMA iteration
        ################################################################################
        terminate = False
        damp_down_cnt = 0
        damp_up_cnt = 0
        for i in range(self._max_iters):
            with nvtx.range(f"Add damp {i}"):
                lhs = JJ + self._optim_dtype(self._damp_cur) * np.eye(JJ.shape[0], dtype=self._optim_dtype)
            with nvtx.range(f"Solve {i}"):
                delta = self._solve_equation(J, lhs, rhs)

            if delta is not None:
                # Update
                with nvtx.range(f"Update {i}"):
                    t_delta = torch.from_dlpack(delta, device=self._flat.device)
                self._flat.sub_(t_delta)

                # Check update criertia
                with nvtx.range(f"Compute loss {i}"):
                    new_loss = self._compute_loss(x, y)

                if new_loss < loss:
                    # Succeed in updating
                    loss = new_loss
                    self._damp_cur = max(self._damp_cur / self._damp_ratio, self._damp_min)
                    damp_down_cnt += 1
                    with nvtx.range(f"Save {i}"):
                        self._save_parameters()
                    break

                # Fail in updating
                with nvtx.range(f"Restore {i}"):
                    self._restore_parameters()

            # Fail in damping
            self._damp_cur = min(self._damp_cur * self._damp_ratio, self._damp_max)
            damp_up_cnt += 1

            # Check termination criteria
            if self._damp_cur >= self._damp_max:
                self._damp_cur = self._damp_start
                terminate = True
                break

        nvtx.mark("End step")

        return LeMAResults(
            iterations=i,
            loss=loss,
            damp_factor=self._damp_cur,
        )

    def _build_equation(self, J: np.ndarray, r: np.ndarray) -> np.ndarray:
        batch_size, model_size = J.shape
        if model_size > batch_size:
            JJ = J @ J.T
            rhs = r
        else:
            JJ = J.T @ J
            rhs = J.T @ r
        return JJ, rhs

    def _solve_equation(self, J: np.ndarray, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray | None:
        batch_size, model_size = J.shape
        try:
            delta = np.linalg.solve(lhs, rhs)
        except Exception as e:
            return None
        if model_size > batch_size:
            delta = J.T @ delta
        return delta

    @torch.no_grad()
    def _compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> float:
        batch_size = x.shape[0] * self._world_size
        loss = self._loss_fn(self._model(x), y)
        dist.all_reduce(loss)
        return float(loss.item()) / batch_size

    @torch.no_grad()
    def _save_parameters(self):
        self._backup.copy_(self._flat)

    @torch.no_grad()
    def _restore_parameters(self):
        self._flat.copy_(self._backup)

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
            return self._residual_fn(yy_hat, yy)

        # Determine the Jacobian's evaluation function
        m, n = x.shape[0], self._flat.shape[0]
        # TODO: jacrev may produce NaN
        jac_fn = torch.func.jacrev if m < n else torch.func.jacfwd
        return jac_fn(lambda p: compute_residual(p, x, y))(self._flat)
