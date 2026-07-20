import cupynumeric as np
import torch
import torch.distributed as dist
import torch.cuda.nvtx as nvtx

from torch.nn.utils import parameters_to_vector
from typing import Callable, Any
from torch import nn
from .comm import torch_reduce_scalar
from legate.core import get_legate_runtime
from legate.core.data_interface import as_logical_array
from legate.core import TaskContext, VariantCode, VariantOptions, get_legate_runtime, broadcast
from legate.core.task import task, OutputStore, InputStore


class LeMAResults:
    def __init__(self, iterations: int, loss: float, damp_factor: float) -> None:
        self.iterations: int = iterations
        self.loss: float = loss
        self.damp_factor: float = damp_factor


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
        # Get process and device information
        runtime = get_legate_runtime()
        self._rank: int = runtime.node_id
        self._world_size: int = runtime.node_count
        self._device: torch.device = next(model.parameters()).device
        assert self._device.type == "cuda", f"Model should be on CUDA, but got {self._device.type}."

        # Setup the model and flatten the parameters
        self._model: nn.Module = model
        params = list(model.parameters())
        self._flat: torch.Tensor = parameters_to_vector(params)
        self._flat.detach()
        if model_dtype is not None:
            self._flat = self._flat.to(dtype=model_dtype)
        # Rebind the parameters
        offset = 0
        for p in params:
            size = p.numel()
            p.data = self._flat[offset : offset + size].view_as(p.data)
            offset += size
        # Synchronize all processes' parameters
        dist.broadcast(self._flat, 0)
        # Backup the parameters
        self._backup: torch.Tensor = self._flat.clone()

        # Setup the residual and the loss functions
        self._residual_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        self._loss_fn: Callable[[torch.Tensor, torch.Tensor], float | torch.Tensor]
        self._residual_fn = lambda y_hat, y: torch.flatten(residual_callable(y_hat, y))
        self._loss_fn = lambda y_hat, y: torch.sum(self._residual_fn(y_hat, y).square())

        # Configure the damping strategy
        self._max_iters: int = max_iters
        self._damp_start: float = damp_start
        self._damp_ratio: float = damp_ratio if damp_ratio >= 1.0 else 1.0 / damp_ratio
        self._damp_min: float = damp_min
        self._damp_max: float = damp_max
        self._damp_cur: float = damp_start

        # Determine the optimization data type
        self._optim_dtype: np.dtype = optim_dtype

    @torch.no_grad()
    def step(self, x: torch.Tensor, y: torch.Tensor, slice_size: int = 64):
        runtime = get_legate_runtime()
        block_size, model_size = x.shape[0], self._flat.shape[0]
        batch_size = torch_reduce_scalar(block_size, device=self._device)

        J = np.empty((batch_size, model_size), dtype=self._optim_dtype)
        R = np.empty(batch_size, dtype=self._optim_dtype)

        # @task(variants=(VariantCode.GPU,))
        # def compute_jacobian(ctx: TaskContext, dst: OutputStore):
        #     """Computea and store the Jacobian"""
        #     dst = torch.from_dlpack(dst, copy=False)
        #     for row_start in range(0, block_size, slice_size):
        #         row_end = min(block_size, row_start + slice_size)
        #         slice = self._compute_jacobian_slice(x[row_start:row_end], y[row_start:row_end])
        #         dst[row_start:row_end].copy_(slice)

        @task(variants=(VariantCode.GPU,))
        def compute_jacobian_overlap(ctx: TaskContext, jac_dst: OutputStore, res_dst: OutputStore):
            """Compute and store the Jacobian, while overlapping computing and storing operations"""
            compute_stream = torch.cuda.ExternalStream(ctx.task_stream)

            # Export legate slice to torch tensor
            with torch.cuda.stream(compute_stream):
                jac_dst = torch.from_dlpack(jac_dst, copy=False)
                res_dst = torch.from_dlpack(res_dst, copy=False)
                assert jac_dst.device == res_dst.device

            store_stream = torch.cuda.Stream(device=jac_dst.device)
            store_stream.wait_stream(compute_stream)

            for row_start in range(0, block_size, slice_size):
                row_end = min(block_size, row_start + slice_size)
                # Compute
                with torch.cuda.stream(compute_stream):
                    jac_slice, res_slice = self._compute_jacobian_slice(x[row_start:row_end], y[row_start:row_end])
                    compute_done = compute_stream.record_event()
                # Store
                with torch.cuda.stream(store_stream):
                    store_stream.wait_event(compute_done)
                    jac_dst[row_start:row_end].copy_(jac_slice)
                    res_dst[row_start:row_end].copy_(res_slice)
                    jac_slice.record_stream(store_stream)
                    res_slice.record_stream(store_stream)

            compute_stream.wait_stream(store_stream)

        jac_partition = as_logical_array(J).data.partition_by_tiling((block_size, model_size))
        res_partition = as_logical_array(R).data.partition_by_tiling((block_size,))
        task_ = runtime.create_manual_task(
            compute_jacobian_overlap.library,
            compute_jacobian_overlap.task_id,
            (self._world_size,),
        )
        task_.add_output(jac_partition)
        task_.add_output(res_partition)
        task_.execute()

        # Build LMA equation
        JJ, rhs = self._build_equation(J, R)

        # Start LMA iteration
        loss = self._compute_loss(x, y)
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
                t_delta = torch.from_dlpack(delta, device=self._flat.device)
                self._flat.sub_(t_delta)

                # Check update criertia
                new_loss = self._compute_loss(x, y)

                if new_loss < loss:
                    # Succeed in updating
                    loss = new_loss
                    self._damp_cur = max(self._damp_cur / self._damp_ratio, self._damp_min)
                    damp_down_cnt += 1
                    self._save_parameters()
                    break

                # Fail in updating
                self._restore_parameters()

            # Fail in damping
            self._damp_cur = min(self._damp_cur * self._damp_ratio, self._damp_max)
            damp_up_cnt += 1

            # Check termination criteria
            if self._damp_cur >= self._damp_max:
                self._damp_cur = self._damp_start
                terminate = True
                break

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
        jac_fn = torch.func.jacrev if m < n else torch.func.jacfwd

        def compute_residual_aux(flat):
            residual = compute_residual(flat, x, y)
            return residual, residual

        return jac_fn(compute_residual_aux, has_aux=True)(self._flat)
