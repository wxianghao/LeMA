import torch
import torch.distributed as dist

from torch import nn
from torch.nn.utils import parameters_to_vector
from typing import Callable
from dataclasses import dataclass


@dataclass
class LeMAResult:
    loss: float
    iterations: int


class LeMA:
    def __init__(
        self,
        model: nn.Module,
        residual_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        model_dtype: torch.dtype | None = None,
        optim_dtype: torch.dtype = torch.float32,
        solver: str = "solve",
        max_iters: int = 10,
        damp_start: float = 1e-3,
        damp_ratio: float = 10.0,
        damp_min: float = 1e-9,
        damp_max: float = 1e9,
    ) -> None:
        # Check distributed environment
        if not dist.is_initialized():
            raise RuntimeError(
                "Default process group has not been initialized. Please make sure to call "
                "torch.distributed.init_process_group() before calling this function."
            )

        # Get process and device information
        self._rank: int = dist.get_rank()
        self._world_size: int = dist.get_world_size()
        self._device: torch.device = next(model.parameters()).device
        assert self._device.type == "cuda", f"Model should be on CUDA, but got {self._device.type}."

        # Setup the model and flatten the parameters
        self._model: nn.Module = model
        params = list(model.parameters())
        self._flat: torch.Tensor = parameters_to_vector(params).detach()
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

        # Aux variables for evaluating the Jacobian
        self._named_params = tuple(self._model.named_parameters())
        self._param_sizes = [param.numel() for _, param in self._named_params]

        # Config optimizer data type and solver
        self._optim_dtype: torch.dtype = optim_dtype

    def step(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> LeMAResult:
        # Check devices of tensors
        assert x.device == self._device, f"Tensor x should be on the device {self._device}, but got {x.device}."
        assert y.device == self._device, f"Tensor y should be on the device {self._device}, but got {y.device}."

        # Get size information
        block_size = x.size()[0]
        batch_size = self._world_size * block_size
        model_size = self._flat.size()[0]

        # Get slicing information
        row_slice_size = kwargs.get("row_slice_size", block_size)
        col_slice_size = kwargs.get("col_slice_size", model_size)

        # Determine the execution path
        overdetermined = batch_size > model_size
        if overdetermined:
            res = self._step_overdetermined(
                x,
                y,
                batch_size=batch_size,
                model_size=model_size,
                block_size=block_size,
                row_slice_size=row_slice_size,
                col_slice_size=col_slice_size,
            )
        else:
            res = self._step_underdetermined(
                x,
                y,
                batch_size=batch_size,
                model_size=model_size,
                block_size=block_size,
                row_slice_size=row_slice_size,
                col_slice_size=col_slice_size,
            )

        return res

    def _step_underdetermined(self, x: torch.Tensor, y: torch.Tensor, **kwargs):
        raise NotImplemented("`_step_underdetermined` is not implemented")
        # Get size information
        block_size = kwargs["block_size"]
        batch_size = kwargs["batch_size"]
        model_size = kwargs["model_size"]
        # Get slicing information
        row_slice_size = kwargs["row_slice_size"]
        col_slice_size = kwargs["col_slice_size"]

        # Allocate local products

    @torch.no_grad()
    def _step_overdetermined(self, x: torch.Tensor, y: torch.Tensor, **kwargs):
        # Get size information
        block_size = kwargs["block_size"]
        batch_size = kwargs["batch_size"]
        model_size = kwargs["model_size"]
        # Get slicing information
        row_slice_size = kwargs["row_slice_size"]
        col_slice_size = kwargs["col_slice_size"]

        # Allocate local products
        jtj = torch.zeros((model_size, model_size), device=self._device, dtype=self._optim_dtype)
        jtr = jtj.new_zeros(model_size)

        # Compute Jacobian row slices and accumulate the product to J.T @ J and J.T @ r
        for row_start in range(0, block_size, row_slice_size):
            row_end = min(block_size, row_start + row_slice_size)
            x_slice, y_slice = x[row_start:row_end], y[row_start:row_end]
            j_slice, res_slice = self._compute_jacobian_slice_reverse(x_slice, y_slice)

            # Reduce to the local results
            jtj.addmm_(j_slice.T, j_slice)
            jtr.addmv_(j_slice.T, res_slice)

        # All reduce the products
        dist.all_reduce(jtj)
        dist.all_reduce(jtr)

        # Pre-allocate buffer
        update = torch.empty_like(jtr)
        lhs = torch.empty_like(jtj)

        # LeMA iterations
        loss = self._loss_fn(self._model(x), y)
        dist.all_reduce(loss)
        loss = float(loss.item())
        iterations = 0
        while iterations < self._max_iters:
            iterations += 1
            # Solve the update
            lhs.copy_(jtj)
            lhs.diagonal().add_(self._damp_cur)
            torch.linalg.solve(lhs, jtr, out=update)
            # Attemp to update
            self._flat.sub_(update)
            # Check update criteria
            new_loss = self._loss_fn(self._model(x), y)
            dist.all_reduce(new_loss)
            new_loss = float(new_loss.item())
            if new_loss < loss:
                # Succeed in updating
                loss = new_loss
                self._damp_cur = max(self._damp_cur / self._damp_ratio, self._damp_min)
                self._backup.copy_(self._flat)
                break
            # Fail in updating
            self._flat.copy_(self._backup)
            self._damp_cur = min(self._damp_cur * self._damp_ratio, self._damp_max)
            # Check terminating
            if self._damp_cur >= self._damp_max:
                self._damp_cur = self._damp_start
                break

        return LeMAResult(loss=loss / batch_size, iterations=iterations)

    def _compute_jacobian_slice_reverse(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute a Jacobian slice using the reverse mode.
        """

        def compute_residual(flat):
            params = {
                name: tensor.view_as(param)
                for (name, param), tensor in zip(self._named_params, torch.split(flat, self._param_sizes))
            }
            y_hat = torch.func.functional_call(self._model, params, x)
            res = self._residual_fn(y_hat, y)
            return res, res

        return torch.func.jacrev(compute_residual, has_aux=True)(self._flat)

    def _compute_jacobian_slice_forward(
        self, x: torch.Tensor, y: torch.Tensor, param_start: int, param_end: int
    ) -> torch.Tensor:
        """
        Compute a Jacobian slice with respect to the parameters
        of the given range [param_start, param_ned], using the forward mode.
        """
        model_size = self._flat.numel()
        assert param_start >= 0 and param_end > param_start and param_end <= model_size
        flat_slice = self._flat[param_start:param_end]

        def compute_residual(param_slice: torch.Tensor) -> torch.Tensor:
            flat = torch.cat((self._flat[:param_start], param_slice, self._flat[param_end:]))
            params = {
                name: tensor.view_as(param)
                for (name, param), tensor in zip(self._named_params, torch.split(flat, self._param_sizes))
            }
            y_hat = torch.func.functional_call(self._model, params, (x,))
            res = self._residual_fn(y_hat, y)
            return res, res

        return torch.func.jacfwd(compute_residual, has_aux=True)(flat_slice)
