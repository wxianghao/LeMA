import torch
import torch.distributed as dist

from torch import nn
from typing import Optional, Callable
from dataclasses import dataclass
from .jacobian import JacobianModel
from .util import iter_batches


@dataclass
class LeMAResult:
    loss: float
    iterations: int
    terminate: bool
    overdetermined: bool = False


class LeMA(JacobianModel):
    _rank: int
    _world_size: int
    _backup: torch.Tensor
    _loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    _max_iters: int
    _damp_start: float
    _damp_end: float
    _damp_min = float
    _damp_max = float
    _damp: float
    _device: torch.device
    _optim_dtype: torch.dtype = torch.float32
    _gloo_group: dist.group

    def __init__(
        self,
        model: nn.Module,
        residual_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        model_dtype: Optional[torch.dtype] = None,
        optim_dtype: torch.dtype = torch.float32,
        max_iters: int = 10,
        damp_start: float = 1e-3,
        damp_ratio: float = 10.0,
        damp_min: float = 1e-9,
        damp_max: float = 1e9,
    ) -> None:
        super().__init__(model=model, residual_fn=residual_fn, model_dtype=model_dtype)

        # Check distributed environment
        if not dist.is_initialized():
            raise RuntimeError(
                "Default process group has not been initialized. Please make sure to call "
                "torch.distributed.init_process_group() before calling this function."
            )

        # Get process and device information
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._device = next(model.parameters()).device

        # Initialize gloo group for scalar communication
        self._gloo_group = dist.new_group(ranks=None, backend="gloo")

        # Check model's device
        if self._device.type != "cuda":
            raise RuntimeError(f"Model should be on a CUDA device, but got {self._device.type}.")

        # Synchronize all processes' model parameters
        dist.broadcast(self._flat, 0)

        # Backup the parameters
        assert self._flat.requires_grad == False
        self._backup: torch.Tensor = self._flat.clone()

        # Configure the loss function
        self._loss_fn = lambda y_hat, y: torch.sum(self._residual_fn(y_hat, y).square())

        # Configure the damping strategy
        self._max_iters: int = max_iters
        self._damp_start: float = damp_start
        self._damp_ratio: float = damp_ratio if damp_ratio >= 1.0 else 1.0 / damp_ratio
        self._damp_min: float = damp_min
        self._damp_max: float = damp_max
        self._damp: float = damp_start

        # Config optimizer data type
        self._optim_dtype = optim_dtype
        self._template = torch.empty(0, device=self._device, dtype=self._optim_dtype)

    def step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        shard_size: Optional[int] = None,
        slice_size: Optional[int] = None,
    ) -> LeMAResult:
        # Check inputs' devices
        if x.device != self._device or y.device != self._device:
            raise RuntimeError(
                f"Rank {self._rank}: Input tensors should be on the device {self._device}, but got x:{x.device}, y: {y.device}."
            )

        # Get size information
        model_size = self._flat.size(0)
        block_size = x.size(0)
        # Calculate the batch size
        block_size_list = [None for _ in range(self._world_size)]
        dist.all_gather_object(block_size_list, block_size, group=self._gloo_group)
        batch_size = sum(block_size_list)

        # Choose the execution path
        overdetermined = False
        # overdetermined = batch_size > model_size
        if overdetermined:
            # Invoke the overdetermined step
            res = self._step_overdetermined(
                x,
                y,
                batch_size=batch_size,
                shard_size=shard_size,
                slice_size=slice_size,
            )
        else:
            # Calcualte the current process' offset in the batch
            block_start = sum(block_size_list[: self._rank])
            # Invoke the underdetermined step
            res = self._step_underdetermined(
                x,
                y,
                block_start=block_start,
                batch_size=batch_size,
                shard_size=shard_size,
                slice_size=slice_size,
            )

        res.overdetermined = overdetermined
        return res

    def _step_overdetermined(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_size: int,
        shard_size: Optional[int],
        slice_size: Optional[int],
    ) -> LeMAResult:
        # Get dimension information
        block_size = x.size(0)
        model_size = self._flat.size(0)
        # Get slicing information
        if shard_size is None:
            shard_size = block_size
        if slice_size is None:
            slice_size = shard_size

        # Allocate memory for products
        jtj = self._template.new_zeros((model_size, model_size))
        jtr = self._template.new_zeros(model_size)
        r_block = self._template.new_empty(block_size)
        # Reduce the products locally
        for start, end in iter_batches(block_size, shard_size):
            j_shard, r_shard = self.jacrev(
                x[start:end],
                y[start:end],
                has_residual=True,
                slice_size=slice_size,
            )
            jtj.addmm_(j_shard.T, j_shard)
            jtr.addmv_(j_shard.T, r_shard)
            r_block[start:end] = r_shard
        # All reduce the products
        dist.all_reduce(jtj)
        dist.all_reduce(jtr)

        # Compute the initial loss
        loss = r_block.square().sum().cpu()
        dist.all_reduce(loss, group=self._gloo_group)
        loss = loss.item()
        # Allocate memory for iterating
        update = self._template.new_empty(model_size)
        damped_jtj = self._template.new_empty((model_size, model_size))
        # LeMA iterations
        iterations = 0
        terminate = False
        while iterations < self._max_iters:
            iterations += 1
            # Solve and update
            damped_jtj.copy_(jtj)
            damped_jtj.diagonal().add_(self._damp)
            torch.linalg.solve(damped_jtj, jtr, out=update)
            self._flat.sub_(update)
            # Check update criterion
            with torch.no_grad():
                new_loss = self._loss_fn(self._model(x), y).cpu()
                dist.all_reduce(new_loss, group=self._gloo_group)
                new_loss = new_loss.item()
            if new_loss < loss:
                # Succeed
                loss = new_loss
                self._damp = max(self._damp / self._damp_ratio, self._damp_min)
                self._backup.copy_(self._flat)
                break
            # Fail
            self._flat.copy_(self._backup)
            self._damp = min(self._damp * self._damp_ratio, self._damp_max)
            # Check termination
            if self._damp >= self._damp_max:
                self._damp = self._damp_start
                terminate = True
                break

        return LeMAResult(loss=loss / batch_size, iterations=iterations, terminate=terminate)

    def _step_underdetermined(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        block_start: int,
        batch_size: int,
        shard_size: Optional[int],
        slice_size: Optional[int],
        # use_jvp: bool = False,
    ) -> LeMAResult:
        # Get dimension information
        block_size = x.size(0)
        model_size = self._flat.size(0)
        # Get slicing information
        if shard_size is None:
            shard_size = block_size
        if slice_size is None:
            slice_size = shard_size
        # All gather input tensors into a batch
        x_batch = x.new_empty((batch_size, *x.size()[1:]))
        y_batch = y.new_empty((batch_size, *y.size()[1:]))
        dist.all_gather_single(x_batch, x)
        dist.all_gather_single(y_batch, y)

        # Allocate memory for products
        jjt_block = self._template.new_empty((block_size, batch_size))
        jjt = self._template.new_empty((batch_size, batch_size))
        r = self._template.new_empty(batch_size)
        # Compute the products locally
        compute_residual = True
        for start, end in iter_batches(block_size, shard_size):
            shard1 = self.jacrev(
                x[start:end],
                y[start:end],
                slice_size=slice_size,
            )
            for col_start, col_end in iter_batches(batch_size, shard_size):
                shard = self.jacrev(
                    x_batch[col_start:col_end],
                    y_batch[col_start:col_end],
                    has_residual=compute_residual,
                    slice_size=slice_size,
                )
                if compute_residual:
                    shard2, r[col_start:col_end] = shard
                else:
                    shard2 = shard
                torch.mm(shard1, shard2.T, out=jjt_block[start:end, col_start:col_end])
            compute_residual = False
        # All gather the product and the residual
        dist.all_gather_single(jjt, jjt_block)

        # Compute the initial loss
        loss = r.square().sum()
        loss = float(loss.item())

        # Allocate memory for iterating
        damped_jtj = self._template.new_empty((batch_size, batch_size))
        solution = self._template.new_empty(batch_size)
        update = self._template.new_empty(model_size)
        # LeMA iterations
        iterations = 0
        terminate = False
        while iterations < self._max_iters:
            iterations += 1
            # Solve the equation
            damped_jtj.copy_(jjt)
            damped_jtj.diagonal().add_(self._damp)
            torch.linalg.solve(damped_jtj, r, out=solution)
            # Calculate the update
            update.zero_()
            for start, end in iter_batches(block_size, shard_size):
                update.add_(self.vjp(x[start:end], y[start:end], solution[block_start : block_start + block_size]))
            dist.all_reduce(update)
            self._flat.sub_(update)
            # Check update criterion
            with torch.no_grad():
                new_loss = self._loss_fn(self._model(x), y)
                dist.all_reduce(new_loss)
                new_loss = float(new_loss.item())
            if new_loss < loss:
                # Succeed
                loss = new_loss
                self._damp = max(self._damp / self._damp_ratio, self._damp_min)
                self._backup.copy_(self._flat)
                break
            # Fail
            self._flat.copy_(self._backup)
            self._damp = min(self._damp * self._damp_ratio, self._damp_max)
            # Check termination
            if self._damp >= self._damp_max:
                self._damp = self._damp_start
                terminate = True
                break

        return LeMAResult(loss=loss / batch_size, iterations=iterations, terminate=terminate)
