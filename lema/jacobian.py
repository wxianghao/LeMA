import torch

from torch import nn
from typing import Callable, Tuple, Optional


class JacobianModel(nn.Module):
    """
    Module wrapper supporting a series of Jacobian computations.

    The Jacobian is defined as the derivative of _residual_fn(_model(x), y)
    with respect to _flat.
    """

    _model: nn.Module
    _flat: torch.Tensor
    _residual_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    _named_params: Tuple[Tuple[str, nn.Parameter], ...]
    _param_sizes: Tuple[int, ...]

    def __init__(
        self,
        model: nn.Module,
        residual_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        model_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self._model = model

        # Flatten the model parameters
        params = list(model.parameters())
        self._flat = nn.utils.parameters_to_vector(params).detach()
        if model_dtype is not None:
            self._flat = self._flat.to(dtype=model_dtype)
        offset = 0
        for param in params:
            size = param.numel()
            param.data = self._flat[offset : offset + size].view_as(param)
            offset += size

        # Parameter structure information
        self._named_params = tuple(self._model.named_parameters())
        self._param_sizes = tuple(param.numel() for _, param in self._named_params)

        # Setup the residual function
        self._residual_fn = lambda y_hat, y: residual_fn(y_hat, y).flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def _apply(self, fn, recurse=True):
        raise RuntimeError(
            "JacobianModel does not support device or dtype conversion. "
            "Move the model before constructing JacobianModel."
        )

    def _stateless_residual(self, flat: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        params = {
            name: tensor.view_as(param)
            for (name, param), tensor in zip(self._named_params, torch.split(flat, self._param_sizes))
        }
        y_hat = torch.func.functional_call(self._model, params, (x,))
        return self._residual_fn(y_hat, y)

    @torch.no_grad()
    def jvp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute the Jacobian-vector product J @ v."""
        compute_residual = lambda p: self._stateless_residual(p, x, y)
        _, jv = torch.func.jvp(compute_residual, (self._flat,), (v,))
        return jv

    @torch.no_grad()
    def vjp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute the vector-Jacobian product J.T @ v."""
        compute_residual = lambda p: self._stateless_residual(p, x, y)
        _, vjp_fn = torch.func.vjp(compute_residual, self._flat)
        (jtv,) = vjp_fn(v)
        return jtv

    @torch.no_grad()
    def jacrev(
        self, x: torch.Tensor, y: torch.Tensor, has_residual: bool = False, slice_size: Optional[int] = None
    ) -> torch.Tensor:
        """Compute the Jacobian matrix using the reverse mode"""

        def compute_single_jacobian(p, xi, yi):
            def compute_single_residual(flat):
                return self._stateless_residual(
                    flat,
                    xi.unsqueeze(0),
                    yi.unsqueeze(0),
                ).reshape(())

            res, vjp_fn = torch.func.vjp(compute_single_residual, p)
            (jacobian_row,) = vjp_fn(torch.ones_like(res))
            return (jacobian_row, res) if has_residual else jacobian_row

        return torch.vmap(
            compute_single_jacobian,
            in_dims=(None, 0, 0),
            randomness="different",
            chunk_size=slice_size,
        )(self._flat, x, y)

    @torch.no_grad()
    def jacfwd(self, x: torch.Tensor, y: torch.Tensor, has_residual=False) -> torch.Tensor:
        """Compute the Jacobian matrix using the forward mode"""

        def compute_residual(p):
            res = self._stateless_residual(p, x, y)
            return (res, res) if has_residual else res

        return torch.func.jacfwd(compute_residual, has_aux=has_residual)(self._flat)

    # @torch.no_grad()
    # def jacfwd_partial(
    #     self, x: torch.Tensor, y: torch.Tensor, param_start: int, param_end: int, has_residual: bool = False
    # ):
    #     """Compute the Jacobian matrix with respect to partial parameters, using the forward mode"""
    #     if param_start < 0 or param_end > self._flat.size(0) or param_start >= param_end:
    #         raise ValueError(f"Got illegal param_start = {param_start} and param_end = {param_end}.")

    #     flat_slice = self._flat[param_start:param_end]

    #     def compute_residual(param_slice: torch.Tensor) -> torch.Tensor:
    #         flat = torch.cat((self._flat[:param_start], param_slice, self._flat[param_end:]))
    #         params = {
    #             name: tensor.view_as(param)
    #             for (name, param), tensor in zip(self._named_params, torch.split(flat, self._param_sizes))
    #         }
    #         y_hat = torch.func.functional_call(self._model, params, (x,))
    #         res = self._residual_fn(y_hat, y)
    #         return (res, res) if has_residual else res

    #     return torch.func.jacfwd(compute_residual, has_aux=has_residual)(flat_slice)
