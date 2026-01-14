import cupynumeric as np
import legate
import torch
import cupy
from legate.core import TaskContext, VariantCode
from legate.core.task import task, InputArray, OutputArray
from .replicate import replicate_tensor, replicate_model
from .utils import unflatten_params, ceil_div
from .constant import DEFAULT_SLICE_PER_DEVICE

def _query_numpy_type(type_str):
    if type_str == 'fp32':
        return np.float32
    else:
        raise ValueError(f'Unsupported type: {type_str}!')

class LeMA:
    def __init__(self, model_host, devices, residual_fn, opt_type="fp32"):
        assert (
            next(model_host.parameters()).device.type == "cpu"
        ), "Model should be on the host!"
        self.model_host = model_host
        self.devices = devices
        self.opt_type = _query_numpy_type(opt_type)

        # Construct residual function with flattened output
        def flat_residual_fn(output, target):
            return torch.flatten(residual_fn(output, target))
        self.residual_fn = flat_residual_fn

        # Construct loss function
        def loss_fn(output, target):
            return self.residual_fn(output, target).square().mean()
        self.loss_fn = loss_fn

        # Replicate the model on the given devices
        self.device_flat_params_map, self.device_params_map = replicate_model(
            model_host, self.devices
        )
        
        # Get total number of parameters
        self.params_size = sum([p.numel() for p in self.model_host.parameters()])

    def step(
        self,
        input_host: torch.Tensor,
        target_host: torch.Tensor,
        slice_size: int = None,
    ):
        # Get all tensors for evaluating the Jacobian
        device_input_map = replicate_tensor(input_host, self.devices)
        device_target_map = replicate_tensor(target_host, self.devices)
        device_output_map = self._forward(device_input_map)
        residual = self._compute_residual(device_output_map, device_target_map)
        
        # Determine the shapes
        batch_size = target_host.shape[0]
        model_size = self.params_size

        # Determine the slice size
        num_devices = len(self.devices)
        if slice_size is None:
            slice_size = ceil_div(batch_size, num_devices * DEFAULT_SLICE_PER_DEVICE)
        elif slice_size <= 0:
            raise ValueError('slice_size should be a positive integer!')

        # Compute all the Jacobian slices
        J_slices = self._compute_jacobian_slices(device_input_map, device_target_map, slice_size, batch_size)

        # Gather Jacobian slices into cupynumeric
        @task(variants=(VariantCode.GPU,))
        def send_jacobian_to_cupynumeric(ctx: TaskContext, slice_legate: OutputArray):
            lo_row, lo_col = slice_legate.domain().lo
            _, hi_col = slice_legate.domain().hi
            assert lo_row == 0
            slice_cupy = cupy.asarray(slice_legate)
            for start_row, end_row, slice_torch in J_slices:
                # print(slice_cupy[start_row:end_row, :].shape, slice_torch[:, lo_col:hi_col+1].shape)
                slice_cupy[start_row:end_row, :] = cupy.asarray(slice_torch)[:, lo_col:hi_col+1]
        J = np.empty((batch_size, model_size), self.opt_type)
        send_jacobian_to_cupynumeric(J)

        # Send residual to cupynumeric
        @task(variants=(VariantCode.GPU,))
        def send_residual_to_cupynumeric(ctx: TaskContext, residual_legate: OutputArray):
            lo, = residual_legate.domain().lo
            hi, = residual_legate.domain().hi
            residual_cupy = cupy.asarray(residual_legate)
            residual_cupy[lo:hi+1] = cupy.asarray(residual)[lo:hi+1]
        r = np.empty(batch_size, self.opt_type)
        send_residual_to_cupynumeric(r)

        # # Build equation
        # if batch_size > model_size:
        #     JJ = J.T @ J
        #     rhs = J.T @ r
        # else:
        #     JJ = J @ J.T
        #     rhs = r

        # # Normalization
        # normalization_factor = 1.0 / batch_size
        # np.multiply(normalization_factor, JJ, out=JJ)
        # np.multiply(normalization_factor, rhs, out=rhs)

    @torch.no_grad()
    def _forward(self, device_input_map):
        device_output_map = {}
        for device, input in device_input_map.items():
            params = self.device_params_map[device]
            output = torch.func.functional_call(self.model_host, params, input)
            device_output_map[device] = output
        return device_output_map

    def _compute_residual(self, device_output_map, device_target_map):
        a_device = self.devices[0]
        output, target = device_output_map[a_device], device_target_map[a_device]
        return self.residual_fn(output, target)

    def _compute_residual_stateless(self, flat_params, input, target):
        # Unflatten the flat parameters for stateless evaluation
        params = unflatten_params(flat_params, dict(self.model_host.named_parameters()))
        # Evaluate the output
        output = torch.func.functional_call(self.model_host, params, input)
        return self.residual_fn(output, target)

    def _compute_jacobian_slices(self, device_input_map, device_target_map, slice_size, batch_size):
        slices = []
        start_idx = 0
        device_idx = 0
        num_devices = len(self.devices)
        while start_idx < batch_size:
            end_idx = start_idx + slice_size
            device = self.devices[device_idx]
            input, target = device_input_map[device], device_target_map[device]
            flat_params = self.device_flat_params_map[device]
            input_slice, target_slice = input[start_idx:end_idx], target[start_idx:end_idx]
            J_slice = self._compute_jacobian_stateless(flat_params, input_slice, target_slice)
            slices.append((start_idx, end_idx, J_slice))
            start_idx = end_idx
            device_idx = (device_idx + 1) % num_devices
        return slices

    @torch.no_grad()
    def _compute_jacobian_stateless(self, flat_params, input, target):
        f = lambda p: self._compute_residual_stateless(p, input, target)
        J = torch.func.jacrev(f)(flat_params)
        return J
