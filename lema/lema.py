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
        self.residual_fn = residual_fn
        self.opt_type = _query_numpy_type(opt_type)

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
        
        # Determine the shapes
        batch_size = target_host.shape[0]
        model_size = self.params_size

        # Determine the slice size
        num_devices = len(self.devices)
        if slice_size is None:
            slice_size = ceil_div(batch_size, num_devices * DEFAULT_SLICE_PER_DEVICE)
        elif slice_size <= 0:
            raise ValueError('slice_size should be a positive integer!')

        # Compute each Jacobian slice
        J_slices = []
        start_idx = 0
        device_idx = 0
        while start_idx < batch_size:
            end_idx = start_idx + slice_size
            device = self.devices[device_idx]
            input, target = device_input_map[device], device_target_map[device]
            flat_params = self.device_flat_params_map[device]
            input_slice, target_slice = input[start_idx:end_idx], target[start_idx:end_idx]
            J = self._compute_jacobian_stateless(flat_params, input_slice, target_slice)
            J_slices.append((start_idx, end_idx, J))
            start_idx = end_idx
            device_idx = (device_idx + 1) % num_devices

        @task(variants=(VariantCode.GPU,))
        def send_to_cupynumeric(
            ctx: TaskContext, J_slice_legate: OutputArray
        ) -> None:
            J_slice_cupy = cupy.asarray(J_slice_legate)
            task_row, task_col = ctx.task_index
            assert task_row == 0
            start_col = task_col * (model_size // len(self.devices))
            end_col = start_col + model_size // len(self.devices)
            for start_row, end_row, J_slice_torch in J_slices:
                J_slice_cupy[start_row:end_row,:] = cupy.asarray(J_slice_torch)[:,start_col:end_col]

        # Transfer Jacobian slices to cuPyNumeric's GPU
        J_arr = np.empty((batch_size, model_size), self.opt_type)
        send_to_cupynumeric(J_arr)

    @torch.no_grad()
    def _forward(self, device_input_map):
        device_output_map = {}
        for device, input in device_input_map.items():
            params = self.device_params_map[device]
            output = torch.func.functional_call(self.model_host, params, input)
            device_output_map[device] = output

    def _compute_residual_stateless(self, flat_params, input, target):
        # Unflatten the flat parameters for stateless evaluation
        params = unflatten_params(flat_params, dict(self.model_host.named_parameters()))
        # Evaluate the output
        output = torch.func.functional_call(self.model_host, params, input)
        return self.residual_fn(output, target)

    @torch.no_grad()
    def _compute_jacobian_stateless(self, flat_params, input, target):
        f = lambda p: self._compute_residual_stateless(p, input, target)
        J = torch.func.jacrev(f)(flat_params)
        return J
