import cupynumeric as np
import legate
import legate.core as lg
import torch
import warnings
import legate.core as lg
from .replicate import replicate_tensor, replicate_model
from .utils import unflatten_params, ceil_div
from .constant import DEFAULT_SLICE_PER_DEVICE
from .comm import gather_to_cupynumeric, send_to_cupynumeric, broadcast_to_torch
from legate.timing import time

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

        # Damping
        self.damping_start = 1e-3
        self.damping_factor = self.damping_start
        self.damping_up = 10.0
        self.damping_down = 0.1
        self.damping_min = 1e-9
        self.damping_max = 1e9

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

        # Backup
        self.device_flat_params_backup_map = {device : None for device in self.devices}
        self._save_parameters()

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
        ranges, slices = self._compute_jacobian_slices(device_input_map, device_target_map, slice_size, batch_size)

        # Gather Jacobian slices into cupynumeric
        J = np.zeros((batch_size, model_size), self.opt_type)
        gather_to_cupynumeric(ranges, slices, J)

        # Send residual to cupynumeric
        r = np.empty(batch_size, self.opt_type)
        send_to_cupynumeric(residual, r)

        # Build equation
        JJ = J.T @ J
        rhs = J.T @ r

        # Normalization
        normalization_factor = 1.0 / batch_size
        np.multiply(normalization_factor, JJ, out=JJ)
        np.multiply(normalization_factor, rhs, out=rhs)

        # Step loop 
        terminating = False
        loss_val = self._compute_loss(device_output_map, device_target_map)
        for i in range(5):
            solved = False
            # Solve update
            try:
                JJ_damped = JJ + self.damping_factor * np.diag(np.diag(JJ))
                update = np.linalg.solve(JJ_damped, rhs)
                solved = True
            except Exception as e:
                warnings.warn(f"Singular matrix occurs: damping_factor={self.damping_factor}")

            # Success in solving update
            if solved:
                # Update
                self._update_parameters(update)
                # Check update criteria
                device_output_map = self._forward(device_input_map)
                new_loss_val = self._compute_loss(device_output_map, device_target_map)
                if new_loss_val < loss_val:
                    # Success in updating
                    loss_val = new_loss_val
                    self.damping_factor = max(self.damping_factor * self.damping_down, self.damping_min)
                    self._save_parameters()
                    break
                else:
                    # Fail in updating
                    warnings.warn('Failed attempt due to increasing loss')
                    self._restore_parameters()
            
            # Fail in damping
            self.damping_factor = self.damping_factor * self.damping_up
            # Check termination criteria
            if self.damping_factor >= self.damping_max:
                warnings.warn("Terminated due to large damping factor")
                self.damping_factor = self.damping_start
                break

        return loss_val, terminating


    @torch.no_grad()
    def _forward(self, device_input_map):
        device_output_map = {}
        for device, input in device_input_map.items():
            params = self.device_params_map[device]
            output = torch.func.functional_call(self.model_host, params, input)
            device_output_map[device] = output
        return device_output_map

    def _compute_loss(self, device_output_map, device_target_map):
        a_device = self.devices[0]
        output, target = device_output_map[a_device], device_target_map[a_device]
        return self.loss_fn(output, target)

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
        ranges, slices = [], []
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
            ranges.append((start_idx, end_idx))
            slices.append(J_slice)
            start_idx = end_idx
            device_idx = (device_idx + 1) % num_devices
        return ranges, slices

    @torch.no_grad()
    def _compute_jacobian_stateless(self, flat_params, input, target):
        f = lambda p: self._compute_residual_stateless(p, input, target)
        J = torch.func.jacrev(f)(flat_params)
        return J

    @torch.no_grad()
    def _save_parameters(self):
        for device, p in self.device_flat_params_map.items():
            self.device_flat_params_backup_map[device] = p.clone()

    @torch.no_grad()
    def _restore_parameters(self):
        for device, backup in self.device_flat_params_backup_map.items():
            self.device_flat_params_map[device].copy_(backup)

    @torch.no_grad()
    def _update_parameters(self, update):
        # rt = lg.get_legate_runtime()
        device_update_map = {device : torch.empty(self.params_size, device=device) for device in self.devices} # TODO: specify dtype
        # rt.issue_execution_fence()
        broadcast_to_torch(update, device_update_map)
        # rt.issue_execution_fence()
        # time()

        for device, p in self.device_flat_params_map.items():
            p.add_(-device_update_map[device])

