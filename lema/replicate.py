import torch
import torch.nn.parallel.comm as comm

from typing import List


def _bind_flat_params(flat_params, params, params_host):
    start_idx = 0
    for name, p_host in params_host.items():
        p_device = params[name]
        end_idx = start_idx + p_host.numel()
        p_device.data = flat_params[start_idx:end_idx]
        start_idx = end_idx


def replicate_model(model_host: torch.nn.Module, devices: List[torch.device]):
    # TODO: currently, we do not support layers with non-empty .buffers()
    params_host = dict(model_host.named_parameters())

    if not next(model_host.parameters()).device.type == "cpu":
        raise RuntimeError("Base model should be on the host!")

    # Collect flat parameters, store it in pinned memory and distribute to GPUs
    flat_params_host = torch.cat(
        [p.detach().flatten() for p in params_host.values()]
    ).pin_memory()
    device_flat_params_map = replicate_tensor(flat_params_host, devices)

    # Bind flat parameters to "shaped" parameters
    device_params_map = {}
    for device, flat_params in device_flat_params_map.items():
        params = {
            name: torch.nn.parameter.Parameter()
            for name, p in params_host.items()
            if p.requires_grad
        }
        _bind_flat_params(flat_params, params, params_host)
        device_params_map[device] = params

    return device_flat_params_map, device_params_map


def replicate_tensor(tensor_host: torch.Tensor, devices: List[torch.device]):
    if not tensor_host.is_pinned():
        # TODO: warn that pinned memory should to be used
        tensor_host = tensor_host.pin_memory()
    tensor_replicas = comm.broadcast(tensor_host, devices)
    return dict(zip(devices, tensor_replicas))
