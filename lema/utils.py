import torch

from legate.core import get_machine, TaskTarget
from typing import List

def ceil_div(a: int, b: int):
    return (a + b - 1) // b

def unflatten_params(flat_params, params_template):
    params_size = [p.numel() for _ , p in params_template.items()]
    params_list = torch.split(flat_params, params_size)
    params = {name : p.view_as(p_template) for p, (name, p_template) in zip(params_list, params_template.items())}
    return params

def get_available_gpus(num_devices=None) -> List[torch.device]:
    num_legate_gpus = get_machine().count(TaskTarget.GPU)
    num_total_gpus = torch.cuda.device_count()
    gpu_id_start = num_legate_gpus
    gpu_devices = []

    max_num_devices = num_total_gpus - gpu_id_start

    if num_devices is None:
        num_devices = max_num_devices
    elif num_devices > max_num_devices:
        raise ValueError(f'Max {max_num_devices} available, {num_devices} required!')
    
    for gpu_id in range(gpu_id_start, gpu_id_start + num_devices):
        gpu_devices.append(torch.device(gpu_id))

    return gpu_devices