import torch

def unflatten_params(flat_params, params_template):
    params_size = [p.numel() for _ , p in params_template.items()]
    params_list = torch.split(flat_params, params_size)
    params = {name : p.view_as(p_template) for p, (name, p_template) in zip(params_list, params_template.items())}
    return params
