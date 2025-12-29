import torch
from .replicate import replicate_tensor, replicate_model
from .util import ceil_div


class LeMA:
    def __init__(self, model_host, devices, opt_type="fp32"):
        assert (
            next(model_host.parameters()).device.type == "cpu"
        ), "Model should be on the host!"
        self.model_host = model_host
        self.devices = devices

        # Replicate the model on the given devices
        self.device_flat_params_map, self.device_params_map = replicate_model(
            model_host, self.devices
        )

    def step(
        self,
        input_host: torch.Tensor,
        target_host: torch.Tensor,
        slice_size: int = None,
    ):
        device_input_map = replicate_tensor(input_host, self.devices)
        device_target_map = replicate_tensor(target_host, self.devices)

    # @torch.no_grad()
    # def _forward(self, device_input_map):
    #     def forward_single(x, device):
    #         buffer = self.device_buffer_map[device]
    #         # print(buffer)

    #     #     print(device, x.device)
    #     #     print(len(buffer))
    #     #     for key, val in buffer.items():
    #     #         print(val.device, end=" ")
    #     #     print()
    #     #     # return torch.func.functional_call(self.model_host, buffer, x)
    #     #     return None

    #     device_output_map = {}
    #     for device in self.devices:
    #         x = device_input_map[device]
    #         device_output_map[device] = forward_single(x, device)
