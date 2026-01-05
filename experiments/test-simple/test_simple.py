from lema import LeMA, get_available_gpus
import torch
import nvtx

class SimpleDense(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_tanh_stack = torch.nn.Sequential(
            torch.nn.Linear(784, 512), torch.nn.Tanh(), torch.nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.linear_tanh_stack(x)


# Get devices
devices = get_available_gpus()
# print(devices)

# Prepare data
x = torch.randn(5000, 784, pin_memory=True)
y = torch.randn(5000, 10, pin_memory=True)

# Create model
residual_fn = lambda a, b: torch.sum(a - b, axis=1).flatten()
model = SimpleDense()
lema_optim = LeMA(model, devices, residual_fn)
for device in devices:
    torch.cuda.synchronize(device)

# Test multiple steps
# start_event = torch.cuda.Event()
# end_event = torch.cuda.Event()
for _ in range(2):
    # start_event.record()
    lema_optim.step(x, y)
    # end_event.record()
