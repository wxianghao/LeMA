from lema import LeMA, get_available_gpus
import torch
import nvtx

torch.manual_seed(12)

class SimpleDense(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_tanh_stack = torch.nn.Sequential(
            torch.nn.Linear(784, 8), torch.nn.Tanh(), torch.nn.Linear(8, 10)
        )

    def forward(self, x):
        return self.linear_tanh_stack(x)


# Get devices
devices = get_available_gpus()
print('Model component is running on:')
for device in devices:
    print(f'\t{device}')
print()

# Prepare data
x = torch.randn(2000, 784, pin_memory=True)
w = torch.randn(784, 10)
y = (x @ w).pin_memory()

# Create model
residual_fn = lambda a, b: torch.sqrt(torch.sum(torch.square(a-b), axis=1))
model = SimpleDense()
lema_optim = LeMA(model, devices, residual_fn)
for device in devices:
    torch.cuda.synchronize(device)

# Test multiple steps
for _ in range(1):
    lema_optim.step(x, y)
    print()