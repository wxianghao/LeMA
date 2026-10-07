import torch
import torch.nn.functional as F

from torch import nn


class WideCNN(nn.Module):
    """The CNN of experiments/mnist/train.py with a configurable width of fc1.

    Almost all parameters live in fc1 (9216 x hidden), so the model size scales
    linearly with the width: M = 18,826 + 9,227 * hidden.
    """

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

    @staticmethod
    def num_params(hidden: int) -> int:
        return 18_826 + 9_227 * hidden


# Width of fc1 for each named model
PRESETS = {
    "cnn-1m": 128,  # 1,199,882 parameters, same as experiments/mnist/train.py
    "cnn-100m": 10_836,  # 100,002,598 parameters
    "cnn-1b": 108_378,  # 1,000,022,632 parameters
}


def residual_fn(a, b):
    return torch.sqrt(F.nll_loss(a, b, reduction="none") + 1e-9)
