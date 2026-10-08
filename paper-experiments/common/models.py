import torch
import torch.nn.functional as F

from torch import nn
from typing import Optional


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


class LeNet5(nn.Module):
    """LeNet-5 with ReLU and max pooling for 28 x 28 inputs, with 61,706 parameters."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)


# Width of fc1 for each named WideCNN
PRESETS = {
    "cnn-1m": 128,  # 1,199,882 parameters, same as experiments/mnist/train.py
    "cnn-100m": 10_836,  # 100,002,598 parameters
    "cnn-1b": 108_378,  # 1,000,022,632 parameters
}
MODELS = [*PRESETS, "lenet5"]


def build_model(name: str, hidden: Optional[int] = None) -> nn.Module:
    """Build a named model, or a WideCNN with fc1 width hidden if given."""
    if hidden is not None:
        return WideCNN(hidden)
    if name == "lenet5":
        return LeNet5()
    return WideCNN(PRESETS[name])


def residual_fn(a, b):
    return torch.sqrt(F.nll_loss(a, b, reduction="none") + 1e-9)
