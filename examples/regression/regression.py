from lema import LeMA
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


# Simple dense network for predicting regression
class SimpleDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_tanh_stack = nn.Sequential(nn.Linear(1, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, x):
        return self.linear_tanh_stack(x)


def main():
    # Prepare dataset
    x = torch.linspace(-1.0, 1.0, 100000, dtype=torch.float32, pin_memory=True).unsqueeze(1)
    y = torch.sinc(x)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=True)

    # Init model and optimizer
    model = SimpleDense().to(torch.device("cuda"))
    lema_optim = LeMA(model=model, residual_callable=lambda y_hat, y: y_hat - y)

    # Train step
    for epoch in range(0, 10):
        for x, y in loader:
            x = x.to(torch.device("cuda"))
            y = y.to(torch.device("cuda"))
            lema_optim.step(x, y)
            break
        break


if __name__ == "__main__":
    main()
