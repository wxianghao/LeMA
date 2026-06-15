import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
from lema import LeMA


################################################################################
# CNN model for classifying images
################################################################################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


################################################################################
# Load MNIST dataset
################################################################################
def load_dataset(root_path: str, batch_size: int):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = torchvision.datasets.MNIST(root_path, train=True, download=False, transform=transform)
    testset = torchvision.datasets.MNIST(root_path, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader


################################################################################
# Residual function
################################################################################
def residual_fn(a, b):
    return torch.sqrt(torch.nn.functional.cross_entropy(a, b, reduction="none"))


################################################################################
# Training one epoch
################################################################################
# def train(args, model, device, train_loader, optimizer, epoch):
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print(
#                 "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
#                     epoch,
#                     batch_idx * len(data),
#                     len(train_loader.dataset),
#                     100.0 * batch_idx / len(train_loader),
#                     loss.item(),
#                 )
#             )
#             if args.dry_run:
#                 break


def main():
    # Train configuration
    epochs = 10

    # Prepare dataset, model and optimizer
    trainloader, testloader = load_dataset("./data", 128)
    device = torch.device("cuda")
    model = Net().to(device)
    optim = LeMA(model, residual_fn)

    # Train
    for epoch in range(1, epochs + 1):
        trainloader(None, model, device, trainloader, optim, epoch)


if __name__ == "__main__":
    main()
