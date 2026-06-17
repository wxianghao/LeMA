import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
import torch.distributed as dist


from lema import LeMA
from legate.core import get_legate_runtime
from torch import nn
from torch.utils.data.distributed import DistributedSampler


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
def load_dataset(args):
    root_path = ".data"

    train_kwargs = {"batch_size": args.batch_size}
    # test_kwargs = {"batch_size": args.test_batch_size}
    accel_kwargs = {"num_workers": 1, "persistent_workers": True, "pin_memory": True}
    train_kwargs.update(accel_kwargs)
    # test_kwargs.update(accel_kwargs)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = torchvision.datasets.MNIST(root_path, train=True, download=False, transform=transform)
    trainsampler = DistributedSampler(trainset)
    # testset = torchvision.datasets.MNIST(root_path, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, sampler=trainsampler, **train_kwargs)
    # testloader = torch.utils.data.DataLoader(testset, **test_kwargs)
    # return trainloader, testloader
    return (trainloader, trainsampler)


################################################################################
# Residual function
################################################################################
def residual_fn(a, b):
    return torch.sqrt(F.nll_loss(a, b, reduction="none"))


################################################################################
# Train function for one epoch
################################################################################
def train(args, rank, model, device, dataloader, optimizer, epoch):
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        terminate, res = optimizer.step(data, target, args.slice_size)
        loss = res["loss"]

        if rank == 0 and batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(dataloader.dataset),
                    100.0 * (batch_idx + 1) / len(dataloader),
                    loss,
                )
            )


################################################################################
# Test function
################################################################################
# def test(model, device, test_loader):
#     model.eval()
#     test_loss = 0.0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()
#     test_loss /= len(test_loader.dataset)
#     print(
#         "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
#             test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
#         )
#     )


################################################################################
# Main function
################################################################################
def main():
    # Train configuration
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        metavar="N",
        help="input batch size for training (default: 1024)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--slice-size",
        type=int,
        default=64,
        metavar="N",
        help="Jacobian row slice size (default: 64)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    args = parser.parse_args()

    # Get process info
    rank = get_legate_runtime().node_id
    world_size = get_legate_runtime().node_count

    # Initialize process
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, init_method="tcp://127.0.0.1:1234")

    # Prepare dataset, model and optimizer
    trainloader, trainsampler = load_dataset(args)
    device = torch.device("cuda")
    model = Net().to(device)
    optim = LeMA(model, residual_fn)

    # Print basic information
    if rank == 0:
        print(f"Number of processes: {world_size}")
        print(f"Batch size: {args.batch_size}")
        print(f"Slice size: {args.slice_size}")

    # Train
    for epoch in range(1, args.epochs):
        trainsampler.set_epoch(epoch=epoch - 1)
        train(args, rank, model, device, trainloader, optim, epoch)
        # test(model, device, testloader)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
