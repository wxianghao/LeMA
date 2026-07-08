import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.distributed as dist
import argparse
import sys


from lema import LeMA
from legate.core import get_legate_runtime
from legate.timing import time
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from loguru import logger

torch.manual_seed(913)

""" Default values of Command-line arguments
"""
EPOCHS = 15
TOTAL_SIZE = 60_000
BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
SLICE_SIZE = 32
DAMP_RATIO = 10.0

""" Logging configuration
"""
LOG_PREFIX_FMT = "{time:YYYY-MM-DD HH:mm:ss.SSS}"


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
    test_kwargs = {"batch_size": args.test_batch_size}
    accel_kwargs = {"num_workers": 1, "persistent_workers": True, "pin_memory": True}
    train_kwargs.update(accel_kwargs)
    test_kwargs.update(accel_kwargs)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = torchvision.datasets.MNIST(root_path, train=True, download=False, transform=transform)
    trainsampler = DistributedSampler(trainset)
    testset = torchvision.datasets.MNIST(root_path, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, sampler=trainsampler, **train_kwargs)
    testloader = torch.utils.data.DataLoader(testset, **test_kwargs)
    return (trainloader, trainsampler), testloader


################################################################################
# Residual function
################################################################################
def squared_residual_fn(a, b):
    return F.nll_loss(a, b, reduction="none")


def main():

    ################################################################################
    # Parse command-line arguments
    ################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        metavar="N",
        help=f"number of epochs to train (default: {EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        metavar="N",
        help=f"input batch size for training (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=TEST_BATCH_SIZE,
        metavar="N",
        help=f"input batch size for testing (default: {TEST_BATCH_SIZE})",
    )
    parser.add_argument(
        "--slice-size",
        type=int,
        default=SLICE_SIZE,
        metavar="N",
        help=f"Jacobian row slice size (default: {SLICE_SIZE})",
    )
    parser.add_argument(
        "--damp-ratio",
        type=float,
        default=DAMP_RATIO,
        metavar="N",
        help=f"Change ratio of the damping factor (default: {DAMP_RATIO})",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help=f"Log file path (default: None)",
    )
    # parser.add_argument(
    #     "--csv",
    #     type=str,
    #     default=None,
    #     help=f"CSV file path (default: None)",
    # )
    parser.add_argument(
        "--precise",
        action="store_true",
        help=f"Enable the precise step time measurement",
    )
    args = parser.parse_args()

    ################################################################################
    # Setup the device and the process
    ################################################################################
    rank = get_legate_runtime().node_id
    world_size = get_legate_runtime().node_count
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, init_method="tcp://127.0.0.1:1234")
    device = torch.device("cuda")

    ################################################################################
    # Setup logging
    ################################################################################
    # Set logger
    logger.remove()
    if rank == 0:
        # Log into the standard output
        logger.add(sys.stdout, format=f"<green>{LOG_PREFIX_FMT}</green> | <level>{{message}}</level>", enqueue=True)
        if args.log is not None:
            # Log into the given file
            logger.add(args.log, format=f"{LOG_PREFIX_FMT} | {{message}}", enqueue=True)
    # Set log format
    epoch_width = len(str(args.epochs))
    batch_width = len(str(TOTAL_SIZE))
    iter_width = 2

    ################################################################################
    # Prepare the dataset
    ################################################################################
    (trainloader, trainsampler), testloader = load_dataset(args)

    ################################################################################
    # Initialze the model and the optimizer
    ################################################################################
    model = Net().to(device)
    optim = LeMA(
        model=model,
        squared_residual_callable=squared_residual_fn,
        damp_ratio=args.damp_ratio,
    )

    ################################################################################
    # Train
    ################################################################################
    for epoch in range(1, args.epochs + 1):
        trainsampler.set_epoch(epoch=epoch - 1)
        for batch_idx, (x, y) in enumerate(trainloader):
            x = x.to(device)
            y = y.to(device)

            if args.precise:
                tbegin = time()
                res = optim.step(x, y, args.slice_size)
                tend = time()
                duration = (tend - tbegin) * 1e-6
            else:
                res = optim.step(x, y, args.slice_size)

            # Logging
            if rank == 0:
                processed = (batch_idx + 1) * world_size * x.shape[0]
                log_dict = {
                    "epoch": f"{epoch:{epoch_width}d}",
                    "batch": f"{processed:{batch_width}d}/{TOTAL_SIZE:{batch_width}d}",
                    "iterations": f"{res.iterations:{iter_width}d}",
                    "loss": f"{res.loss:.3e}",
                    "damp_factor": f"{res.damp_factor:.3e}",
                }
                if args.precise:
                    log_dict["duration(s)"] = f"{duration:.3e}"

                # if args.csv is not None:
                log_msg = " | ".join([f"{key} {value}" for key, value in log_dict.items()])
                logger.info(log_msg)

    ################################################################################
    # Destroy the process
    ################################################################################
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
