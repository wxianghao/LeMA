import torch
import torch.distributed as dist
import argparse
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from legate.core import get_legate_runtime
from lema import LeMA
from loguru import logger

""" Default values of Command-line arguments
"""
EPOCHS = 15
TOTAL_SIZE = 20000
BATCH_SIZE = 2500
SLICE_SIZE = 500
LOG_INTERVAL = 5


def create_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(1, 20),
        nn.Tanh(),
        nn.Linear(20, 1),
    )


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
        "--total-size",
        type=int,
        default=TOTAL_SIZE,
        metavar="N",
        help=f"number of samples in the train dataset (default: {TOTAL_SIZE})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        metavar="N",
        help=f"input batch size for training (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--slice-size",
        type=int,
        default=SLICE_SIZE,
        metavar="N",
        help=f"Jacobian row slice size (default: {SLICE_SIZE})",
    )
    parser.add_argument(
        "--figure",
        type=str,
        default=None,
        help=f"figure export path (default: None)",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="log file path (default: None)",
    )
    args = parser.parse_args()

    ################################################################################
    # Setup the device and the process
    ################################################################################
    device = torch.device("cuda")
    rt = get_legate_runtime()
    rank = rt.node_id
    world_size = rt.node_count
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method="tcp://127.0.0.1:1234")

    ################################################################################
    # Setup logging
    ################################################################################
    logger.remove()
    if rank == 0 and args.log is not None:
        logger.add(args.log, format="{time:YYYY-MM-DD HH:mm:ss} | {message}", enqueue=True)
    epoch_width = len(str(args.epochs))
    batch_width = len(str(args.total_size))

    ################################################################################
    # Prepare the dataset
    ################################################################################
    datax = torch.linspace(-1, 1, args.total_size, dtype=torch.float32).unsqueeze(1)
    datay = torch.sinc(10.0 * datax)
    dataset = TensorDataset(datax, datay)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=1,
        persistent_workers=True,
        pin_memory=True,
    )

    ################################################################################
    # Initialze the model and the optimizer
    ################################################################################
    model = create_model().to(device)
    optim = LeMA(model, lambda x, y: x - y)

    ################################################################################
    # Train
    ################################################################################
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch - 1)
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            res = optim.step(x, y, args.slice_size)

            # Logging
            if rank == 0 and args.log is not None:
                processed = (batch_idx + 1) * world_size * x.shape[0]
                logger.info(
                    f"epoch {epoch:{epoch_width}d} | batch {processed:{batch_width}d}/{args.total_size:{batch_width}d} | iterations {res.iterations} | "
                    f"loss {res.loss:.3e} | damp_factor {res.damp_factor:.3e}",
                )

        # Validate and print epoch information
        if rank == 0:
            print(f"Train epoch: {epoch:{epoch_width}d}/{args.epochs:{epoch_width}d}\tLoss: {res.loss:.3e}")

    ################################################################################
    # Draw the result figure
    ################################################################################
    if args.figure is not None:
        xx = torch.linspace(-1.0, 1.0, 200).unsqueeze(1)
        yy = torch.sinc(10.0 * xx)
        plt.plot(xx, yy, c="b", label="data")
        plt.plot(xx, model(xx.to(device)).cpu().detach().numpy(), c="r", label="regression")
        plt.legend()
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.savefig(args.figure)

    ################################################################################
    # Destroy the process
    ################################################################################
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
