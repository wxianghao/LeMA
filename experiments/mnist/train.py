import torch
import torchvision
import torch.distributed as dist
import torchvision.transforms as transforms
import torch.nn.functional as F
import lema
import argparse
import os
import sys

from torch import nn
from torch.utils.data.distributed import DistributedSampler
from typing import cast, Optional
from loguru import logger

# Ensure corrent import
exp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if exp_dir not in sys.path:
    sys.path.insert(0, exp_dir)

from common.parser import (
    add_train_arguments,
    add_lema_arguments,
    add_profile_arguments,
)
from common.bench import measure_peak_memory, PeakMemoryMeasurement

TEST_INTERVAL = 1
DATA_DIR = ".data"
LOG_PREFIX_FMT = "{time:YYYY-MM-DD HH:mm:ss.SSS}"


class Net(nn.Module):
    """Simple convolutional neural network with"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def init_logger(rank: int, args):
    if rank != 0:
        return
    logger.remove()
    # Log into the standard output
    logger.add(sys.stdout, format=f"<green>{LOG_PREFIX_FMT}</green> | <level>{{message}}</level>", enqueue=True)
    if args.log is not None:
        open(args.log, "w").close()
        # Log into the given file
        logger.add(args.log, format=f"{LOG_PREFIX_FMT} | {{message}}", enqueue=True)


def log_train(rank: int, result: lema.LeMAResult, epoch: int, batch_start: int):
    if rank != 0:
        return
    step_method = "overdetermined" if result.overdetermined else "underdetermined"
    msg = "step info: epoch: {epoch} | batch: [{start}, {end}] | loss: {loss:.3e} | method: {method} | iterations: {iterations} | damp: {damp:.3e}".format(
        epoch=epoch,
        start=batch_start,
        end=batch_start + result.batch_size - 1,
        loss=result.loss,
        method=step_method,
        iterations=result.iterations,
        damp=result.damp,
    )
    logger.info(msg)


def log_epoch(rank: int, epoch: int, loss: float, mem_in_bytes: int):
    if rank != 0:
        return
    mem_in_gb = mem_in_bytes / 1e9
    msg = "epoch info | epoch: {epoch} | loss: {loss:.3e} | memory: {mem:.2f} GB".format(
        epoch=epoch,
        loss=loss,
        mem=mem_in_gb,
    )
    logger.info(msg)


def log_test(rank: int, epoch: int, loss: float, acc: float):
    if rank != 0:
        return
    msg = "test info | epoch: {epoch} | loss: {loss:.3e} | accuracy: {acc:.2%}".format(
        epoch=epoch,
        loss=loss,
        acc=acc,
    )
    logger.info(msg)


def load_dataset(args):
    root_path = args.data_dir
    train_kwargs = {"batch_size": args.block_size}
    test_kwargs = {"batch_size": args.test_block_size}
    accel_kwargs = {"num_workers": 1, "persistent_workers": True, "pin_memory": True}

    train_kwargs.update(accel_kwargs)
    test_kwargs.update(accel_kwargs)
    # Load dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = torchvision.datasets.MNIST(root_path, train=True, download=False, transform=transform)
    testset = torchvision.datasets.MNIST(root_path, train=False, transform=transform)
    # Create distributed samplers
    trainsampler = DistributedSampler(trainset)
    testsampler = DistributedSampler(testset)
    # Create loaders
    trainloader = torch.utils.data.DataLoader(trainset, sampler=trainsampler, **train_kwargs)
    testloader = torch.utils.data.DataLoader(testset, sampler=testsampler, **test_kwargs)
    return trainloader, testloader


def residual_fn(a, b):
    return torch.sqrt(F.nll_loss(a, b, reduction="none") + 1e-9)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = add_lema_arguments(
        parser,
        block_size=1024,
        shard_size=128,
        slice_size=32,
        damp_ratio=10.0,
    )
    parser = add_train_arguments(
        parser,
        epochs=10,
        test_block_size=1024,
    )
    parser = add_profile_arguments(
        parser,
        profile_steps=5,
    )
    parser.add_argument(
        "--test-interval",
        type=int,
        default=TEST_INTERVAL,
        metavar="N",
        help=f"Interval of test",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DATA_DIR,
        metavar="FILE",
        help=f"Dataset directory",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        metavar="FILE",
        help=f"Log file path (default: None)",
    )
    args = parser.parse_args()

    # Initialize process
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # Initialize logger
    init_logger(rank, args)

    # Load dataset
    trainloader, testloader = load_dataset(args)

    # Initialize the model
    device = torch.cuda.current_device()
    model = Net().to(device)
    optim = lema.LeMA(
        model=model,
        residual_fn=residual_fn,
        damp_ratio=args.damp_ratio,
    )

    # Log launch information and estimated memory usage
    batch_size = args.block_size * dist.get_world_size()
    model_size = optim._flat.numel()
    element_size = torch.tensor([], dtype=optim._optim_dtype).element_size()
    hessian_bytes = min(batch_size * batch_size, model_size * model_size) * element_size
    jacobian_shard_bytes = args.shard_size * model_size * element_size
    if rank == 0:
        logger.info(f"Model size: {model_size:,}")
        logger.info(f"Batch size: {batch_size:,}")
        logger.info(f"Hessian memory: {hessian_bytes / 1e9:.3f} GB")
        logger.info(f"Jacobian shard memory: {jacobian_shard_bytes / 1e9:.3f} GB")

    # Train
    nsteps = 0
    terminate = False
    for epoch in range(1, args.epochs + 1):
        cast(DistributedSampler, trainloader.sampler).set_epoch(epoch - 1)
        loss = 0.0
        num_samples = 0
        batch_start = 1
        with measure_peak_memory(device=device) as mem:  # Measure peak memory usage
            for x, y in trainloader:
                x = x.to(device)
                y = y.to(device)
                res = optim.step(x, y, shard_size=args.shard_size, slice_size=args.slice_size)
                loss += res.loss
                num_samples += res.batch_size
                terminate = res.terminate
                log_train(rank=rank, result=res, epoch=epoch, batch_start=batch_start)
                batch_start += res.batch_size
                nsteps += 1
                if args.profile and nsteps >= args.profile_steps:
                    terminate = True
                if terminate:
                    break
        log_epoch(rank=rank, epoch=epoch, loss=loss, mem_in_bytes=mem.peak_bytes)
        if terminate:
            break
        # Test
        if epoch % args.test_interval == 0:
            model.eval()
            loss = torch.zeros((), device=device)
            corrects = torch.zeros((), dtype=torch.long, device=device)
            num_samples = torch.zeros((), dtype=torch.long, device=device)
            with torch.no_grad():
                for x, y in testloader:
                    x = x.to(device)
                    y = y.to(device)
                    output = model(x)
                    loss += F.nll_loss(output, y, reduction="sum")
                    corrects += (output.argmax(dim=1) == y).sum()
                    num_samples += y.numel()
            dist.all_reduce(loss)
            dist.all_reduce(corrects)
            dist.all_reduce(num_samples)
            loss = loss.item() / num_samples.item()
            accuracy = corrects.item() / num_samples.item()
            log_test(rank=rank, epoch=epoch, loss=loss, acc=accuracy)
            model.train()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
