import argparse
from dataclasses import dataclass


@dataclass
class LeMADefaults:
    block_size: int
    shard_size: int
    slice_size: int
    damp_ratio: float


@dataclass
class TrainDefaults(LeMADefaults):
    epochs: int
    test_block_size: int


def add_lema_arguments(parser: argparse.ArgumentParser, defaults: LeMADefaults) -> argparse.ArgumentParser:
    parser.add_argument(
        "--block-size",
        type=int,
        default=defaults.block_size,
        metavar="N",
        help=f"number of samples per process for training",
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=defaults.shard_size,
        metavar="N",
        help=f"shard size for matrix computations",
    )

    parser.add_argument(
        "--slice-size",
        type=int,
        default=defaults.slice_size,
        metavar="N",
        help=f"slice size for Jacobian evaluation",
    )

    parser.add_argument(
        "--damp-ratio",
        type=float,
        default=defaults.damp_ratio,
        metavar="N",
        help=f"change ratio of the damping factor",
    )

    return parser


def add_train_arguments(parser: argparse.ArgumentParser, defaults: TrainDefaults) -> argparse.ArgumentParser:
    parser = add_lema_arguments(parser, defaults)

    parser.add_argument(
        "--epochs",
        type=int,
        default=defaults.epochs,
        metavar="N",
        help=f"number of epochs to train",
    )

    parser.add_argument(
        "--test-block-size",
        type=int,
        default=defaults.test_block_size,
        metavar="N",
        help=f"input block size for testing",
    )

    return parser
