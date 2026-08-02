import argparse
from dataclasses import dataclass


def add_lema_arguments(
    parser: argparse.ArgumentParser,
    block_size: int,
    shard_size: int,
    slice_size: int,
    damp_ratio: float,
    **kwargs,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--block-size",
        type=int,
        default=block_size,
        metavar="N",
        help=f"number of samples per process for training",
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=shard_size,
        metavar="N",
        help=f"shard size for matrix computations",
    )

    parser.add_argument(
        "--slice-size",
        type=int,
        default=slice_size,
        metavar="N",
        help=f"slice size for Jacobian evaluation",
    )

    parser.add_argument(
        "--damp-ratio",
        type=float,
        default=damp_ratio,
        metavar="N",
        help=f"change ratio of the damping factor",
    )

    return parser


def add_train_arguments(
    parser: argparse.ArgumentParser,
    epochs: int,
    test_block_size: int,
    **kwargs,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--epochs",
        type=int,
        default=epochs,
        metavar="N",
        help=f"number of epochs to train",
    )

    parser.add_argument(
        "--test-block-size",
        type=int,
        default=test_block_size,
        metavar="N",
        help=f"input block size for testing",
    )

    return parser


def add_profile_arguments(parser: argparse.ArgumentParser, profile_steps: int, **kwargs) -> argparse.ArgumentParser:
    parser.add_argument(
        "--profile",
        action="store_true",
        help=f"Enable the profiling mode",
    )

    parser.add_argument(
        "--profile-steps",
        type=int,
        default=profile_steps,
        metavar="N",
        help=f"number of steps before exiting profiling",
    )

    return parser
