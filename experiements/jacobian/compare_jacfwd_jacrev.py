import argparse
import gc
import statistics
import time
from dataclasses import dataclass

import torch
from lema import JacobianModel
from torch import nn

DEFAULT_SIZE = 500


def create_model(hidden_size: int = DEFAULT_SIZE) -> nn.Module:
    """Create a dense neural network"""
    return nn.Sequential(
        nn.Linear(1, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, 1),
    )


@dataclass
class BenchmarkResult:
    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    peak_memory_mib: float


@dataclass
class ValidationResult:
    name: str
    max_abs_error: float
    max_rel_error: float
    matches: bool


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def measure_peak_memory(fn, device: torch.device) -> int:
    """Measure peak additional CUDA memory used by one invocation."""
    gc.collect()
    torch.cuda.empty_cache()
    synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    output = fn()
    synchronize(device)
    peak_memory = torch.cuda.max_memory_allocated(device) - baseline
    del output
    return max(0, peak_memory)


def benchmark(name, fn, device: torch.device, warmup: int, repeats: int) -> BenchmarkResult:
    for _ in range(warmup):
        output = fn()
        synchronize(device)
        del output

    timings = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        output = fn()
        synchronize(device)
        timings.append((time.perf_counter() - start) * 1_000.0)
        del output

    peak_memory = measure_peak_memory(fn, device)
    return BenchmarkResult(
        name=name,
        mean_ms=statistics.fmean(timings),
        std_ms=statistics.stdev(timings) if len(timings) > 1 else 0.0,
        min_ms=min(timings),
        peak_memory_mib=peak_memory / (1024**2),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare JacobianModel.jacfwd and JacobianModel.jacrev.")
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=DEFAULT_SIZE,
        help=f"Number of hidden neurons; default: {DEFAULT_SIZE}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_SIZE,
        help=f"Number of samples; default: {DEFAULT_SIZE}.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--device",
        default="cuda",
        help="CUDA device (for example cuda or cuda:1); default: cuda.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hidden_size <= 0:
        raise ValueError("--hidden-size must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this benchmark")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must be a CUDA device, for example cuda or cuda:1")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device_index)
    torch.manual_seed(0)

    datax = torch.linspace(-1, 1, args.batch_size, dtype=torch.float32, device=device).unsqueeze(1)
    datay = torch.sinc(10.0 * datax)
    model = create_model(args.hidden_size).to(device=device)
    model = JacobianModel(model, residual_fn=lambda prediction, target: prediction - target)

    calls = [
        ("JacobianModel.jacfwd", lambda: model.jacfwd(datax, datay)),
        ("JacobianModel.jacrev", lambda: model.jacrev(datax, datay)),
    ]

    reference_name, reference_fn = calls[0]
    reference = reference_fn()
    synchronize(device)
    validation_results = [ValidationResult(reference_name, 0.0, 0.0, True)]
    for name, fn in calls[1:]:
        candidate = fn()
        synchronize(device)
        abs_error = (candidate - reference).abs()
        rel_error = abs_error / reference.abs().clamp_min(torch.finfo(reference.dtype).eps)
        matches = torch.allclose(candidate, reference, rtol=1e-5, atol=1e-6)
        validation_results.append(
            ValidationResult(
                name=name,
                max_abs_error=abs_error.max().item(),
                max_rel_error=rel_error.max().item(),
                matches=matches,
            )
        )
        torch.testing.assert_close(candidate, reference, rtol=1e-5, atol=1e-6)
        del candidate

    jacobian_shape = tuple(reference.shape)
    del reference
    results = [benchmark(name, fn, device, args.warmup, args.repeats) for name, fn in calls]

    print(f"Device: {device}")
    print(
        f"Hidden size: {args.hidden_size}, batch size: {args.batch_size}, "
        f"parameters: {sum(parameter.numel() for parameter in model.parameters())}, "
        f"Jacobian shape: {jacobian_shape}"
    )
    print("\nCorrectness check (reference: JacobianModel.jacfwd)")
    print(f"{'Method':<24} {'Max abs error':>16} {'Max rel error':>16} {'Matches':>10}")
    print("-" * 70)
    for result in validation_results:
        print(
            f"{result.name:<24} "
            f"{result.max_abs_error:>16.6e} "
            f"{result.max_rel_error:>16.6e} "
            f"{str(result.matches):>10}"
        )

    print("\nPerformance")
    print(f"{'Method':<24} {'Mean (ms)':>12} {'Std (ms)':>12} {'Min (ms)':>12} {'Peak memory (MiB)':>20}")
    print("-" * 84)
    for result in results:
        print(
            f"{result.name:<24} "
            f"{result.mean_ms:>12.3f} "
            f"{result.std_ms:>12.3f} "
            f"{result.min_ms:>12.3f} "
            f"{result.peak_memory_mib:>20.3f}"
        )
    print("\nPeak memory is the maximum additional allocated memory above the pre-call baseline.")


if __name__ == "__main__":
    main()
