import argparse
import gc
import statistics
import time
from dataclasses import dataclass
from typing import Optional

import torch
from lema import JacobianModel
from torch import nn


def create_model(hidden_size: int = 100) -> nn.Module:
    """Create a dense neural network"""
    return nn.Sequential(
        nn.Linear(1, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, 1),
    )


class SlowJacrevJacobianModel(JacobianModel):
    @torch.no_grad()
    def jacrev_version1(
        self, x: torch.Tensor, y: torch.Tensor, has_residual: bool = False, slice_size: Optional[int] = None
    ) -> torch.Tensor:
        def compute_residual(p):
            res = self._stateless_residual(p, x, y)
            return (res, res) if has_residual else res

        return torch.func.jacrev(compute_residual, has_aux=has_residual, chunk_size=slice_size)(self._flat)

    @torch.no_grad()
    def jacrev_version2(
        self, x: torch.Tensor, y: torch.Tensor, has_residual: bool = False, slice_size: Optional[int] = None
    ):
        def compute_single_residual(p, xi, yi):
            res = self._stateless_residual(p, xi.unsqueeze(0), yi.unsqueeze(0)).reshape(())
            return (res, res) if has_residual else res

        return torch.vmap(
            torch.func.jacrev(compute_single_residual, has_aux=has_residual),
            in_dims=(None, 0, 0),
            chunk_size=slice_size,
        )(self._flat, x, y)

    @torch.no_grad()
    def jacrev_version3(
        self, x: torch.Tensor, y: torch.Tensor, has_residual: bool = False, slice_size: Optional[int] = None
    ):
        def compute_single_residual(p, xi, yi):
            res = self._stateless_residual(p, xi.unsqueeze(0), yi.unsqueeze(0)).reshape(())
            return (res, res) if has_residual else res

        return torch.vmap(
            torch.func.grad(compute_single_residual, has_aux=has_residual),
            in_dims=(None, 0, 0),
            chunk_size=slice_size,
        )(self._flat, x, y)


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
    """Wait for asynchronous CUDA work before reading time or memory."""
    torch.cuda.synchronize(device)


def measure_peak_memory(fn, device: torch.device) -> int:
    """Measure peak additional memory used by one invocation of fn."""
    gc.collect()
    torch.cuda.empty_cache()
    synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    output = fn()
    synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) - baseline
    del output
    return max(0, peak)


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
    parser = argparse.ArgumentParser(description="Benchmark reverse-mode Jacobian implementations.")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=100,
        help="Number of neurons in the model's hidden layer; default: 100.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--slice-size",
        type=int,
        default=None,
        help="chunk_size passed to jacrev/vmap; the default processes the whole batch.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="CUDA device (for example cuda or cuda:1); default: cuda.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.hidden_size <= 0:
        raise ValueError("--hidden-size must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.slice_size is not None and args.slice_size <= 0:
        raise ValueError("--slice-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this benchmark")

    torch.manual_seed(0)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must be a CUDA device, for example cuda or cuda:1")

    datax = torch.linspace(-1, 1, args.batch_size, dtype=torch.float32, device=device).unsqueeze(1)
    datay = torch.sinc(10.0 * datax)

    model = create_model(hidden_size=args.hidden_size).to(device=device)
    model = SlowJacrevJacobianModel(model, residual_fn=lambda a, b: a - b)

    methods = [
        ("jacrev_version1", model.jacrev_version1),
        ("jacrev_version2", model.jacrev_version2),
        ("jacrev_version3", model.jacrev_version3),
        ("JacobianModel.jacrev", model.jacrev),
    ]
    calls = [
        (
            name,
            lambda method=method: method(
                datax,
                datay,
                slice_size=args.slice_size,
            ),
        )
        for name, method in methods
    ]

    # Compare every implementation against jacrev_version1 before benchmarking.
    reference = calls[0][1]()
    synchronize(device)
    validation_results = [ValidationResult(calls[0][0], 0.0, 0.0, True)]
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

    results = [benchmark(name, fn, device, warmup=args.warmup, repeats=args.repeats) for name, fn in calls]

    print(f"Device: {device}")
    print(
        f"Batch size: {args.batch_size}, hidden size: {args.hidden_size}, "
        f"Jacobian shape: {jacobian_shape}, slice size: {args.slice_size}"
    )
    print("\nCorrectness check (reference: jacrev_version1)")
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
    print()
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
