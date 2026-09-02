"""Compare the hand-written CUDA kernel against torch / torch.compile / Triton.

Answers the first question a GPU reviewer asks: *why write CUDA at all when
``torch`` and Triton already do this?* The harness times every implementation
of the same contract, over the same shapes, with the same protocol as
``src/benchmark.cu`` (CUDA events, N warmup launches, N timed launches on one
stream), and scores each against a float64 oracle.

Two integrity rules are enforced in code rather than left to discipline:

1. **Correctness gates timing.** A baseline whose output deviates from the
   float64 oracle beyond tolerance is reported as an error, not as a fast row.
2. **Numbers from different devices are never mixed.** When previously-measured
   CUDA results are merged in via ``--cuda-report``, the device name must match
   the device the baselines just ran on, or the merge is refused. Publishing a
   Triton time from one GPU next to a CUDA time from another would be a fake
   comparison.

Usage (GPU host):

    python -m kernelforge.compare \\
      --binary build/kernelforge_bench \\
      --output artifacts/baseline_comparison.json

Usage (CPU host, correctness only -- no timings are emitted):

    python -m kernelforge.compare --correctness-only
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .baselines import (
    CUDA_ROWS,
    DEVICE_ACCUMULATION_DTYPE,
    EPSILON_PLACEMENT,
    ORACLE_ACCUMULATION_DTYPE,
    WEIGHT_PLACEMENT,
    BaselineUnavailable,
    iter_baselines,
    make_inputs,
    reference_float64,
)
from .matrix import DEFAULT_SHAPES, parse_shapes, run_matrix

DEFAULT_EPSILON = 1e-5
DEFAULT_ITERATIONS = 200
DEFAULT_WARMUP = 25

# Mirrors the mixed tolerance in src/benchmark.cu: a row fails only when it
# breaches BOTH the absolute and the relative bound.
ABSOLUTE_TOLERANCE = 2e-5
RELATIVE_TOLERANCE = 2e-4

# The irreducible traffic for this operation in fp32: read x, read residual,
# write output (the (hidden,) weight vector is negligible and cached). Used as
# the one bandwidth metric that is comparable across every implementation --
# see `minimum_traffic_gbps` below.
MINIMUM_BYTES_PER_ELEMENT = 12
# The traffic the CUDA fused path is modeled at in src/benchmark.cu, kept here
# so the two reports use one number.
FUSED_MODELED_BYTES_PER_ELEMENT = 16


def minimum_traffic_bytes(rows: int, hidden: int) -> int:
    """Bytes that *any* correct fp32 implementation must move for this shape."""
    return MINIMUM_BYTES_PER_ELEMENT * rows * hidden


def minimum_traffic_gbps(rows: int, hidden: int, latency_ms: float) -> float:
    """Bandwidth implied by the irreducible traffic at ``latency_ms``.

    This is a **lower bound** on the bandwidth an implementation actually
    achieved: an implementation that materializes intermediates moves more than
    the minimum, so its true achieved bandwidth is higher than this figure
    while its latency is worse. Reporting the lower bound keeps the column
    comparable without needing a per-implementation traffic model, which for
    ``torch.compile`` would be a guess about inductor's fusion decisions.
    """
    if latency_ms <= 0:
        raise ValueError("latency must be positive")
    return minimum_traffic_bytes(rows, hidden) / latency_ms / 1e6


def error_metrics(actual, expected) -> tuple[float, float]:
    """Return ``(max_absolute, max_relative)`` against a float64 oracle."""
    import torch

    difference = (actual.to(torch.float64) - expected).abs()
    denominator = expected.abs().clamp_min(1e-6)
    return float(difference.max().item()), float((difference / denominator).max().item())


def within_tolerance(maximum_absolute: float, maximum_relative: float) -> bool:
    """Apply the same mixed tolerance rule as ``src/benchmark.cu``."""
    return not (
        maximum_absolute > ABSOLUTE_TOLERANCE and maximum_relative > RELATIVE_TOLERANCE
    )


def time_callable(
    function,
    arguments: tuple,
    *,
    iterations: int,
    warmup: int,
    on_cuda: bool,
) -> float:
    """Average per-launch milliseconds, timed the way ``src/benchmark.cu`` times.

    On CUDA this uses ``torch.cuda.Event`` around the whole timed batch, which
    is the same instrument and the same batching as the C++ harness, so the
    numbers are directly comparable. Warmup absorbs autotuning and, for
    ``torch.compile``, the compile itself.
    """
    import torch

    for _ in range(warmup):
        function(*arguments)
    if not on_cuda:
        raise RuntimeError(
            "refusing to report a latency measured without CUDA events; "
            "CPU timings are not comparable to the T4 evidence"
        )
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function(*arguments)
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / iterations


def check_correctness(
    shapes: list[tuple[int, int]],
    *,
    device: str,
    epsilon: float,
    names: list[str] | None = None,
) -> dict[str, Any]:
    """Verify every available baseline against the float64 oracle.

    Runs on CPU, so CI can prove the baselines implement the *same formula* as
    the CUDA kernel without a GPU. This is the check that makes the latency
    comparison meaningful; a fast implementation of a different formula is not
    a baseline.
    """
    rows_out: list[dict[str, Any]] = []
    for baseline in iter_baselines(names):
        try:
            function = baseline.resolve()
        except BaselineUnavailable as error:
            rows_out.append(
                {
                    "baseline": baseline.name,
                    "label": baseline.label,
                    "status": "unavailable",
                    "detail": str(error),
                }
            )
            continue
        if baseline.requires_cuda and device != "cuda":
            rows_out.append(
                {
                    "baseline": baseline.name,
                    "label": baseline.label,
                    "status": "gpu_required",
                    "detail": "implementation only runs on CUDA",
                }
            )
            continue
        shape_rows: list[dict[str, Any]] = []
        status = "pass"
        detail = ""
        for rows, hidden in shapes:
            x, residual, weight = make_inputs(rows, hidden, device=device)
            expected = reference_float64(x, residual, weight, epsilon)
            try:
                actual = function(x, residual, weight, epsilon)
            except Exception as error:  # noqa: BLE001 - reported, not swallowed
                status = "error"
                detail = f"{type(error).__name__}: {error}"
                break
            maximum_absolute, maximum_relative = error_metrics(actual, expected)
            passed = within_tolerance(maximum_absolute, maximum_relative)
            if not passed:
                status = "fail"
                detail = f"tolerance exceeded at {rows}x{hidden}"
            shape_rows.append(
                {
                    "rows": rows,
                    "hidden": hidden,
                    "max_absolute_error": maximum_absolute,
                    "max_relative_error": maximum_relative,
                    "within_tolerance": passed,
                }
            )
        rows_out.append(
            {
                "baseline": baseline.name,
                "label": baseline.label,
                "status": status,
                "detail": detail,
                "shapes": shape_rows,
            }
        )
    return {
        "device": device,
        "epsilon": epsilon,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "contract": {
            "expression": "output = (x + residual) * rsqrt(mean((x+residual)^2) + eps) * weight",
            "epsilon_placement": EPSILON_PLACEMENT,
            "weight_placement": WEIGHT_PLACEMENT,
            "device_accumulation": DEVICE_ACCUMULATION_DTYPE,
            "oracle_accumulation": ORACLE_ACCUMULATION_DTYPE,
        },
        "baselines": rows_out,
    }


def measure_baselines(
    shapes: list[tuple[int, int]],
    *,
    epsilon: float,
    iterations: int,
    warmup: int,
    names: list[str] | None = None,
) -> dict[str, Any]:
    """Time every available baseline on CUDA. Requires a GPU."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "no CUDA device visible; baseline latency requires a GPU runner "
            "(use --correctness-only on a CPU host)"
        )
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    properties = torch.cuda.get_device_properties(0)
    measurements: list[dict[str, Any]] = []
    for baseline in iter_baselines(names):
        try:
            function = baseline.resolve()
        except BaselineUnavailable as error:
            measurements.append(
                {
                    "baseline": baseline.name,
                    "label": baseline.label,
                    "tier": baseline.tier,
                    "status": "unavailable",
                    "detail": str(error),
                }
            )
            continue
        for rows, hidden in shapes:
            x, residual, weight = make_inputs(rows, hidden, device="cuda")
            expected = reference_float64(x, residual, weight, epsilon)
            arguments = (x, residual, weight, epsilon)
            try:
                actual = function(*arguments)
                maximum_absolute, maximum_relative = error_metrics(actual, expected)
                if not within_tolerance(maximum_absolute, maximum_relative):
                    measurements.append(
                        {
                            "baseline": baseline.name,
                            "label": baseline.label,
                            "tier": baseline.tier,
                            "rows": rows,
                            "hidden": hidden,
                            "status": "fail",
                            "detail": "correctness tolerance exceeded; latency withheld",
                            "max_absolute_error": maximum_absolute,
                            "max_relative_error": maximum_relative,
                        }
                    )
                    continue
                latency_ms = time_callable(
                    function,
                    arguments,
                    iterations=iterations,
                    warmup=warmup,
                    on_cuda=True,
                )
            except Exception as error:  # noqa: BLE001 - reported, not swallowed
                measurements.append(
                    {
                        "baseline": baseline.name,
                        "label": baseline.label,
                        "tier": baseline.tier,
                        "rows": rows,
                        "hidden": hidden,
                        "status": "error",
                        "detail": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            measurements.append(
                {
                    "baseline": baseline.name,
                    "label": baseline.label,
                    "tier": baseline.tier,
                    "rows": rows,
                    "hidden": hidden,
                    "status": "ok",
                    "latency_ms": latency_ms,
                    "minimum_traffic_gbps": minimum_traffic_gbps(rows, hidden, latency_ms),
                    "max_absolute_error": maximum_absolute,
                    "max_relative_error": maximum_relative,
                }
            )
    return {
        "device": device_name,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "torch_version": torch.__version__,
        "sm_count": properties.multi_processor_count,
        "iterations": iterations,
        "warmup": warmup,
        "epsilon": epsilon,
        "minimum_bytes_per_element": MINIMUM_BYTES_PER_ELEMENT,
        "measurements": measurements,
    }


def _cuda_rows_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Project a ``benchmark_matrix.json`` report into comparison rows."""
    label_by_name = {name: label for name, label, _ in CUDA_ROWS}
    tier_by_name = {name: tier for name, _, tier in CUDA_ROWS}
    rows_out: list[dict[str, Any]] = []
    for result in report["results"]:
        rows, hidden = int(result["rows"]), int(result["hidden"])
        for name, key in (("cuda_unfused", "unfused_ms"), ("cuda_fused", "fused_ms")):
            latency_ms = float(result[key])
            rows_out.append(
                {
                    "baseline": name,
                    "label": label_by_name[name],
                    "tier": tier_by_name[name],
                    "rows": rows,
                    "hidden": hidden,
                    "status": "ok",
                    "latency_ms": latency_ms,
                    "minimum_traffic_gbps": minimum_traffic_gbps(rows, hidden, latency_ms),
                    "max_absolute_error": float(
                        result[
                            "fused_max_absolute_error"
                            if name == "cuda_fused"
                            else "unfused_max_absolute_error"
                        ]
                    ),
                    "fused_variant": result.get("fused_variant"),
                }
            )
    return rows_out


def merge_reports(
    cuda_report: dict[str, Any], baseline_report: dict[str, Any]
) -> dict[str, Any]:
    """Combine CUDA and baseline measurements into one comparison.

    Refuses to merge across devices: a cross-GPU table would read as a
    comparison while measuring two different machines.
    """
    cuda_device = cuda_report["device"]
    baseline_device = baseline_report["device"]
    if cuda_device != baseline_device:
        raise ValueError(
            "refusing to merge measurements from different devices: "
            f"CUDA report is {cuda_device!r}, baselines ran on {baseline_device!r}. "
            "Re-run both on the same GPU."
        )
    rows_out = _cuda_rows_from_report(cuda_report) + [
        dict(measurement) for measurement in baseline_report["measurements"]
    ]
    fused_latency = {
        (row["rows"], row["hidden"]): row["latency_ms"]
        for row in rows_out
        if row["baseline"] == "cuda_fused" and row["status"] == "ok"
    }
    for row in rows_out:
        if row["status"] != "ok" or "rows" not in row:
            continue
        reference = fused_latency.get((row["rows"], row["hidden"]))
        if reference:
            # >1 means the hand-written fused kernel is faster than this row.
            row["fused_cuda_speedup_over_this"] = row["latency_ms"] / reference
    return {
        "device": cuda_device,
        "compute_capability": baseline_report.get("compute_capability"),
        "torch_version": baseline_report.get("torch_version"),
        "iterations": baseline_report["iterations"],
        "warmup": baseline_report["warmup"],
        "minimum_bytes_per_element": MINIMUM_BYTES_PER_ELEMENT,
        "fused_modeled_bytes_per_element": FUSED_MODELED_BYTES_PER_ELEMENT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": rows_out,
        "summary": summarize_comparison(rows_out),
    }


def summarize_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Geometric-mean speedup of the fused CUDA kernel over each other row."""
    by_baseline: dict[str, list[float]] = {}
    for row in rows:
        ratio = row.get("fused_cuda_speedup_over_this")
        if row["status"] == "ok" and ratio and row["baseline"] != "cuda_fused":
            by_baseline.setdefault(row["baseline"], []).append(ratio)
    return {
        "fused_cuda_geomean_speedup_over": {
            name: math.exp(sum(math.log(value) for value in values) / len(values))
            for name, values in sorted(by_baseline.items())
        },
        "shapes_compared": len(
            {
                (row["rows"], row["hidden"])
                for row in rows
                if row["status"] == "ok" and "rows" in row
            }
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", default=DEFAULT_SHAPES)
    parser.add_argument("--baselines", help="comma-separated subset of baseline names")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument(
        "--correctness-only",
        action="store_true",
        help="verify baselines against the float64 oracle and emit no timings",
    )
    parser.add_argument(
        "--correctness-device",
        default="cpu",
        help="device for --correctness-only (cpu or cuda)",
    )
    parser.add_argument(
        "--binary",
        help="path to kernelforge_bench; runs the CUDA matrix fresh on this GPU",
    )
    parser.add_argument(
        "--cuda-report",
        help="existing benchmark_matrix.json to merge instead of running --binary",
    )
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shapes = parse_shapes(args.shapes)
    names = args.baselines.split(",") if args.baselines else None

    if args.correctness_only:
        report: dict[str, Any] = check_correctness(
            shapes,
            device=args.correctness_device,
            epsilon=args.epsilon,
            names=names,
        )
        report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        failures = [
            row["baseline"]
            for row in report["baselines"]
            if row["status"] in {"fail", "error"}
        ]
        exit_code = 1 if failures else 0
    else:
        if bool(args.binary) == bool(args.cuda_report):
            build_parser().error("pass exactly one of --binary or --cuda-report")
        baseline_report = measure_baselines(
            shapes,
            epsilon=args.epsilon,
            iterations=args.iterations,
            warmup=args.warmup,
            names=names,
        )
        if args.binary:
            cuda_report = run_matrix(
                args.binary, shapes, iterations=args.iterations, warmup=args.warmup
            )
        else:
            cuda_report = json.loads(Path(args.cuda_report).read_text(encoding="utf-8"))
        report = merge_reports(cuda_report, baseline_report)
        exit_code = 0

    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # newline="\n" pins LF cross-platform; see the note in
        # scripts/triton_codegen_check.py. CI uploads this report as an artifact,
        # so its bytes should not vary with the host that wrote it.
        destination.write_text(serialized, encoding="utf-8", newline="\n")
    print(serialized, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
