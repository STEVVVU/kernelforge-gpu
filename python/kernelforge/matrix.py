"""Run and summarize a reproducible matrix of CUDA benchmark shapes."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SHAPES = "4096x768,4096x1024,4096x4095,4096x4096,1024x8192"


def parse_shapes(value: str) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for raw_shape in value.split(","):
        try:
            rows_text, hidden_text = raw_shape.lower().split("x", maxsplit=1)
            rows, hidden = int(rows_text), int(hidden_text)
        except ValueError as error:
            raise ValueError(f"invalid shape {raw_shape!r}; expected ROWSxHIDDEN") from error
        if rows <= 0 or hidden <= 0:
            raise ValueError("shape dimensions must be positive")
        shapes.append((rows, hidden))
    if not shapes:
        raise ValueError("at least one shape is required")
    return shapes


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("cannot summarize an empty benchmark")
    devices = {result["device"] for result in results}
    if len(devices) != 1:
        raise ValueError("all benchmark rows must come from the same device")
    speedups = [float(result["speedup"]) for result in results]
    geometric_mean = math.exp(sum(math.log(speedup) for speedup in speedups) / len(speedups))
    return {
        "device": next(iter(devices)),
        "shapes": len(results),
        "geometric_mean_speedup": geometric_mean,
        "minimum_speedup": min(speedups),
        "maximum_speedup": max(speedups),
        "maximum_absolute_error": max(
            float(result["fused_max_absolute_error"]) for result in results
        ),
        "results": results,
    }


def run_matrix(
    binary: str | Path,
    shapes: list[tuple[int, int]],
    *,
    iterations: int,
    warmup: int,
) -> dict[str, Any]:
    if iterations <= 0 or warmup < 0:
        raise ValueError("iterations must be positive and warmup cannot be negative")
    results: list[dict[str, Any]] = []
    for rows, hidden in shapes:
        process = subprocess.run(
            [
                str(binary),
                "--rows",
                str(rows),
                "--hidden",
                str(hidden),
                "--iterations",
                str(iterations),
                "--warmup",
                str(warmup),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(process.stdout)
        if result["rows"] != rows or result["hidden"] != hidden:
            raise RuntimeError("benchmark output shape differs from requested shape")
        results.append(result)
    summary = summarize_results(results)
    summary["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["iterations"] = iterations
    summary["warmup"] = warmup
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="build/kernelforge_bench")
    parser.add_argument("--shapes", default=DEFAULT_SHAPES)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--output", default="artifacts/benchmark_matrix.json")
    args = parser.parse_args(argv)
    report = run_matrix(
        args.binary,
        parse_shapes(args.shapes),
        iterations=args.iterations,
        warmup=args.warmup,
    )
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0
