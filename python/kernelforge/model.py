"""Hardware-independent correctness and memory-traffic models."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


def rmsnorm_reference(
    x: list[list[float]],
    residual: list[list[float]],
    weight: list[float],
    epsilon: float = 1e-5,
) -> list[list[float]]:
    if not x or len(x) != len(residual):
        raise ValueError("x and residual must have the same non-zero row count")
    hidden = len(weight)
    if hidden == 0:
        raise ValueError("weight cannot be empty")
    output: list[list[float]] = []
    for left, right in zip(x, residual, strict=True):
        if len(left) != hidden or len(right) != hidden:
            raise ValueError("all rows must match the weight length")
        combined = [a + b for a, b in zip(left, right, strict=True)]
        inverse_rms = 1.0 / math.sqrt(
            sum(value * value for value in combined) / hidden + epsilon
        )
        output.append(
            [value * inverse_rms * scale for value, scale in zip(combined, weight, strict=True)]
        )
    return output


@dataclass(frozen=True)
class TrafficEstimate:
    rows: int
    hidden: int
    elements: int
    unfused_bytes: int
    fused_bytes: int
    bytes_saved: int
    traffic_reduction: float
    bandwidth_bound_speedup_ceiling: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def traffic_estimate(rows: int, hidden: int, *, element_bytes: int = 4) -> TrafficEstimate:
    if rows <= 0 or hidden <= 0 or element_bytes <= 0:
        raise ValueError("dimensions and element size must be positive")
    elements = rows * hidden
    # Unfused: add (3 tensors), reduce (1), normalize (3), plus row scale write/read.
    unfused = 7 * elements * element_bytes + 2 * rows * element_bytes
    # Fused: x + residual + weight reads and output write.
    fused = 4 * elements * element_bytes
    saved = unfused - fused
    return TrafficEstimate(
        rows=rows,
        hidden=hidden,
        elements=elements,
        unfused_bytes=unfused,
        fused_bytes=fused,
        bytes_saved=saved,
        traffic_reduction=saved / unfused,
        bandwidth_bound_speedup_ceiling=unfused / fused,
    )

