#!/usr/bin/env python
"""Fail if the figures quoted in the README disagree with the committed artifacts.

The README quotes specific measured numbers -- a 1.876x geometric-mean speedup, a
2.059x peak, a 42.86% traffic reduction, and a five-row Tesla T4 latency table.
Those are credibility-critical: a stale figure is worse than no figure at all.

Why this guard is needed *here* specifically. The T4 numbers were measured on a
dedicated GPU host on 2026-07-16, and CI cannot reach a GPU: the `gpu-comparison`
job in baseline-comparison.yml is deliberately `if: false`. So unlike
artifacts/triton_ptx_vectorization.json -- which CI regenerates and diffs -- there
is no job that can ever re-derive artifacts/benchmark_matrix.json. Nothing would
notice if someone edited a README number, or regenerated the artifact on a new
device and forgot to update the prose. Both files would stay self-consistent and
the prose would be wrong, indefinitely.

This script does NOT measure anything. It has no GPU and makes no timing claim. It
only asserts that the prose and the committed evidence agree, which is the one
thing checkable without hardware.

Note the direction of the comparison: every figure is *parsed out of README.md*
and checked against a value *derived from the artifact*. It deliberately does not
compare an artifact value against a literal copied from that same artifact -- that
would pass vacuously. Editing either side alone fails this check; that is the
point, not an inconvenience.

Usage:
    python scripts/check_documented_metrics.py [repo_root]
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

# Tolerances match the precision the README actually prints, not tighter. The
# speedups are shown to 3 decimals, so 5e-4 fails on a real drift but not on a
# digit the prose never displayed. Latencies are printed to 4 decimals (ms).
SPEEDUP_TOLERANCE = 5e-4
LATENCY_TOLERANCE = 5e-5
PERCENT_TOLERANCE = 5e-3


def _geometric_mean(values: list[float]) -> float:
    """Geometric mean, recomputed from the per-shape speedups.

    Deliberately re-derived rather than read from the artifact's own
    `geometric_mean_speedup` field so that an artifact whose summary disagrees
    with its own rows is also caught, not just a stale README.
    """

    return math.exp(sum(math.log(value) for value in values) / len(values))


def _parse_number(text: str) -> float:
    """Parse a README figure, tolerating the thousands separators the prose uses."""

    return float(text.replace(",", "").replace("*", "").strip())


def _fail(failures: list[str], label: str, documented: float, actual: float, tol: float) -> None:
    if not math.isclose(documented, actual, abs_tol=tol):
        failures.append(f"{label}: README says {documented}, artifact gives {actual}")


def check_latency_table(readme: str, results: list[dict], failures: list[str]) -> int:
    """Check the five-row 'Measured Tesla T4 result' table against the artifact.

    Each row is matched on its shape, so a reordered table still checks the right
    numbers and a row quoting a shape absent from the artifact is an error rather
    than a silently skipped line.
    """

    by_shape = {(row["rows"], row["hidden"]): row for row in results}
    # | 4,096 x 768 | `float4` | 0.3159 ms | 0.1534 ms | **2.059x** |
    pattern = re.compile(
        r"^\|\s*([\d,]+)\s*x\s*([\d,]+)\s*\|\s*`?(\w+)`?\s*\|\s*"
        r"([\d.]+)\s*ms\s*\|\s*([\d.]+)\s*ms\s*\|\s*\*\*([\d.]+)x\*\*\s*\|",
        re.MULTILINE,
    )
    matches = pattern.findall(readme)
    if not matches:
        failures.append("latency table: no rows matched -- has the README table format changed?")
        return 0
    for rows_text, hidden_text, variant, unfused, fused, speedup in matches:
        shape = (int(_parse_number(rows_text)), int(_parse_number(hidden_text)))
        row = by_shape.get(shape)
        label = f"latency table {shape[0]}x{shape[1]}"
        if row is None:
            failures.append(f"{label}: documented but absent from the artifact")
            continue
        if row["fused_variant"] != variant:
            failures.append(
                f"{label}: README says dispatch {variant!r}, artifact says "
                f"{row['fused_variant']!r}"
            )
        # The README rounds to 4 decimals; compare at that displayed precision.
        _fail(
            failures,
            f"{label} baseline ms",
            float(unfused),
            round(row["unfused_ms"], 4),
            LATENCY_TOLERANCE,
        )
        _fail(
            failures,
            f"{label} fused ms",
            float(fused),
            round(row["fused_ms"], 4),
            LATENCY_TOLERANCE,
        )
        _fail(failures, f"{label} speedup", float(speedup), row["speedup"], SPEEDUP_TOLERANCE)
    if len(matches) != len(results):
        failures.append(
            f"latency table: README documents {len(matches)} rows but the artifact "
            f"holds {len(results)}"
        )
    return len(matches)


def _find_one(readme: str, pattern: str, label: str, failures: list[str]) -> float | None:
    """Extract exactly one figure, failing if the prose no longer matches."""

    found = re.findall(pattern, readme)
    if not found:
        failures.append(f"{label}: not found in README -- was the wording changed?")
        return None
    values = {_parse_number(value) for value in found}
    if len(values) > 1:
        failures.append(f"{label}: README quotes conflicting values {sorted(values)}")
        return None
    return values.pop()


def main(argv: list[str]) -> int:
    root = Path(argv[0] if argv else ".")
    readme_path = root / "README.md"
    matrix_path = root / "artifacts" / "benchmark_matrix.json"
    analytical_path = root / "artifacts" / "analytical.json"
    for path in (readme_path, matrix_path, analytical_path):
        if not path.exists():
            print(f"missing input: {path}", file=sys.stderr)
            return 2

    readme = readme_path.read_text(encoding="utf-8")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    analytical = json.loads(analytical_path.read_text(encoding="utf-8"))
    results = matrix["results"]
    failures: list[str] = []
    checked = 0

    checked += check_latency_table(readme, results, failures)

    speedups = [row["speedup"] for row in results]
    recomputed_geomean = _geometric_mean(speedups)
    # The artifact's own summary must agree with its rows before the README is
    # compared against either.
    if not math.isclose(recomputed_geomean, matrix["geometric_mean_speedup"], rel_tol=1e-9):
        failures.append(
            f"artifact is internally inconsistent: geometric_mean_speedup is "
            f"{matrix['geometric_mean_speedup']} but its rows give {recomputed_geomean}"
        )
    for value in (row["speedup"] for row in results):
        if value <= 0:
            failures.append("artifact holds a non-positive speedup")

    # "Achieved a **1.876x geometric-mean speedup**" / "geometric-mean speedup is **1.876x**"
    documented_geomean = _find_one(
        readme, r"\*\*([\d.]+)x geometric-mean speedup\*\*", "geometric-mean speedup", failures
    )
    if documented_geomean is not None:
        _fail(failures, "geometric mean", documented_geomean, recomputed_geomean, SPEEDUP_TOLERANCE)
        checked += 1
    restated_geomean = _find_one(
        readme, r"geometric-mean speedup is \*\*([\d.]+)x\*\*", "restated geometric mean", failures
    )
    if restated_geomean is not None:
        _fail(
            failures,
            "restated geometric mean",
            restated_geomean,
            recomputed_geomean,
            SPEEDUP_TOLERANCE,
        )
        checked += 1

    # "with a **2.059x** peak speedup"
    documented_peak = _find_one(readme, r"\*\*([\d.]+)x\*\* peak speedup", "peak speedup", failures)
    if documented_peak is not None:
        _fail(failures, "peak speedup", documented_peak, max(speedups), SPEEDUP_TOLERANCE)
        checked += 1
    if not math.isclose(max(speedups), matrix["maximum_speedup"], rel_tol=1e-9):
        failures.append("artifact's maximum_speedup disagrees with its rows")
    if not math.isclose(min(speedups), matrix["minimum_speedup"], rel_tol=1e-9):
        failures.append("artifact's minimum_speedup disagrees with its rows")

    # "and **1e-6** maximum error" / "at or below **1e-6**"
    max_error = max(
        max(row["fused_max_absolute_error"], row["unfused_max_absolute_error"]) for row in results
    )
    if not math.isclose(max_error, matrix["maximum_absolute_error"], rel_tol=1e-9):
        failures.append("artifact's maximum_absolute_error disagrees with its rows")
    for pattern, label in (
        (r"\*\*(1e-0?6)\*\* maximum error", "documented maximum error"),
        (r"at or below \*\*(1e-0?6)\*\*", "restated maximum error"),
    ):
        documented_error = _find_one(readme, pattern, label, failures)
        if documented_error is not None:
            if max_error > documented_error * (1 + 1e-9):
                failures.append(
                    f"{label}: README claims error at or below {documented_error} but the "
                    f"artifact holds {max_error}"
                )
            checked += 1

    # "Reduced modeled global-memory traffic by **42.86%**" and the restatement in
    # the analytical section. Checked against analytical.json, which is the file
    # that actually derives it, and cross-checked against the per-row model.
    #
    # Every derived figure below is recomputed from unfused_bytes and fused_bytes
    # -- the two raw counts -- rather than read from the artifact's own summary
    # fields, for the same reason the geometric mean is re-derived above: reading
    # traffic_reduction and bandwidth_bound_speedup_ceiling straight out of the
    # artifact would let a corrupted summary certify itself, and the README would
    # be checked against a number nothing had validated.
    recomputed_saved = analytical["unfused_bytes"] - analytical["fused_bytes"]
    recomputed_traffic = recomputed_saved / analytical["unfused_bytes"]
    recomputed_ceiling = analytical["unfused_bytes"] / analytical["fused_bytes"]
    for key, recomputed in (
        ("bytes_saved", recomputed_saved),
        ("traffic_reduction", recomputed_traffic),
        ("bandwidth_bound_speedup_ceiling", recomputed_ceiling),
    ):
        if not math.isclose(recomputed, analytical[key], rel_tol=1e-9):
            failures.append(
                f"analytical artifact is internally inconsistent: {key} is "
                f"{analytical[key]} but its byte counts give {recomputed}"
            )

    traffic_pct = recomputed_traffic * 100
    for pattern, label in (
        (r"global-memory traffic by \*\*([\d.]+)%\*\*", "traffic reduction"),
        (r"a \*\*([\d.]+)% traffic reduction\*\*", "restated traffic reduction"),
        (r"cutting modeled traffic\s+([\d.]+)%", "traffic reduction in prose"),
    ):
        documented = _find_one(readme, pattern, label, failures)
        if documented is not None:
            _fail(failures, label, documented, round(traffic_pct, 2), PERCENT_TOLERANCE)
            checked += 1

    # The byte counts in the analytical table. unfused/fused are the raw counts;
    # the saved row is checked against the difference, not against the artifact's
    # own bytes_saved field.
    for pattern, actual, label in (
        (
            r"\|\s*Three-kernel baseline\s*\|\s*([\d,]+) bytes",
            float(analytical["unfused_bytes"]),
            "unfused bytes",
        ),
        (
            r"\|\s*Shared-memory fused\s*\|\s*([\d,]+) bytes",
            float(analytical["fused_bytes"]),
            "fused bytes",
        ),
        (
            r"\|\s*Saved\s*\|\s*\*\*([\d,]+) bytes\*\*",
            float(recomputed_saved),
            "bytes saved",
        ),
    ):
        documented = _find_one(readme, pattern, label, failures)
        if documented is not None:
            _fail(failures, label, documented, actual, 0)
            checked += 1
    if analytical["unfused_bytes"] - analytical["fused_bytes"] != analytical["bytes_saved"]:
        failures.append("analytical artifact: unfused - fused does not equal bytes_saved")

    # "a **1.75x bandwidth-bound speedup ceiling**"
    documented_ceiling = _find_one(
        readme, r"\*\*([\d.]+)x bandwidth-bound speedup\s+ceiling\*\*", "speedup ceiling", failures
    )
    if documented_ceiling is not None:
        _fail(
            failures,
            "speedup ceiling",
            documented_ceiling,
            round(recomputed_ceiling, 2),
            PERCENT_TOLERANCE,
        )
        checked += 1

    # "the vector path reduced fused latency by 21.4%" -- the adjacent 4,095/4,096
    # pair, recomputed from the two fused timings rather than restated.
    by_shape = {(row["rows"], row["hidden"]): row for row in results}
    scalar_row, vector_row = by_shape.get((4096, 4095)), by_shape.get((4096, 4096))
    documented_delta = _find_one(
        readme, r"reduced fused latency by ([\d.]+)%", "vector-path latency delta", failures
    )
    if documented_delta is None:
        pass
    elif scalar_row is None or vector_row is None:
        failures.append(
            "vector-path latency delta: the 4,095/4,096 pair is missing from the artifact"
        )
    else:
        actual_delta = (
            (scalar_row["fused_ms"] - vector_row["fused_ms"]) / scalar_row["fused_ms"] * 100
        )
        _fail(failures, "vector-path latency delta", documented_delta, round(actual_delta, 1), 5e-2)
        checked += 1

    # Device and toolkit provenance: the prose names the GPU and CUDA version, and
    # a table assembled from two devices would invalidate the geometric mean.
    devices = {row["device"] for row in results}
    if len(devices) != 1:
        failures.append(f"artifact mixes devices {sorted(devices)}; the table must be one GPU")
    elif devices != {matrix["device"]}:
        failures.append("artifact's top-level device disagrees with its rows")
    else:
        device = devices.pop()
        if device not in readme:
            failures.append(f"README does not name the measured device {device!r}")
        checked += 1

    # "with CUDA 13.2", stored as 13020 (1000*major + 10*minor).
    documented_cuda = _find_one(readme, r"with CUDA ([\d.]+)\b", "CUDA version", failures)
    if documented_cuda is not None:
        runtimes = {row["cuda_runtime_version"] for row in results}
        if len(runtimes) != 1:
            failures.append(f"artifact mixes CUDA runtimes {sorted(runtimes)}")
        else:
            runtime = runtimes.pop()
            major, minor = divmod(runtime, 1000)
            actual_cuda = f"{major}.{minor // 10}"
            if f"{documented_cuda:g}" != actual_cuda:
                failures.append(
                    f"CUDA version: README says {documented_cuda:g}, artifact says {actual_cuda}"
                )
            checked += 1

    # "25 warmup launches followed by 200 CUDA-event-timed iterations"
    for pattern, key, label in (
        (r"(\d+) warmup launches", "warmup", "warmup launches"),
        (r"followed by ([\d,]+) CUDA-event-timed\s+iterations", "iterations", "timed iterations"),
    ):
        documented = _find_one(readme, pattern, label, failures)
        if documented is not None:
            _fail(failures, label, documented, float(matrix[key]), 0)
            checked += 1

    # The Triton-vs-CUDA access-width table. Yes/no per width, from the PTX report.
    checked += check_vectorization_table(root, readme, failures)

    if failures:
        print("Documented metrics do not match the committed artifacts:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            "\nIf a change legitimately moved one of these, update the README and the "
            "artifact together. This script measures nothing; it only compares them.",
            file=sys.stderr,
        )
        return 1

    print(f"All {checked} documented figures match the committed artifacts.")
    return 0


def check_vectorization_table(root: Path, readme: str, failures: list[str]) -> int:
    """Check the README's per-width CUDA/Triton agreement table against the PTX report.

    Skipped rather than failed when the PTX report is absent: it is regenerated and
    diffed by its own CI job, and this script should still be runnable in a checkout
    that has not produced it.
    """

    report_path = root / "artifacts" / "triton_ptx_vectorization.json"
    if not report_path.exists():
        return 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    by_width = {entry["hidden"]: entry for entry in report["widths"]}
    # | 4,092 | yes | **no**, 64 scalar accesses | CUDA only |
    pattern = re.compile(
        r"^\|\s*([\d,]+)\s*\|\s*\*{0,2}(yes|no)\*{0,2}\s*\|\s*\*{0,2}(yes|no)\*{0,2}([^|]*)\|",
        re.MULTILINE,
    )
    checked = 0
    for width_text, cuda_text, triton_text, trailer in pattern.findall(readme):
        width = int(_parse_number(width_text))
        entry = by_width.get(width)
        if entry is None:
            failures.append(f"vectorization table: width {width} is not in the PTX report")
            continue
        documented_cuda = cuda_text == "yes"
        documented_triton = triton_text == "yes"
        if documented_cuda != entry["cuda_takes_float4_path"]:
            failures.append(
                f"vectorization table {width}: README says CUDA float4="
                f"{documented_cuda}, report says {entry['cuda_takes_float4_path']}"
            )
        if documented_triton != entry["triton_vectorized"]:
            failures.append(
                f"vectorization table {width}: README says Triton vectorized="
                f"{documented_triton}, report says {entry['triton_vectorized']}"
            )
        # "**no**, 64 scalar accesses" -- the count is quoted, so check it.
        quoted = re.search(r"(\d+) scalar accesses", trailer)
        if quoted and int(quoted.group(1)) != entry["scalar_global_accesses"]:
            failures.append(
                f"vectorization table {width}: README quotes "
                f"{quoted.group(1)} scalar accesses, report says "
                f"{entry['scalar_global_accesses']}"
            )
        checked += 1
    if checked != len(by_width):
        failures.append(
            f"vectorization table: README documents {checked} widths, the PTX report "
            f"holds {len(by_width)}"
        )
    # "its only divisibility tier is 16" and the CUDA divisor of 4.
    tier = _find_one(
        readme, r"only divisibility tier is (\d+)", "Triton divisibility tier", failures
    )
    if tier is not None:
        _fail(
            failures,
            "Triton divisibility tier",
            tier,
            float(report["triton_divisibility_tier"]),
            0,
        )
        checked += 1
    return checked


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
