"""The comparison harness must refuse to publish a misleading table."""

from __future__ import annotations

import json

import pytest

from kernelforge.compare import (
    ABSOLUTE_TOLERANCE,
    MINIMUM_BYTES_PER_ELEMENT,
    RELATIVE_TOLERANCE,
    build_parser,
    main,
    merge_reports,
    minimum_traffic_bytes,
    minimum_traffic_gbps,
    within_tolerance,
)


def _cuda_report(device: str = "Tesla T4") -> dict:
    return {
        "device": device,
        "results": [
            {
                "rows": 4096,
                "hidden": 4096,
                "device": device,
                "unfused_ms": 1.6,
                "fused_ms": 0.8,
                "fused_variant": "float4",
                "fused_max_absolute_error": 1e-6,
                "unfused_max_absolute_error": 1e-6,
            }
        ],
    }


def _baseline_report(device: str = "Tesla T4") -> dict:
    return {
        "device": device,
        "compute_capability": "7.5",
        "torch_version": "2.13.0+cu128",
        "iterations": 200,
        "warmup": 25,
        "measurements": [
            {
                "baseline": "torch_eager",
                "label": "torch eager (`F.rms_norm`)",
                "tier": "library",
                "rows": 4096,
                "hidden": 4096,
                "status": "ok",
                "latency_ms": 1.6,
                "max_absolute_error": 1e-6,
            },
            {
                "baseline": "triton",
                "label": "OpenAI Triton",
                "tier": "dsl",
                "rows": 4096,
                "hidden": 4096,
                "status": "error",
                "detail": "RuntimeError: boom",
            },
        ],
    }


def test_merge_refuses_to_mix_devices() -> None:
    """A cross-GPU table would read as a comparison but measure two machines."""
    with pytest.raises(ValueError, match="different devices"):
        merge_reports(_cuda_report("Tesla T4"), _baseline_report("A100-SXM4-40GB"))


def test_merge_computes_speedup_against_the_fused_cuda_kernel() -> None:
    merged = merge_reports(_cuda_report(), _baseline_report())
    by_name = {row["baseline"]: row for row in merged["rows"]}
    assert by_name["cuda_fused"]["fused_cuda_speedup_over_this"] == pytest.approx(1.0)
    assert by_name["cuda_unfused"]["fused_cuda_speedup_over_this"] == pytest.approx(2.0)
    assert by_name["torch_eager"]["fused_cuda_speedup_over_this"] == pytest.approx(2.0)
    # cuda_fused is excluded from the summary; its own ratio is trivially 1.
    geomean = merged["summary"]["fused_cuda_geomean_speedup_over"]
    assert "cuda_fused" not in geomean
    assert geomean["torch_eager"] == pytest.approx(2.0)


def test_failed_rows_carry_no_latency_and_no_speedup() -> None:
    """A broken baseline must not silently become a fast row."""
    merged = merge_reports(_cuda_report(), _baseline_report())
    triton_row = next(row for row in merged["rows"] if row["baseline"] == "triton")
    assert triton_row["status"] == "error"
    assert "latency_ms" not in triton_row
    assert "fused_cuda_speedup_over_this" not in triton_row
    assert "triton" not in merged["summary"]["fused_cuda_geomean_speedup_over"]


def test_minimum_traffic_is_the_irreducible_fp32_cost() -> None:
    # Two fp32 reads plus one fp32 write per element.
    assert MINIMUM_BYTES_PER_ELEMENT == 12
    assert minimum_traffic_bytes(4096, 4096) == 12 * 4096 * 4096
    # 12 * 4096 * 4096 bytes in 0.8 ms.
    assert minimum_traffic_gbps(4096, 4096, 0.8) == pytest.approx(251.658, rel=1e-4)
    with pytest.raises(ValueError, match="positive"):
        minimum_traffic_gbps(4096, 4096, 0.0)


def test_mixed_tolerance_matches_the_cuda_benchmark_rule() -> None:
    """src/benchmark.cu fails only when BOTH bounds are breached."""
    assert within_tolerance(0.0, 0.0)
    assert within_tolerance(ABSOLUTE_TOLERANCE * 10, 0.0)
    assert within_tolerance(0.0, RELATIVE_TOLERANCE * 10)
    assert not within_tolerance(ABSOLUTE_TOLERANCE * 10, RELATIVE_TOLERANCE * 10)


def test_timing_mode_requires_exactly_one_cuda_source() -> None:
    with pytest.raises(SystemExit):
        main([])
    with pytest.raises(SystemExit):
        main(["--binary", "x", "--cuda-report", "y"])


def test_parser_defaults_match_the_cuda_benchmark_protocol() -> None:
    args = build_parser().parse_args([])
    assert args.iterations == 200
    assert args.warmup == 25
    assert args.epsilon == pytest.approx(1e-5)


def test_correctness_only_mode_writes_a_report(tmp_path) -> None:
    pytest.importorskip("torch", reason="correctness mode needs torch")
    destination = tmp_path / "correctness.json"
    exit_code = main(
        [
            "--correctness-only",
            "--shapes",
            "4x8,3x37",
            "--baselines",
            "torch_eager,torch_explicit,triton",
            "--output",
            str(destination),
        ]
    )
    assert exit_code == 0
    report = json.loads(destination.read_text(encoding="utf-8"))
    statuses = {row["baseline"]: row["status"] for row in report["baselines"]}
    assert statuses["torch_eager"] == "pass"
    assert statuses["torch_explicit"] == "pass"
    # Triton is GPU-only; on a CPU runner it must be labelled, never "pass".
    assert statuses["triton"] in {"gpu_required", "unavailable"}
    assert "epsilon_placement" in report["contract"]


def test_unknown_baseline_name_is_rejected() -> None:
    from kernelforge.baselines import iter_baselines

    with pytest.raises(ValueError, match="unknown baselines: nope"):
        list(iter_baselines(["nope"]))
