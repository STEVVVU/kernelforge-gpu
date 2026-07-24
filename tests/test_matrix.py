import math

import pytest

from kernelforge.matrix import parse_shapes, summarize_results


def test_shape_parser_includes_vectorized_and_scalar_controls() -> None:
    assert parse_shapes("4096x4096,4096x4095") == [(4096, 4096), (4096, 4095)]
    with pytest.raises(ValueError, match="expected ROWSxHIDDEN"):
        parse_shapes("4096")
    with pytest.raises(ValueError, match="positive"):
        parse_shapes("0x4096")


def test_matrix_summary_uses_geometric_mean_and_error_maximum() -> None:
    results = [
        {
            "device": "Test GPU",
            "speedup": 2.0,
            "fused_max_absolute_error": 1e-6,
        },
        {
            "device": "Test GPU",
            "speedup": 8.0,
            "fused_max_absolute_error": 2e-6,
        },
    ]
    summary = summarize_results(results)
    assert math.isclose(summary["geometric_mean_speedup"], 4.0)
    assert summary["minimum_speedup"] == 2.0
    assert summary["maximum_speedup"] == 8.0
    assert summary["maximum_absolute_error"] == 2e-6


def test_matrix_rejects_mixed_devices() -> None:
    with pytest.raises(ValueError, match="same device"):
        summarize_results(
            [
                {"device": "GPU A", "speedup": 2.0, "fused_max_absolute_error": 0.0},
                {"device": "GPU B", "speedup": 2.0, "fused_max_absolute_error": 0.0},
            ]
        )

