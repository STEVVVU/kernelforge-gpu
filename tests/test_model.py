import math

from kernelforge.model import rmsnorm_reference, traffic_estimate


def test_reference_matches_hand_computation() -> None:
    result = rmsnorm_reference([[1.0, 2.0]], [[1.0, 0.0]], [1.0, 0.5], epsilon=0.0)
    inverse = 1.0 / math.sqrt(4.0)
    assert result == [[2.0 * inverse, 2.0 * inverse * 0.5]]


def test_fusion_reduces_global_memory_traffic() -> None:
    estimate = traffic_estimate(4096, 4096)
    assert estimate.fused_bytes < estimate.unfused_bytes
    assert 0.42 < estimate.traffic_reduction < 0.43
    assert 1.74 < estimate.bandwidth_bound_speedup_ceiling < 1.76


def test_invalid_shapes_are_rejected() -> None:
    try:
        rmsnorm_reference([[1.0]], [[1.0, 2.0]], [1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch was accepted")

