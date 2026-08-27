"""Baselines must implement the CUDA kernel's contract, not an approximation.

These run on CPU, so they gate the comparison in CI without a GPU. The Triton
kernel itself is GPU-only and is marked accordingly.
"""

from __future__ import annotations

import math

import pytest

from kernelforge import baselines
from kernelforge.baselines import (
    make_inputs,
    reference_float64,
    torch_eager,
    torch_explicit,
)
from kernelforge.model import rmsnorm_reference

torch = pytest.importorskip("torch", reason="torch baselines require the 'baseline' extra")

EPSILON = 1e-5
# fp32 tolerance for a contract-equivalent implementation scored against the
# float64 oracle. Deliberately far tighter than the benchmark's 2e-5 pass gate:
# a formula difference (wrong epsilon placement, weight in the reduction) moves
# the result orders of magnitude beyond this, so this bound is what makes the
# equivalence assertion meaningful rather than vacuous.
EQUIVALENCE_TOLERANCE = 1e-5

# Small shapes plus a non-power-of-two width, which is the case that exercises
# masked tails in Triton and the scalar dispatch in CUDA.
SHAPES = [(4, 8), (3, 37), (2, 129), (5, 256)]


@pytest.mark.parametrize("implementation", [torch_eager, torch_explicit])
@pytest.mark.parametrize(("rows", "hidden"), SHAPES)
def test_torch_baselines_match_float64_oracle(implementation, rows, hidden) -> None:
    x, residual, weight = make_inputs(rows, hidden)
    expected = reference_float64(x, residual, weight, EPSILON)
    actual = implementation(x, residual, weight, EPSILON)
    assert actual.shape == expected.shape
    assert actual.dtype == torch.float32
    difference = (actual.to(torch.float64) - expected).abs().max().item()
    assert difference < EQUIVALENCE_TOLERANCE, f"max abs error {difference}"


@pytest.mark.parametrize(("rows", "hidden"), SHAPES)
def test_torch_module_baseline_matches_oracle(rows, hidden) -> None:
    implementation = baselines.BASELINES_BY_NAME["torch_module"].resolve()
    x, residual, weight = make_inputs(rows, hidden)
    expected = reference_float64(x, residual, weight, EPSILON)
    actual = implementation(x, residual, weight, EPSILON)
    assert (actual.to(torch.float64) - expected).abs().max().item() < EQUIVALENCE_TOLERANCE


def test_torch_baselines_agree_bitwise_with_each_other() -> None:
    """`F.rms_norm` and the explicit ops must be the same computation.

    If ATen ever changes its internal order enough to break this, the
    comparison table needs a footnote rather than a silent divergence.
    """
    x, residual, weight = make_inputs(6, 64)
    eager = torch_eager(x, residual, weight, EPSILON)
    explicit = torch_explicit(x, residual, weight, EPSILON)
    assert torch.equal(eager, explicit)


@pytest.mark.parametrize(("rows", "hidden"), SHAPES)
def test_torch_baseline_matches_the_pure_python_oracle(rows, hidden) -> None:
    """Ties the torch path to the in-repo reference the CUDA build is scored on.

    ``kernelforge.model.rmsnorm_reference`` is the same formula as
    ``cpu_reference`` in ``src/benchmark.cu``, so agreeing with it is agreeing
    with the CUDA kernel's contract.
    """
    x, residual, weight = make_inputs(rows, hidden)
    expected = torch.tensor(
        rmsnorm_reference(x.tolist(), residual.tolist(), weight.tolist(), epsilon=EPSILON),
        dtype=torch.float64,
    )
    actual = torch_eager(x, residual, weight, EPSILON).to(torch.float64)
    assert (actual - expected).abs().max().item() < EQUIVALENCE_TOLERANCE


# --- Negative controls -------------------------------------------------------
# The equivalence assertions above are only worth something if a *wrong*
# formula would actually fail them. Each of these is a plausible mistake; each
# must be detectable at EQUIVALENCE_TOLERANCE. A tolerance loose enough to
# accept these would make the whole comparison meaningless.


def _max_deviation(actual) -> float:
    x, residual, weight = make_inputs(6, 64)
    expected = reference_float64(x, residual, weight, EPSILON)
    return (actual(x, residual, weight).to(torch.float64) - expected).abs().max().item()


def test_epsilon_outside_the_root_is_detectable() -> None:
    def wrong(x, residual, weight):
        combined = x + residual
        return combined * (1.0 / torch.sqrt(combined.pow(2).mean(-1, keepdim=True)) + EPSILON) * weight

    assert _max_deviation(wrong) > EQUIVALENCE_TOLERANCE


def test_epsilon_added_to_the_sum_instead_of_the_mean_is_detectable() -> None:
    def wrong(x, residual, weight):
        combined = x + residual
        hidden = combined.shape[-1]
        inverse = torch.rsqrt((combined.pow(2).sum(-1, keepdim=True) + EPSILON) / hidden)
        return combined * inverse * weight

    assert _max_deviation(wrong) > EQUIVALENCE_TOLERANCE


def test_weight_folded_into_the_reduction_is_detectable() -> None:
    def wrong(x, residual, weight):
        combined = (x + residual) * weight
        return combined * torch.rsqrt(combined.pow(2).mean(-1, keepdim=True) + EPSILON)

    assert _max_deviation(wrong) > EQUIVALENCE_TOLERANCE


def test_layernorm_instead_of_rmsnorm_is_detectable() -> None:
    """RMSNorm has no mean subtraction; LayerNorm does. Common confusion."""

    def wrong(x, residual, weight):
        combined = x + residual
        return torch.nn.functional.layer_norm(
            combined, (combined.shape[-1],), weight=weight, eps=EPSILON
        )

    assert _max_deviation(wrong) > EQUIVALENCE_TOLERANCE


# --- Input generation --------------------------------------------------------


def test_inputs_are_deterministic_and_distribution_matched() -> None:
    first = make_inputs(4, 16, seed=7)
    second = make_inputs(4, 16, seed=7)
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left, right)
    _, _, weight = make_inputs(64, 512)
    # Matches the U(0.8, 1.2) weight generator in src/benchmark.cu.
    assert weight.min().item() >= baselines.WEIGHT_LOW
    assert weight.max().item() <= baselines.WEIGHT_HIGH


def test_inputs_reject_non_positive_shapes() -> None:
    with pytest.raises(ValueError, match="positive"):
        make_inputs(0, 16)


def test_inputs_are_contiguous() -> None:
    """Triton and the CUDA launcher both assume row-major contiguity."""
    for tensor in make_inputs(4, 33):
        assert tensor.is_contiguous()


# --- Triton --------------------------------------------------------------------


def test_triton_helper_rounds_up_to_a_power_of_two() -> None:
    """Pure helper, testable without a GPU or a Triton install."""
    from kernelforge.triton_rmsnorm import next_power_of_two

    assert next_power_of_two(1) == 1
    assert next_power_of_two(4095) == 4096
    assert next_power_of_two(4096) == 4096
    assert next_power_of_two(4097) == 8192
    with pytest.raises(ValueError):
        next_power_of_two(0)


def test_triton_baseline_is_declared_gpu_required() -> None:
    """The report must never present the Triton row as CPU-verified."""
    assert baselines.BASELINES_BY_NAME["triton"].requires_cuda is True
    assert baselines.BASELINES_BY_NAME["torch_eager"].requires_cuda is False


@pytest.mark.parametrize(("rows", "hidden"), SHAPES)
def test_triton_matches_oracle_on_gpu(rows, hidden) -> None:
    """GPU-required: the Triton kernel's own numerics.

    Skipped on CPU runners. This is the assertion that must go green on a GPU
    runner before any Triton latency row is published.
    """
    triton_rmsnorm = pytest.importorskip(
        "kernelforge.triton_rmsnorm", reason="triton is Linux-only"
    )
    if not triton_rmsnorm.HAVE_TRITON:
        pytest.skip("triton is not installed")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device visible")
    x, residual, weight = make_inputs(rows, hidden, device="cuda")
    expected = reference_float64(x, residual, weight, EPSILON)
    actual = triton_rmsnorm.fused_residual_rmsnorm(x, residual, weight, EPSILON)
    assert (actual.to(torch.float64) - expected).abs().max().item() < EQUIVALENCE_TOLERANCE


def test_triton_rejects_widths_beyond_the_single_pass_limit() -> None:
    from kernelforge import triton_rmsnorm

    if not triton_rmsnorm.HAVE_TRITON:
        pytest.skip("triton is not installed")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device visible")
    hidden = triton_rmsnorm.MAX_SINGLE_PASS_HIDDEN + 4
    x, residual, weight = make_inputs(1, hidden, device="cuda")
    with pytest.raises(ValueError, match="single-pass limit"):
        triton_rmsnorm.fused_residual_rmsnorm(x, residual, weight, EPSILON)


def test_geometric_mean_matches_hand_computation() -> None:
    """Guards the summary math the README's speedup column is derived from."""
    from kernelforge.compare import summarize_comparison

    rows = [
        {
            "baseline": "torch_eager",
            "status": "ok",
            "rows": 4096,
            "hidden": 4096,
            "fused_cuda_speedup_over_this": 2.0,
        },
        {
            "baseline": "torch_eager",
            "status": "ok",
            "rows": 4096,
            "hidden": 768,
            "fused_cuda_speedup_over_this": 8.0,
        },
    ]
    summary = summarize_comparison(rows)
    assert math.isclose(
        summary["fused_cuda_geomean_speedup_over"]["torch_eager"], 4.0
    )
    assert summary["shapes_compared"] == 2
