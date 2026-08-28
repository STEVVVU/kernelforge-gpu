"""Tests for the PTX codegen check's pure helpers.

The check itself needs Triton (Linux-only wheels), so the interesting parts run
only in the CUDA-container CI job. The pieces tested here are pure, so they can
be pinned everywhere -- which matters because they encode the *rule* the README
cites. If `expects_vectorization` or `alignment_attributes` silently changed, the
Linux job would keep passing against a moved goalpost.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "triton_codegen_check.py"


def load_check_module():
    """Import the script by path; it is a script, not an installed module."""
    spec = importlib.util.spec_from_file_location("triton_codegen_check", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = load_check_module()


def test_script_imports_without_triton_installed():
    """Importing must not require Triton, so these tests run on any platform."""
    assert check.TRITON_DIVISIBILITY == 16
    # The CUDA kernel's own divisor. The gap between 4 and 16 IS the finding, so
    # if these ever became equal the documented band would not exist.
    assert check.TRITON_DIVISIBILITY != 4


@pytest.mark.parametrize(
    ("hidden", "expected"),
    [
        (768, True),
        (1024, True),
        (4096, True),
        (8192, True),
        # Divisible by 4 (so the CUDA kernel vectorizes) but not by 16, which is
        # the one case where the two implementations disagree.
        (4092, False),
        (1028, False),
        # Divisible by neither; both go scalar.
        (4095, False),
    ],
)
def test_expected_vectorization_follows_the_divisibility_rule(hidden, expected):
    assert check.expects_vectorization(hidden) is expected


def test_disagreement_band_is_non_empty():
    """A width divisible by 4 but not 16 must exist in the default set.

    Without such a width the check would compile only widths where both
    implementations agree, and would therefore be unable to observe the finding
    at all -- passing while proving nothing.
    """
    disagreeing = [
        width
        for width in check.DEFAULT_WIDTHS
        if width % 4 == 0 and not check.expects_vectorization(width)
    ]
    assert disagreeing, f"no CUDA-vectorizes/Triton-does-not width in {check.DEFAULT_WIDTHS}"


def test_ragged_control_width_is_present():
    """A width divisible by neither divisor keeps the masked-tail path covered."""
    assert any(width % 4 != 0 for width in check.DEFAULT_WIDTHS)


def test_pointers_are_always_hinted_and_bound_is_conditional():
    """Alignment hints must mirror what a real torch launch would specialize on.

    Pointers always (torch's allocator over-aligns); the width-derived integers
    only when truly divisible. Hinting the bound unconditionally would be a false
    promise that makes the ragged widths look artificially good.
    """
    pointer_indices = {
        check.ARGUMENT_INDEX[name]
        for name in ("x_pointer", "residual_pointer", "weight_pointer", "output_pointer")
    }
    integer_indices = {check.ARGUMENT_INDEX["row_stride"], check.ARGUMENT_INDEX["hidden"]}

    aligned = {index for (index,) in check.alignment_attributes(4096)}
    assert aligned == pointer_indices | integer_indices

    ragged = {index for (index,) in check.alignment_attributes(4092)}
    assert ragged == pointer_indices
    assert not (ragged & integer_indices)


def test_alignment_hint_value_is_the_divisibility_tier():
    for hint in check.alignment_attributes(4096).values():
        assert hint == [["tt.divisibility", check.TRITON_DIVISIBILITY]]


@pytest.mark.parametrize(
    ("accesses", "vector", "scalar"),
    [
        ({"ld.global.v4.b32": 3, "st.global.v4.b32": 1}, 4, 0),
        ({"ld.global.b32": 48, "st.global.b32": 16}, 0, 64),
        ({"ld.global.v2.b32": 2, "ld.global.b32": 5}, 2, 5),
        ({}, 0, 0),
    ],
)
def test_vector_and_scalar_split(accesses, vector, scalar):
    assert check.vector_and_scalar(accesses) == (vector, scalar)


def test_global_access_pattern_ignores_shared_and_local_traffic():
    """Only *global* traffic speaks to DRAM bandwidth."""
    ptx = """
        ld.global.v4.b32 {%f1,%f2,%f3,%f4}, [%rd1];
        ld.shared.b32 %f5, [%rd2];
        st.local.b32 [%rd3], %f6;
        st.global.b32 [%rd4], %f7;
    """
    assert check.count_global_accesses(ptx) == {
        "ld.global.v4.b32": 1,
        "st.global.b32": 1,
    }
