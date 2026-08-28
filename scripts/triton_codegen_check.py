#!/usr/bin/env python3
"""Lower the Triton RMSNorm kernel to PTX without a GPU.

There is a gap between "the Triton source parses" and "the Triton source is a
valid GPU kernel". Triton's compiler accepts an explicit ``GPUTarget``, so the
whole front end -- AST to Triton IR to TritonGPU IR to LLVM IR to PTX -- can run
on a machine with no NVIDIA device attached. That makes this a real check that
CI can perform on a hosted runner.

What this proves:
  * the kernel lowers cleanly for the requested compute capability;
  * ``num_warps=8`` (matching the CUDA kernel's 256 threads) is a legal
    configuration at each width, including the non-power-of-two control;
  * the *global-access width* Triton actually chose, counted from the emitted
    PTX, and that it follows the documented rule: Triton vectorizes when
    ``hidden % 16 == 0`` and falls back to scalar otherwise. ``--check-rule``
    fails the build if that stops holding, so the note in
    ``kernelforge.triton_rmsnorm`` and the README cannot quietly go stale;
  * that the rule is caused by *divisibility specialization* and not by the
    mask, via the A/B control in :func:`run_hint_control`. Without the control,
    "the PTX is scalar here" would be an unattributed observation that could
    just as easily be a bug in how this script drives the compiler.

What this does NOT prove: any numerical result, and any latency. The kernel is
never executed, and an instruction count is not a runtime. Correctness needs a
GPU (``tests/test_baselines.py``), and so does timing (``kernelforge.compare``).
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

# Counts 4-byte scalar vs 16-byte vector global accesses in the emitted PTX.
GLOBAL_ACCESS_PATTERN = re.compile(r"\b(?:ld|st)\.global[.\w]*")

# Positional index of each kernel parameter, because `tt.divisibility`
# attributes are keyed by argument position, not by name.
ARGUMENT_INDEX = {
    "x_pointer": 0,
    "residual_pointer": 1,
    "weight_pointer": 2,
    "output_pointer": 3,
    "row_stride": 4,
    "hidden": 5,
    "epsilon": 6,
}

# The only divisibility tier Triton tracks. `BaseBackend.parse_attr` emits
# exactly `["tt.divisibility", 16]` and nothing else -- there is no
# divisible-by-4 tier -- which is the root cause of the width band documented
# in kernelforge.triton_rmsnorm.
TRITON_DIVISIBILITY = 16

# Widths mirroring the benchmark matrix in kernelforge.matrix, plus two controls
# that separate the two divisibility rules:
#   4095 -> divisible by neither 4 nor 16; both implementations go scalar.
#   4092 -> divisible by 4 but NOT by 16; the CUDA kernel takes its float4 path
#           and Triton does not. This is the only shape in the set where the two
#           disagree, so it is the one that makes the finding visible.
DEFAULT_WIDTHS = (768, 1024, 4092, 4095, 4096, 8192)

# sm_75 is the Tesla T4 the committed CUDA evidence was measured on, so it is
# the default target: lowering for any other arch would not speak to that
# evidence.
DEFAULT_ARCH = 75

# The A/B control kernels are single-statement, so their warp count is
# irrelevant to the comparison; 4 keeps the emitted PTX short enough to read.
NUM_WARPS_CONTROL = 4


def alignment_attributes(hidden: int) -> dict[tuple[int, ...], list[list[object]]]:
    """Replicate the divisibility hints a real launch would specialize on.

    ``BaseBackend.get_tensor_specialization`` tags a pointer argument with
    ``tt.divisibility = 16`` when ``data_ptr() % 16 == 0``, and
    ``get_int_specialization`` does the same for an integer divisible by 16.
    Supplying these by hand is what makes the emitted PTX representative:
    without them nothing vectorizes at any width, and the scalar accesses would
    be an artifact of this script rather than a property of the kernel.

    Torch's caching allocator returns 512-byte-aligned storage, so all four
    pointers get the hint unconditionally. ``row_stride`` and ``hidden`` get it
    only when the width really is divisible by 16 -- so the 4,092- and
    4,095-wide controls are compiled exactly as unfavourably as they would
    actually run.
    """
    divisible = [["tt.divisibility", TRITON_DIVISIBILITY]]
    attributes: dict[tuple[int, ...], list[list[object]]] = {
        (ARGUMENT_INDEX[name],): divisible
        for name in ("x_pointer", "residual_pointer", "weight_pointer", "output_pointer")
    }
    if hidden % TRITON_DIVISIBILITY == 0:
        attributes[(ARGUMENT_INDEX["row_stride"],)] = divisible
        attributes[(ARGUMENT_INDEX["hidden"],)] = divisible
    return attributes


def expects_vectorization(hidden: int) -> bool:
    """Whether Triton is expected to emit vector global accesses at ``hidden``."""
    return hidden % TRITON_DIVISIBILITY == 0


def count_global_accesses(ptx: str) -> dict[str, int]:
    """Count global load/store opcodes in ``ptx`` by exact mnemonic."""
    return dict(sorted(collections.Counter(GLOBAL_ACCESS_PATTERN.findall(ptx)).items()))


def vector_and_scalar(accesses: dict[str, int]) -> tuple[int, int]:
    """Split access counts into ``(vector, scalar)`` totals."""
    vector = sum(
        count for mnemonic, count in accesses.items() if ".v4" in mnemonic or ".v2" in mnemonic
    )
    return vector, sum(accesses.values()) - vector


def run_hint_control(arch: int, block_size: int = 4096) -> dict[str, object]:
    """Attribute the scalar fallback to divisibility, ruling out the mask.

    Two hypotheses explain a scalar access in a masked Triton kernel:

    1. the mask itself -- a per-lane predicate cannot be folded into one 16-byte
       transaction;
    2. divisibility -- Triton cannot prove the tail bound is 16-divisible, so it
       cannot prove a vector access stays in bounds.

    These predict different things, so they are separable. This control compiles
    a masked and an unmasked kernel, each with and without a divisibility hint on
    the bound. If (1) held, the masked kernel would stay scalar in both cases. It
    does not: hinting the bound vectorizes it. So (2) is the cause, and the mask
    is exonerated.

    Getting this backwards is easy -- an earlier revision of this script supplied
    no hints at all, saw scalar output everywhere, and blamed the mask. The
    control exists so that mistake cannot survive a CI run.
    """
    import triton
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    @triton.jit
    def unmasked(input_pointer, output_pointer, hidden, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        tl.store(output_pointer + offsets, tl.load(input_pointer + offsets) * 2.0)

    @triton.jit
    def masked(input_pointer, output_pointer, hidden, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        mask = offsets < hidden
        value = tl.load(input_pointer + offsets, mask=mask, other=0.0)
        tl.store(output_pointer + offsets, value * 2.0, mask=mask)

    target = GPUTarget("cuda", arch, 32)
    backend = triton.compiler.make_backend(target)
    signature = {
        "input_pointer": "*fp32",
        "output_pointer": "*fp32",
        "hidden": "i32",
        "BLOCK": "constexpr",
    }
    divisible = [["tt.divisibility", TRITON_DIVISIBILITY]]

    def compile_variant(kernel, hint_bound: bool) -> tuple[int, int, dict[str, int]]:
        # Both pointers always hinted, as torch's allocator would guarantee; only
        # the bound hint varies, which is the single variable under test.
        attributes = {(0,): divisible, (1,): divisible}
        if hint_bound:
            attributes[(2,)] = divisible
        source = ASTSource(
            fn=kernel,
            signature=signature,
            constexprs={"BLOCK": block_size},
            attrs=attributes,
        )
        options = backend.parse_options({"num_warps": NUM_WARPS_CONTROL})
        ptx = triton.compile(source, target=target, options=options.__dict__).asm["ptx"]
        accesses = count_global_accesses(ptx)
        vector, scalar = vector_and_scalar(accesses)
        return vector, scalar, accesses

    outcome: dict[str, object] = {}
    for name, kernel in (("unmasked", unmasked), ("masked", masked)):
        for hint_bound in (True, False):
            vector, scalar, accesses = compile_variant(kernel, hint_bound)
            key = f"{name}_bound_hinted" if hint_bound else f"{name}_bound_unhinted"
            outcome[key] = {
                "global_accesses": accesses,
                "vector_global_accesses": vector,
                "scalar_global_accesses": scalar,
            }
            print(
                f"  control {name:<9} bound_hinted={hint_bound!s:<5} "
                f"vector={vector:<3} scalar={scalar:<3} {accesses}"
            )

    def vectorized(key: str) -> bool:
        return bool(outcome[key]["vector_global_accesses"])  # type: ignore[index]

    # The mask is exonerated precisely when a masked kernel vectorizes given the
    # bound hint and goes scalar without it.
    attributable = vectorized("masked_bound_hinted") and not vectorized("masked_bound_unhinted")
    outcome["divisibility_explains_fallback"] = attributable
    outcome["mask_alone_suppresses_vectorization"] = not vectorized("masked_bound_hinted")
    print(
        "  control conclusion: "
        + (
            "divisibility specialization -- not the mask -- decides "
            "vectorization (finding is attributable)"
            if attributable
            else "INCONCLUSIVE -- re-derive the cause before documenting it"
        )
    )
    return outcome


def compile_for_arch(
    arch: int,
    widths: tuple[int, ...],
    *,
    check_rule: bool = False,
    report_path: Path | None = None,
    run_control: bool = True,
) -> int:
    """Compile the kernel for each width. Returns a process exit code."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from kernelforge.triton_rmsnorm import (
        NUM_WARPS,
        _fused_residual_rmsnorm_kernel,
        next_power_of_two,
    )

    kernel = _fused_residual_rmsnorm_kernel
    # NVIDIA warps are 32 lanes wide on every supported architecture.
    target = GPUTarget("cuda", arch, 32)
    backend = triton.compiler.make_backend(target)
    print(f"triton {triton.__version__} -> sm_{arch}, num_warps={NUM_WARPS}")

    failures = 0
    widths_report: list[dict[str, object]] = []
    for hidden in widths:
        block_size = next_power_of_two(hidden)
        # Pointers are fp32; row_stride and hidden are i32; epsilon is fp32.
        # BLOCK_SIZE is a constexpr and so is passed as a constant, not a
        # signature entry.
        signature = {
            "x_pointer": "*fp32",
            "residual_pointer": "*fp32",
            "weight_pointer": "*fp32",
            "output_pointer": "*fp32",
            "row_stride": "i32",
            "hidden": "i32",
            "epsilon": "fp32",
            "BLOCK_SIZE": "constexpr",
        }
        # `ASTSource` is imported directly rather than via
        # `kernel.create_binder()`, which calls `driver.active` and therefore
        # raises "0 active drivers" on a machine with no GPU -- the exact
        # machine this check is meant to run on.
        attributes = alignment_attributes(hidden)
        source = ASTSource(
            fn=kernel,
            signature=signature,
            constexprs={"BLOCK_SIZE": block_size},
            attrs=attributes,
        )
        options = backend.parse_options({"num_warps": NUM_WARPS})
        try:
            compiled = triton.compile(source, target=target, options=options.__dict__)
        except Exception as error:  # noqa: BLE001 - reported per width, not swallowed
            failures += 1
            print(
                f"  FAIL hidden={hidden:>5} block={block_size:>5}: "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
            )
            continue
        metadata = compiled.metadata
        shared = getattr(metadata, "shared", "n/a")
        ptx = compiled.asm.get("ptx", "")
        accesses = count_global_accesses(ptx)
        vector_accesses, scalar_accesses = vector_and_scalar(accesses)
        expected = expects_vectorization(hidden)
        # The CUDA kernel forks on `hidden % 4 == 0`; Triton effectively forks on
        # `% 16`. Recording both makes the disagreement band explicit per width.
        cuda_takes_vector_path = hidden % 4 == 0
        print(
            f"  ok   hidden={hidden:>5} block={block_size:>5} "
            f"shared={shared} ptx_lines={len(ptx.splitlines())} "
            f"global_scalar={scalar_accesses} global_vector={vector_accesses} "
            f"cuda_float4={cuda_takes_vector_path!s:<5} "
            f"triton_vector={bool(vector_accesses)!s:<5}"
        )
        widths_report.append(
            {
                "hidden": hidden,
                "block_size": block_size,
                "shared_bytes": shared,
                "alignment_hinted_arguments": sorted(index for (index,) in attributes),
                "global_accesses": accesses,
                "scalar_global_accesses": scalar_accesses,
                "vector_global_accesses": vector_accesses,
                "cuda_takes_float4_path": cuda_takes_vector_path,
                "triton_vectorized": bool(vector_accesses),
                "triton_vectorization_expected": expected,
                "implementations_disagree": cuda_takes_vector_path != bool(vector_accesses),
            }
        )
        if not ptx:
            failures += 1
            print(f"  FAIL hidden={hidden}: no PTX emitted", file=sys.stderr)
        if check_rule and bool(vector_accesses) != expected:
            # Not a bug -- a documentation obligation. The README and the
            # triton_rmsnorm docstring both assert that Triton vectorizes exactly
            # when `hidden % 16 == 0`. If that stops holding, those claims must be
            # rewritten rather than left standing as a stale measurement.
            failures += 1
            print(
                f"  FAIL hidden={hidden}: expected "
                f"{'vector' if expected else 'scalar-only'} global accesses "
                f"(hidden % {TRITON_DIVISIBILITY} "
                f"{'==' if expected else '!='} 0) but found "
                f"{vector_accesses} vector and {scalar_accesses} scalar: {accesses}. "
                "Triton's codegen changed; update the vectorization note in "
                "python/kernelforge/triton_rmsnorm.py and the README.",
                file=sys.stderr,
            )

    control: dict[str, object] | None = None
    if run_control:
        print("divisibility-vs-mask A/B control:")
        control = run_hint_control(arch)
        if not control["divisibility_explains_fallback"]:
            # The per-width counts above are still real, but their documented
            # cause would be unsupported, so fail rather than let a stale
            # explanation stand.
            failures += 1
            print(
                "  FAIL control: divisibility no longer explains the scalar "
                "fallback; re-derive the cause before trusting the vectorization "
                "note in python/kernelforge/triton_rmsnorm.py",
                file=sys.stderr,
            )

    if report_path is not None:
        report = {
            "triton_version": triton.__version__,
            "target_arch": f"sm_{arch}",
            "num_warps": NUM_WARPS,
            "executed": False,
            "note": (
                "PTX-only lowering on a host with no GPU. Instruction counts, "
                "not timings; no latency or correctness is implied."
            ),
            "triton_divisibility_tier": TRITON_DIVISIBILITY,
            "cuda_vector_path_divisor": 4,
            "widths": widths_report,
            "divisibility_vs_mask_control": control,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {report_path}")

    if failures:
        print(f"{failures} check(s) failed", file=sys.stderr)
        return 1
    print(f"all {len(widths)} widths lowered to PTX for sm_{arch} (not executed)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch",
        type=int,
        default=DEFAULT_ARCH,
        help=f"CUDA compute capability without the dot (default {DEFAULT_ARCH} = Tesla T4)",
    )
    parser.add_argument(
        "--widths",
        default=",".join(str(width) for width in DEFAULT_WIDTHS),
        help="comma-separated hidden widths to lower",
    )
    parser.add_argument(
        "--check-rule",
        action="store_true",
        help=(
            "fail unless each width vectorizes exactly when "
            f"hidden %% {TRITON_DIVISIBILITY} == 0; guards the documented "
            "vectorization finding against a silent Triton codegen change"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write a JSON report of the per-width PTX access counts",
    )
    parser.add_argument(
        "--no-control",
        action="store_true",
        help=(
            "skip the divisibility-vs-mask A/B control (it is what makes the "
            "scalar-fallback finding attributable)"
        ),
    )
    args = parser.parse_args(argv)
    widths = tuple(int(part) for part in args.widths.split(","))

    try:
        import triton  # noqa: F401
    except ImportError:
        print(
            "triton is not installed; wheels are published for Linux only. "
            "Install the 'baseline' extra on Linux.",
            file=sys.stderr,
        )
        return 2
    return compile_for_arch(
        args.arch,
        widths,
        check_rule=args.check_rule,
        report_path=args.output,
        run_control=not args.no_control,
    )


if __name__ == "__main__":
    raise SystemExit(main())
