"""OpenAI Triton implementation of the fused residual RMSNorm.

This is the closest apples-to-apples competitor to ``fused_vector_kernel`` /
``fused_scalar_kernel`` in ``src/rmsnorm.cu``: one program per row, the row
held on-chip for both the reduction and the rescale, one output write, no
global scratch buffer.

Deliberate parity choices with the CUDA kernel, so a latency difference is
attributable to codegen and not to a different algorithm:

* **One program per row.** ``grid = (rows,)``, exactly like the CUDA
  ``<<<rows, kThreads>>>`` launch.
* **``num_warps=8``.** The CUDA kernel launches ``kThreads = 256`` threads per
  block, which is 8 warps. Triton's block size is expressed in warps, so 8
  keeps the per-row parallelism identical.
* **fp32 accumulation.** ``tl.sum`` over an fp32 tile matches the CUDA
  kernel's ``float sum``. Neither upcasts to fp64.
* **Same epsilon placement.** ``epsilon`` is added to the mean of squares
  inside the square root.
* **Masked tail.** Out-of-range lanes load ``0.0``, so they contribute nothing
  to the sum-of-squares. That is what lets the 4,095-wide control shape run on
  the same kernel, where the CUDA side instead dispatches a separate scalar
  kernel. The mask costs nothing in access width -- see the note below.
* **One on-chip pass.** The row is loaded once into registers and reused for
  the rescale, mirroring the CUDA kernel's shared-memory reuse. Triton chooses
  registers where the CUDA kernel chooses dynamic shared memory; both pay an
  occupancy cost, and both therefore have a width ceiling.

One thing is intentionally *not* matched, because Triton has no equivalent knob:
the CUDA build uses ``--use_fast_math``, while Triton's fp32 lowering is its own
(it emits ``sqrt.approx.ftz.f32`` here, which is the fast path anyway).

Vectorization: the two implementations disagree on a band of widths
------------------------------------------------------------------

The CUDA kernel forks an explicit ``float4`` path (``fused_vector_kernel``) at
``hidden % 4 == 0``. Triton also emits 16-byte accesses -- but it decides on a
*coarser* rule, and the gap between the two rules is a real, narrow band of
widths where the hand-written kernel gets wide accesses and Triton does not.

Triton vectorizes only what it can prove in bounds. Its sole divisibility tier
is 16: ``BaseBackend.parse_attr`` emits exactly ``["tt.divisibility", 16]`` and
nothing else, set by ``get_int_specialization`` (``arg % 16 == 0``) and
``get_tensor_specialization`` (``data_ptr() % 16 == 0``). There is no
divisible-by-4 tier, so a width that is a multiple of 4 but not of 16 carries no
usable hint and the whole row falls back to ``ld.global.b32``.

Measured by lowering to PTX for sm_75 with the hints a real torch launch would
specialize on (``scripts/triton_codegen_check.py``, which CI runs; counts in
``artifacts/triton_ptx_vectorization.json``):

========  ==================  =====================  ==========
 hidden    CUDA ``float4``?    Triton vector PTX?     verdict
========  ==================  =====================  ==========
    768    yes                 yes                    agree
   1024    yes                 yes                    agree
   4092    yes                 **no** (64 scalar)     CUDA only
   4095    no                  no                     agree
   4096    yes                 yes                    agree
   8192    yes                 yes                    agree
========  ==================  =====================  ==========

The mask is **not** the cause, which is worth stating because it is the
intuitive guess and it is wrong. An A/B control (``run_hint_control`` in the same
script) compiles a masked kernel with and without a divisibility hint on the
bound: hinted, it emits ``ld.global.v4.b32``; unhinted, it goes scalar. A masked
kernel vectorizes perfectly well. Predication costs nothing in access width;
only unprovable divisibility does.

The framing this supports is narrower than "hand-written CUDA vectorizes and
Triton doesn't", and narrower is the honest version. Both reach 16-byte accesses
on the shapes real models use -- 768, 1024, 4096, 8192 are all multiples of 16,
and on those Triton needs no help. What the hand-written kernel buys is control
over *where the cliff sits*: it chose the divisor (4) and can fork on it, while
Triton's divisor is fixed at 16 by its specialization machinery and is not a
parameter the kernel author can reach. On a width like 4,092 that difference is
the whole access width of the kernel. Knowing the cliff exists, and being able to
move it, is the argument for hand-writing -- not a blanket codegen deficit.

Whether those extra transactions actually cost latency on a T4 is a separate
question that needs a GPU; an instruction count is not a runtime. The comparison
harness measures that when a GPU runner is attached.
"""

from __future__ import annotations

try:  # pragma: no cover - depends on the environment CI provides
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

# Widths above this need a multi-pass kernel. The CUDA side has the analogous
# boundary at the device's opt-in dynamic shared-memory limit and returns
# cudaErrorInvalidValue past it; this raises rather than silently degrading.
MAX_SINGLE_PASS_HIDDEN = 65536

# Matches kThreads = 256 in src/rmsnorm.cu (256 threads == 8 warps).
NUM_WARPS = 8


def next_power_of_two(value: int) -> int:
    """Smallest power of two greater than or equal to ``value`` (``value >= 1``)."""
    if value < 1:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


if HAVE_TRITON:  # pragma: no cover - requires a Triton install

    @triton.jit
    def _fused_residual_rmsnorm_kernel(
        x_pointer,
        residual_pointer,
        weight_pointer,
        output_pointer,
        row_stride,
        hidden,
        epsilon,
        BLOCK_SIZE: tl.constexpr,
    ):
        """One program handles one row: add, reduce, rescale, store."""
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden
        row_offset = row * row_stride

        # Masked lanes read 0.0 so they contribute 0 to the sum of squares.
        x = tl.load(x_pointer + row_offset + offsets, mask=mask, other=0.0)
        residual = tl.load(residual_pointer + row_offset + offsets, mask=mask, other=0.0)
        combined = x + residual

        # fp32 accumulation, matching the CUDA kernel's `float sum`.
        square_sum = tl.sum(combined * combined, axis=0)
        # `epsilon` goes inside the root, added to the *mean* -- and `hidden`,
        # not BLOCK_SIZE, is the divisor, so the padded tail cannot skew it.
        # Spelled as 1/sqrt rather than an rsqrt intrinsic because the exact
        # intrinsic name has moved across Triton releases; the division is
        # exact enough that it does not affect the fp32 comparison.
        inverse_rms = 1.0 / tl.sqrt(square_sum / hidden + epsilon)

        weight = tl.load(weight_pointer + offsets, mask=mask, other=0.0)
        # Reuses `combined` from registers -- no second global read, the same
        # saving the CUDA kernel gets from shared memory.
        tl.store(
            output_pointer + row_offset + offsets,
            combined * inverse_rms * weight,
            mask=mask,
        )


def fused_residual_rmsnorm(x, residual, weight, epsilon: float):
    """Launch the Triton kernel for ``output = RMSNorm(x + residual) * weight``.

    Args:
        x: ``(rows, hidden)`` contiguous tensor.
        residual: ``(rows, hidden)`` contiguous tensor, same dtype as ``x``.
        weight: ``(hidden,)`` contiguous tensor, same dtype as ``x``.
        epsilon: added to the mean of squares inside the square root.

    Returns:
        A new ``(rows, hidden)`` tensor with the same dtype and device as ``x``.
    """
    if not HAVE_TRITON:
        raise RuntimeError(
            "triton is not installed; wheels are published for Linux only"
        )
    import torch

    if x.shape != residual.shape or x.ndim != 2:
        raise ValueError("x and residual must be 2-D tensors of the same shape")
    rows, hidden = x.shape
    if weight.shape != (hidden,):
        raise ValueError("weight must have shape (hidden,)")
    if hidden > MAX_SINGLE_PASS_HIDDEN:
        raise ValueError(
            f"hidden={hidden} exceeds the single-pass limit "
            f"{MAX_SINGLE_PASS_HIDDEN}; a tiled two-pass kernel is required"
        )
    if not (x.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()):
        raise ValueError("inputs must be contiguous")

    output = torch.empty_like(x)
    _fused_residual_rmsnorm_kernel[(rows,)](
        x,
        residual,
        weight,
        output,
        x.stride(0),
        hidden,
        epsilon,
        BLOCK_SIZE=next_power_of_two(hidden),
        num_warps=NUM_WARPS,
    )
    return output
