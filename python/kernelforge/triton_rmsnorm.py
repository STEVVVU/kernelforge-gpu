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
  kernel.
* **One on-chip pass.** The row is loaded once into registers and reused for
  the rescale, mirroring the CUDA kernel's shared-memory reuse. Triton chooses
  registers where the CUDA kernel chooses dynamic shared memory; both pay an
  occupancy cost, and both therefore have a width ceiling.

Two things are intentionally *not* matched, because Triton has no equivalent
knob and pretending otherwise would be dishonest:

* The CUDA build uses ``--use_fast_math``; Triton's fp32 defaults are its own.
* The CUDA kernel dispatches an explicit ``float4`` path. Triton emits
  vectorized loads itself based on the access pattern, so there is no
  scalar/vector fork to mirror. This is exactly the "what does the compiler do
  for you" question the comparison exists to answer.
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
