"""Library baselines for the fused residual-RMSNorm operation.

The CUDA kernels in ``src/rmsnorm.cu`` are the subject of this study. This
module supplies the things a reviewer immediately asks about: what does
PyTorch already give you for the same operation, and does the hand-written
kernel beat it?

Every implementation here must compute *exactly* the contract the CUDA kernel
and the CPU oracle implement:

    combined = x + residual
    inverse_rms = 1 / sqrt(mean(combined ** 2, dim=-1) + epsilon)
    output = combined * inverse_rms * weight

Three details make or break the comparison, so they are named explicitly:

1. ``epsilon`` is added to the **mean** of the squares, **inside** the square
   root. Adding it to the sum, or outside the root, changes the result.
2. ``weight`` is applied **after** normalization and is **not** folded into the
   reduction.
3. The sum of squares is accumulated in **fp32** on device (both the CUDA
   kernel's ``float sum`` and Triton's ``tl.sum``). The host oracles
   (``kernelforge.model.rmsnorm_reference`` and ``cpu_reference`` in
   ``src/benchmark.cu``) accumulate in **float64** on purpose, so a small
   fp32-vs-fp64 gap is expected and is not a contract difference.

A comparison against a subtly different formula is meaningless, so
``tests/test_baselines.py`` asserts the equivalence and also asserts that each
of the three wrong variants above is *detectably* different.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

try:  # pragma: no cover - exercised by whichever environment CI provides
    import torch

    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False

# Documented once so tests and docs cannot drift from the implementations.
EPSILON_PLACEMENT = "added to the mean of squares, inside the square root"
WEIGHT_PLACEMENT = "applied after normalization, excluded from the reduction"
DEVICE_ACCUMULATION_DTYPE = "float32"
ORACLE_ACCUMULATION_DTYPE = "float64"

# Matches the generators in src/benchmark.cu: x and residual are N(0, 0.5) and
# weight is U(0.8, 1.2). The values are distribution-matched, not bit-matched
# (different RNG), which does not affect a bandwidth-bound measurement.
INPUT_STDDEV = 0.5
WEIGHT_LOW = 0.8
WEIGHT_HIGH = 1.2


class BaselineUnavailable(RuntimeError):
    """Raised when a baseline cannot be constructed in this environment."""


def _require_torch() -> None:
    if not HAVE_TORCH:
        raise BaselineUnavailable(
            "torch is not installed; install the 'baseline' extra "
            "(pip install -e '.[baseline]')"
        )


def make_inputs(
    rows: int,
    hidden: int,
    *,
    device: str = "cpu",
    dtype: str = "float32",
    seed: int = 12345,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build contiguous ``(x, residual, weight)`` inputs for one shape."""
    _require_torch()
    if rows <= 0 or hidden <= 0:
        raise ValueError("rows and hidden must be positive")
    torch_dtype = getattr(torch, dtype)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(rows, hidden, generator=generator) * INPUT_STDDEV
    residual = torch.randn(rows, hidden, generator=generator) * INPUT_STDDEV
    weight = (
        torch.rand(hidden, generator=generator) * (WEIGHT_HIGH - WEIGHT_LOW) + WEIGHT_LOW
    )
    return (
        x.to(device=device, dtype=torch_dtype).contiguous(),
        residual.to(device=device, dtype=torch_dtype).contiguous(),
        weight.to(device=device, dtype=torch_dtype).contiguous(),
    )


def reference_float64(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """The contract, evaluated in float64.

    This mirrors ``cpu_reference`` in ``src/benchmark.cu`` (double accumulation)
    and is the oracle every device implementation is scored against.
    """
    _require_torch()
    combined = x.to(torch.float64) + residual.to(torch.float64)
    inverse_rms = torch.rsqrt(combined.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    return combined * inverse_rms * weight.to(torch.float64)


def torch_eager(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """``torch.nn.functional.rms_norm`` on an eagerly materialized residual sum.

    This is the "for free" path: one add kernel plus whatever ATen dispatches
    for ``rms_norm``. It is the honest thing to beat.
    """
    _require_torch()
    return torch.nn.functional.rms_norm(
        x + residual, (x.shape[-1],), weight=weight, eps=epsilon
    )


def torch_explicit(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """The same math spelled out in elementwise ops.

    Included because it is what most model code actually contains, and because
    it shows the launch/traffic cost of the unfused decomposition in a
    framework rather than in hand-written CUDA.
    """
    _require_torch()
    combined = x + residual
    inverse_rms = torch.rsqrt(combined.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    return combined * inverse_rms * weight


def _build_torch_module() -> Callable[..., torch.Tensor]:
    """``torch.nn.RMSNorm`` module form, weight-loaded from the input tensor."""
    _require_torch()

    def run(
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        module = torch.nn.RMSNorm(
            x.shape[-1], eps=epsilon, device=x.device, dtype=x.dtype
        )
        with torch.no_grad():
            module.weight.copy_(weight)
        with torch.no_grad():
            return module(x + residual)

    return run


def _build_torch_compiled() -> Callable[..., torch.Tensor]:
    """``torch.compile`` over :func:`torch_explicit`.

    ``torch.compile`` is expected to fuse the add, the reduction, and the scale
    into a single generated Triton kernel, which makes it the closest automated
    analogue of the hand-written fused kernel. Compilation happens on first
    call, so the harness's warmup phase must absorb it.

    Inductor needs a working backend toolchain: a host C++ compiler for the CPU
    backend, Triton for the CUDA one. A tiny probe compile runs here so a
    *missing toolchain* surfaces as ``BaselineUnavailable`` instead of
    masquerading as a numerical failure. Only the probe is translated -- once
    compilation succeeds, a wrong answer still fails the correctness gate.
    """
    _require_torch()
    if not hasattr(torch, "compile"):  # pragma: no cover - very old torch
        raise BaselineUnavailable("this torch build has no torch.compile")
    try:
        compiled = torch.compile(torch_explicit, dynamic=False)
        compiled(*make_inputs(2, 8), 1e-5)
    except Exception as error:  # pragma: no cover - backend/toolchain dependent
        raise BaselineUnavailable(
            "torch.compile backend unavailable in this environment: "
            f"{type(error).__name__}: {error}"
        ) from error
    return compiled


def _build_triton() -> Callable[..., torch.Tensor]:
    from .triton_rmsnorm import HAVE_TRITON, fused_residual_rmsnorm

    if not HAVE_TRITON:
        raise BaselineUnavailable(
            "triton is not installed (Linux-only wheels; install the "
            "'baseline' extra on Linux)"
        )
    return fused_residual_rmsnorm


@dataclass(frozen=True)
class Baseline:
    """One comparable implementation of the residual-RMSNorm contract."""

    name: str
    label: str
    tier: str
    requires_cuda: bool
    build: Callable[[], Callable[..., torch.Tensor]]
    notes: str = ""

    def resolve(self) -> Callable[..., torch.Tensor]:
        """Return the callable, raising :class:`BaselineUnavailable` if absent."""
        return self.build()


BASELINES: tuple[Baseline, ...] = (
    Baseline(
        name="torch_eager",
        label="torch eager (`F.rms_norm`)",
        tier="library",
        requires_cuda=False,
        build=lambda: torch_eager,
        notes="add kernel + ATen rms_norm",
    ),
    Baseline(
        name="torch_explicit",
        label="torch eager (explicit ops)",
        tier="library",
        requires_cuda=False,
        build=lambda: torch_explicit,
        notes="add, pow, mean, rsqrt, two muls",
    ),
    Baseline(
        name="torch_module",
        label="torch `nn.RMSNorm`",
        tier="library",
        requires_cuda=False,
        build=_build_torch_module,
        notes="module form, same ATen path as F.rms_norm",
    ),
    Baseline(
        name="torch_compile",
        label="`torch.compile`",
        tier="compiler",
        requires_cuda=False,
        build=_build_torch_compiled,
        notes="inductor codegen; fuses to generated Triton on CUDA",
    ),
    Baseline(
        name="triton",
        label="OpenAI Triton",
        tier="dsl",
        requires_cuda=True,
        build=_build_triton,
        notes="one program per row, row held in registers",
    ),
)

BASELINES_BY_NAME = {baseline.name: baseline for baseline in BASELINES}

# The hand-written CUDA rows come from build/kernelforge_bench, not from this
# module. Named here so the comparison report and the README table agree on
# labels for every row.
CUDA_ROWS: tuple[tuple[str, str, str], ...] = (
    ("cuda_unfused", "hand-written CUDA, three kernels", "handwritten"),
    ("cuda_fused", "hand-written CUDA, fused", "handwritten"),
)


def iter_baselines(names: list[str] | None = None) -> Iterator[Baseline]:
    """Yield the requested baselines in declaration order."""
    if names is None:
        yield from BASELINES
        return
    unknown = sorted(set(names) - set(BASELINES_BY_NAME))
    if unknown:
        raise ValueError(f"unknown baselines: {', '.join(unknown)}")
    for baseline in BASELINES:
        if baseline.name in names:
            yield baseline
