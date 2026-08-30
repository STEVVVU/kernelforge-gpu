# KernelForge GPU

A CUDA C++ optimization study of a transformer primitive:

```text
output = RMSNorm(x + residual) * weight
```

The project starts from a readable three-kernel implementation, fuses the
operation into one cooperative thread block per row, validates both paths
against a double-accumulation CPU reference, and reports latency and effective
DRAM bandwidth with CUDA events.

## Optimization story

The unfused path launches:

1. residual add, writing an intermediate tensor;
2. row-wise sum-of-squares reduction;
3. scale and normalize.

The fused path:

- reads `x` and `residual` once;
- caches their sum in dynamic shared memory;
- reduces with warp shuffles and one value per warp;
- reuses the shared values for normalization;
- vectorizes aligned widths with `float4`;
- falls back to a scalar path for widths not divisible by four;
- emits one output tensor with no global scratch buffer.

This is a bandwidth-focused optimization. The fused kernel trades shared-memory
capacity for fewer launches and less DRAM traffic.

For how this compares to `torch`, `torch.compile`, and Triton -- and why a
hand-written kernel is still worth writing when those exist -- see
[library baselines](#library-baselines-torch-torchcompile-triton) and
[why hand-write this kernel](#why-hand-write-this-kernel).

## Resume highlights

- Combined three transformer operations into one CUDA C++ RMSNorm kernel,
  eliminating intermediate memory writes and extra kernel launches.
- Achieved a **1.876x geometric-mean speedup** across five NVIDIA Tesla T4
  workloads, with a **2.059x** peak speedup and **1e-6** maximum error.
- Reduced modeled global-memory traffic by **42.86%** using `float4` loads,
  shared-memory reuse, and warp-shuffle reductions.
- Built reproducible CUDA-event benchmarks, CPU correctness checks, scalar
  fallback coverage, CUDA compile CI, and a checked-in Nsight Compute report.

## Measured Tesla T4 result

Each workload used 25 warmup launches followed by 200 CUDA-event-timed
iterations on a Tesla T4 (compute capability 7.5) with CUDA 13.2. The matrix
includes common transformer widths, a non-vectorized 4,095-wide control, and
an 8,192-wide row.

| Shape | Fused dispatch | Baseline | Fused | Speedup |
| --- | --- | ---: | ---: | ---: |
| 4,096 x 768 | `float4` | 0.3159 ms | 0.1534 ms | **2.059x** |
| 4,096 x 1,024 | `float4` | 0.4158 ms | 0.2080 ms | **1.999x** |
| 4,096 x 4,095 | scalar | 1.6043 ms | 1.0194 ms | **1.574x** |
| 4,096 x 4,096 | `float4` | 1.5900 ms | 0.8008 ms | **1.986x** |
| 1,024 x 8,192 | `float4` | 0.8036 ms | 0.4453 ms | **1.805x** |

The geometric-mean speedup is **1.876x**. Every result passed the CPU oracle
with maximum absolute error at or below **1e-6**. The adjacent 4,095/4,096
cases exercise scalar and vector dispatch without materially changing the
tensor size; the vector path reduced fused latency by 21.4%.

Raw results, device metadata, and per-shape correctness values are committed
in [`artifacts/benchmark_matrix.json`](artifacts/benchmark_matrix.json). The
corresponding Nsight Compute capture is available as
[`artifacts/kernelforge-fused.ncu-rep`](artifacts/kernelforge-fused.ncu-rep).
These measurements establish performance on one T4, not every GPU
architecture.

> **Where these numbers came from.** They were measured on a dedicated Tesla T4
> host, not in CI. No GitHub-hosted runner has an NVIDIA GPU, so CI never
> reproduces them; it verifies compilation and numerics only. Any timing in this
> README is from that T4 run and is labelled as such.

## Library baselines: torch, `torch.compile`, Triton

The first question a reviewer should ask is why this kernel is hand-written when
PyTorch ships `F.rms_norm` and Triton generates competitive kernels from a few
lines of Python. [`python/kernelforge/baselines.py`](python/kernelforge/baselines.py)
implements the same operation five more ways so the comparison is concrete
rather than asserted:

| Implementation | What it is | Correctness (CPU, CI) | Latency / GB/s |
| --- | --- | :---: | :---: |
| hand-written CUDA, fused | the subject of this study | pass (T4) | measured, see above |
| hand-written CUDA, three kernels | the unfused baseline | pass (T4) | measured, see above |
| `torch` eager (`F.rms_norm`) | add kernel + ATen `rms_norm` | pass | GPU-required, not yet run |
| `torch` eager (explicit ops) | add, pow, mean, rsqrt, two muls | pass | GPU-required, not yet run |
| `torch` `nn.RMSNorm` | module form, same ATen path | pass | GPU-required, not yet run |
| `torch.compile` | TorchInductor codegen | pass<sup>1</sup> | GPU-required, not yet run |
| OpenAI Triton | one program per row, row in registers | GPU-required<sup>2</sup> | GPU-required, not yet run |

<sup>1</sup> Verified where Inductor has a working backend. It reports
`unavailable` rather than `error` on a host with no C++ compiler, so a missing
toolchain cannot masquerade as a numerical failure.
<sup>2</sup> Triton publishes manylinux wheels only and its kernels need a real
device, so this row is Linux + GPU by construction. Its *codegen* is still
checked without a GPU (below).

**The latency column is deliberately empty.** Filling it needs a GPU runner, and
a plausible-looking number would be worse than a blank. Remove the `if: false`
guard on the `gpu-comparison` job in
[`.github/workflows/baseline-comparison.yml`](.github/workflows/baseline-comparison.yml)
and attach a CUDA runner; nothing else has to change. The harness refuses to
merge measurements taken on different devices, so the table cannot be
accidentally assembled from two machines.

What *is* established without a GPU is the part that makes a latency table
meaningful at all: every baseline computes the identical formula. Epsilon is
added to the mean of squares inside the root, `weight` is applied after
normalization and excluded from the reduction, and the device path accumulates in
fp32 while the oracle uses float64. Comparing against a subtly different formula
would be meaningless, so
[`tests/test_baselines.py`](tests/test_baselines.py) asserts the equivalence
*and* asserts that four plausible wrong variants -- epsilon outside the root,
epsilon on the sum, weight folded into the reduction, LayerNorm substituted --
are each detectably different. Without those negative controls the equivalence
assertion could pass vacuously.

## Why hand-write this kernel

The honest answer is not "libraries are slow". It is that the fused kernel is
bandwidth-bound, and the three levers that decide a bandwidth-bound kernel's
speed are ones the library abstracts away.

**The ceiling is memory, not math.** The operation does about two flops per
byte moved, so nothing here is compute-limited: the only way to go faster is to
move fewer bytes. That is why the win is structural. Fusing three kernels into
one removes an intermediate tensor write and read, cutting modeled traffic
42.86% and setting a 1.75x ceiling -- which the measured 1.876x geometric mean
sits near, because launch overhead is removed on top of the traffic saving.
`torch` eager cannot do this: `F.rms_norm` does not know a residual add preceded
it, so the sum must be materialized to DRAM before it can be read back.

**Access width is a decision, and its cliff is placed differently.** The CUDA
kernel forks at `hidden % 4 == 0` to take a `float4` path. Triton also emits
16-byte accesses, but on a coarser rule -- its only divisibility tier is 16 --
so the two disagree on a narrow band of widths. Measured by lowering both to PTX
for sm_75 ([`artifacts/triton_ptx_vectorization.json`](artifacts/triton_ptx_vectorization.json)):

| `hidden` | CUDA `float4` path | Triton vector PTX | |
| ---: | :---: | :---: | --- |
| 768 | yes | yes | agree |
| 1,024 | yes | yes | agree |
| 4,092 | yes | **no**, 64 scalar accesses | CUDA only |
| 4,095 | no | no | agree |
| 4,096 | yes | yes | agree |
| 8,192 | yes | yes | agree |

Note what this does *not* say. On the widths real models use -- 768, 1,024,
4,096, 8,192, all multiples of 16 -- Triton vectorizes fine and needs no help,
and the intuitive explanation for the 4,092 case is also wrong: an A/B control
shows a *masked* Triton kernel vectorizes perfectly well once the bound carries a
divisibility hint, so predication is not the cause. What the hand-written kernel
actually buys is control over **where the cliff sits**: it picked the divisor and
forks on it, while Triton's divisor is fixed at 16 inside its specialization
machinery and is not reachable from the kernel source. Knowing a cliff exists and
being able to move it is the argument -- not a blanket codegen deficit.

**Fusion and occupancy trade against each other.** Holding a row on-chip is what
makes the single pass possible, and it is also what limits how many blocks fit
per SM. The CUDA kernel spends dynamic shared memory and returns
`cudaErrorInvalidValue` past the device's opt-in limit; Triton spends registers
and hits its own width ceiling. Both have a boundary. The hand-written version
makes it explicit and checkable instead of leaving it to a heuristic, which is
exactly what you want when a shape regresses in production.

So the defensible claim is narrow: for a bandwidth-bound fusion at a chosen
shape, hand-writing buys explicit control of traffic, access width, and the
occupancy tradeoff, and the cost is that you own every shape yourself. Reaching
for `torch.compile` first is usually right. This repo is the study of what the
remaining margin is made of.

## Analytical result

For 4,096 rows by 4,096 hidden elements in FP32:

| Path | Estimated global traffic |
| --- | ---: |
| Three-kernel baseline | 469,794,816 bytes |
| Shared-memory fused | 268,435,456 bytes |
| Saved | **201,359,360 bytes** |

That is a **42.86% traffic reduction** and a **1.75x bandwidth-bound speedup
ceiling** before launch overhead. These are source-derived traffic counts, not
claimed GPU timings. They are generated in
[`artifacts/analytical.json`](artifacts/analytical.json).

Hardware speedup depends on architecture, occupancy, row width, and shared
memory pressure. The benchmark prints the actual device, latency, speedup,
effective GB/s, and maximum correctness errors as JSON.

## Build and benchmark

Requirements: CUDA Toolkit 12.6+, CMake 3.24+, Python 3.10+, and an NVIDIA GPU
with compute capability 7.5 or newer.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/kernelforge_bench \
  --rows 4096 --hidden 4096 --iterations 200 \
  > artifacts/benchmark.json

PYTHONPATH=python python3 -m kernelforge.matrix \
  --binary build/kernelforge_bench \
  --output artifacts/benchmark_matrix.json
```

Or build with the pinned CUDA development image:

```bash
docker build -t kernelforge .
docker run --rm --gpus all kernelforge \
  --rows 4096 --hidden 4096 --iterations 200
```

The executable exits nonzero if the fused result exceeds its correctness
tolerance.

## Profile

[`scripts/profile.sh`](scripts/profile.sh) records a benchmark and invokes
Nsight Compute for:

- DRAM bytes read and written;
- percentage of peak sustained DRAM throughput;
- active warps;
- separate scalar or vector fused kernel names.

The script persists both the JSON benchmark matrix and an `.ncu-rep` capture
for independent inspection.

```bash
bash scripts/profile.sh
```

See [`docs/optimization.md`](docs/optimization.md) for the measurement plan and
the next decisions to make from profiler evidence.

## Hardware-independent checks

The Python side contains the CPU reference and memory-traffic model, so CI can
still validate semantics without pretending a hosted runner has a GPU.

```powershell
$env:PYTHONPATH = "python"
python -m kernelforge --rows 4096 --hidden 4096
python -m pip install -e ".[dev]"
pytest
```

The baselines need the `baseline` extra, which pins `torch` and `triton`
(`triton` installs on Linux only -- upstream ships manylinux wheels
exclusively). Versions are pinned rather than floated because a minor bump can
change Inductor's fusion decisions and therefore the numbers quoted here.

```bash
python -m pip install -e ".[dev,baseline]"

# Do every baseline compute the same formula? No GPU needed.
PYTHONPATH=python python -m kernelforge.compare --correctness-only

# Which global-access width does Triton pick? Lowers to PTX, never executes.
PYTHONPATH=python python scripts/triton_codegen_check.py --arch 75 --check-rule

# Latency and bandwidth for every implementation. GPU required.
PYTHONPATH=python python -m kernelforge.compare \
  --binary build/kernelforge_bench \
  --output artifacts/baseline_comparison.json
```

`--correctness-only` exits nonzero if any available baseline disagrees with the
float64 oracle. Correctness gates timing: a baseline that fails the tolerance has
its latency withheld from the report rather than published alongside a wrong
answer.

## Repository map

```text
include/kernelforge/rmsnorm.cuh   public CUDA launch API
src/rmsnorm.cu                    baseline and fused kernels (the subject)
src/benchmark.cu                  CPU oracle, CUDA-event timing, JSON report
python/kernelforge/model.py       reference and traffic model
python/kernelforge/baselines.py   torch / torch.compile / Triton baselines
python/kernelforge/triton_rmsnorm.py  Triton port of the fused kernel
python/kernelforge/compare.py     correctness gate + comparison harness
scripts/profile.sh                Nsight Compute capture
scripts/triton_codegen_check.py   PTX lowering check, no GPU needed
artifacts/                        analytical and measured evidence
tests/                            hardware-independent correctness checks
```

## Research context

The project direction is informed by NVIDIA
[CUTLASS](https://github.com/NVIDIA/cutlass), where data movement and reusable
kernel structure are explicit, and the
[Triton Puzzles](https://github.com/gpu-mode/Triton-Puzzles), which emphasize
mapping tensor programs onto GPU execution. This implementation is standalone
CUDA C++ so the memory hierarchy, reductions, dispatch, and launch behavior
remain visible.
