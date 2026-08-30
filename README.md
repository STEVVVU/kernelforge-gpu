# GPU Kernel Optimizer

A CUDA C++ GPU kernel optimization project built around a transformer
primitive:

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

## Repository map

```text
include/kernelforge/rmsnorm.cuh  public CUDA launch API
src/rmsnorm.cu                  baseline and fused kernels
src/benchmark.cu                CPU oracle, CUDA-event timing, JSON report
python/kernelforge/             reference and traffic model
scripts/profile.sh              Nsight Compute capture
artifacts/                      analytical and measured evidence
tests/                          hardware-independent correctness checks
```

## Research context

The project direction is informed by NVIDIA
[CUTLASS](https://github.com/NVIDIA/cutlass), where data movement and reusable
kernel structure are explicit, and the
[Triton Puzzles](https://github.com/gpu-mode/Triton-Puzzles), which emphasize
mapping tensor programs onto GPU execution. This implementation is standalone
CUDA C++ so the memory hierarchy, reductions, dispatch, and launch behavior
remain visible.
