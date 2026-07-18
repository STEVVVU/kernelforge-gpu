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

## Resume highlights

- Implemented fused residual RMSNorm in CUDA C++ using `float4` vectorization,
  warp-shuffle reductions, dynamic shared memory, and scalar fallback dispatch.
- Reduced modeled global-memory traffic by **42.86%** and established a
  **1.75x bandwidth-bound speedup ceiling** versus a three-kernel FP32
  baseline.
- Built a CUDA-event benchmark with a double-accumulation CPU oracle,
  correctness gates, effective-bandwidth reporting, Docker packaging, and
  Nsight Compute profiling hooks.

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

Requirements: CUDA Toolkit 12.x, CMake 3.24+, and an NVIDIA GPU with compute
capability 7.5 or newer.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/kernelforge_bench \
  --rows 4096 --hidden 4096 --iterations 200 \
  > artifacts/benchmark.json
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
