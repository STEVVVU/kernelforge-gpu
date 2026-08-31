# Optimization Notes

## Hypothesis

Residual RMSNorm is memory-bound for common transformer widths. The unfused
baseline moves the intermediate residual sum through global memory twice and
pays three launch costs. Holding one row in shared memory should reduce global
traffic from approximately 28 to 16 bytes per element.

## Correctness gate

Every benchmark run computes a CPU result with double-precision accumulation.
Both CUDA paths are copied back and compared. The fused benchmark aborts before
timing results are reported when error exceeds the configured mixed absolute
and relative tolerance.

## Measurement protocol

1. Lock application clocks when the environment permits it.
2. Run the configured warmup before creating the timed sample.
3. Measure 200 launches with CUDA events on the same stream.
4. Repeat for hidden sizes 768, 1,024, 4,096, 8,192, and a non-vectorized
   control such as 4,095.
5. Save device name, toolkit version, clocks, latency, and correctness output.
6. Use Nsight Compute after timing, never around the latency measurement.

## Tesla T4 evidence

The committed benchmark matrix was captured on 2026-07-16 with a Tesla T4
(compute capability 7.5), CUDA driver/runtime 13.2, 25 warmup launches, and
200 timed iterations per shape.

| Rows x hidden | Dispatch | Unfused (ms) | Fused (ms) | Speedup |
| --- | --- | ---: | ---: | ---: |
| 4,096 x 768 | `float4` | 0.315914 | 0.153415 | 2.059213x |
| 4,096 x 1,024 | `float4` | 0.415816 | 0.208033 | 1.998799x |
| 4,096 x 4,095 | scalar | 1.604311 | 1.019351 | 1.573856x |
| 4,096 x 4,096 | `float4` | 1.590025 | 0.800767 | 1.985626x |
| 1,024 x 8,192 | `float4` | 0.803615 | 0.445302 | 1.804652x |

The geometric-mean speedup is 1.875622x, with a 1e-6 maximum absolute error
across both CUDA implementations. The 4,095-wide scalar control is 21.4%
slower than the adjacent 4,096-wide vector path despite processing one fewer
element, supporting the aligned-load dispatch decision.

The machine-readable matrix is
[`artifacts/benchmark_matrix.json`](../artifacts/benchmark_matrix.json).
[`artifacts/kernelforge-fused.ncu-rep`](../artifacts/kernelforge-fused.ncu-rep)
contains the post-timing Nsight Compute capture. Profiler replay changes wall
time, so latency claims come only from the unprofiled CUDA-event matrix.

This is evidence from one GPU model and one capture. Architecture-wide claims
would require repeated runs on additional NVIDIA generations.

## Profiler decisions

| Observation | Next experiment |
| --- | --- |
| DRAM throughput near peak, low stalls | Compare traffic directly and keep fusion |
| Occupancy limited by shared memory | Process a row in two tiles or recompute residual sums |
| Long scoreboard stalls | Increase independent vector loads per thread |
| Reduction dominates small widths | Use one warp per row and multiple rows per block |
| `float4` shows no gain | Check alignment, transaction count, and instruction mix |
| Large widths reject shared-memory launch | Add a two-pass fused fallback |

## Access width vs the Triton baseline

The `float4` fork in `launch_fused_residual_rmsnorm` triggers at
`hidden % 4 == 0`. The Triton baseline in `python/kernelforge/triton_rmsnorm.py`
also reaches 16-byte accesses, but it decides on a coarser rule: Triton's only
divisibility tier is 16 (`BaseBackend.parse_attr` emits exactly
`tt.divisibility = 16`), so a width that is a multiple of 4 but not of 16 carries
no usable hint and the row falls back to `ld.global.b32`.

Measured by lowering to PTX for sm_75 with the hints a real launch would
specialize on (`scripts/triton_codegen_check.py`, counts in
`artifacts/triton_ptx_vectorization.json`): at `hidden = 4092` the CUDA kernel
takes its `float4` path while Triton emits 64 scalar global accesses. At 768,
1,024, 4,096, and 8,192 -- all multiples of 16 -- both vectorize. At 4,095
neither does.

The mask is not the cause, which is worth recording because it is the intuitive
guess. An A/B control compiles a masked kernel with and without a divisibility
hint on its bound: hinted, it emits `ld.global.v4.b32`. Predication costs nothing
in access width; only unprovable divisibility does.

Two consequences for the experiments above:

- The `float4 shows no gain` row now has a companion question: whether the extra
  transactions at a `%4`-but-not-`%16` width cost measurable latency, or whether
  the kernel is bound elsewhere. An instruction count is not a runtime and this
  is unmeasured -- it needs the GPU comparison job.
- If a two-pass fallback is added for large rows, its tail handling should keep
  the `%4` fork rather than adopting a single masked path, since that is the
  degree of freedom Triton does not expose.

## Known boundary

The one-row shared-memory design is intentionally optimized for hidden widths
that fit the device's opt-in dynamic shared-memory limit. The launcher returns
`cudaErrorInvalidValue` outside that boundary rather than silently selecting a
misleading path. A production extension would dispatch a tiled two-pass kernel
for larger rows and benchmark the crossover.
