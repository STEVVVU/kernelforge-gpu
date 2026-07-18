#!/usr/bin/env bash
set -euo pipefail

build/kernelforge_bench --rows 4096 --hidden 4096 --iterations 200 \
  > artifacts/benchmark.json

ncu --kernel-name-base demangled \
  --kernel-name "regex:fused_.*_kernel" \
  --metrics \
dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active \
  build/kernelforge_bench --rows 4096 --hidden 4096 --iterations 5

