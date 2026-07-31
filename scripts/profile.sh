#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=python python3 -m kernelforge.matrix \
  --binary build/kernelforge_bench \
  --output artifacts/benchmark_matrix.json

ncu --kernel-name-base demangled \
  --kernel-name "regex:fused_.*_kernel" \
  --launch-count 1 \
  --metrics \
dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active \
  --export artifacts/kernelforge-fused \
  --force-overwrite \
  build/kernelforge_bench --rows 4096 --hidden 4096 --iterations 5
