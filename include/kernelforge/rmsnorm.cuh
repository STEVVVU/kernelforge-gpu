#pragma once

#include <cuda_runtime.h>

namespace kernelforge {

cudaError_t launch_unfused_residual_rmsnorm(
    const float* x,
    const float* residual,
    const float* weight,
    float* scratch,
    float* inverse_rms,
    float* output,
    int rows,
    int hidden,
    float epsilon,
    cudaStream_t stream = nullptr);

cudaError_t launch_fused_residual_rmsnorm(
    const float* x,
    const float* residual,
    const float* weight,
    float* output,
    int rows,
    int hidden,
    float epsilon,
    cudaStream_t stream = nullptr);

}  // namespace kernelforge

