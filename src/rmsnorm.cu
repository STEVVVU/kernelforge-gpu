#include "kernelforge/rmsnorm.cuh"

#include <cstdint>

namespace kernelforge {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_sum(float value) {
    __shared__ float warp_sums[32];
    __shared__ float total;
    const int lane = threadIdx.x % warpSize;
    const int warp = threadIdx.x / warpSize;
    value = warp_sum(value);
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();

    if (warp == 0) {
        const int warp_count = (blockDim.x + warpSize - 1) / warpSize;
        float aggregate = lane < warp_count ? warp_sums[lane] : 0.0f;
        aggregate = warp_sum(aggregate);
        if (lane == 0) {
            total = aggregate;
        }
    }
    __syncthreads();
    return total;
}

__global__ void add_residual_kernel(
    const float* x, const float* residual, float* scratch, int elements) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
         index += blockDim.x * gridDim.x) {
        scratch[index] = x[index] + residual[index];
    }
}

__global__ void inverse_rms_kernel(
    const float* scratch, float* inverse_rms, int hidden, float epsilon) {
    const int row = blockIdx.x;
    const int base = row * hidden;
    float sum = 0.0f;
    for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
        const float value = scratch[base + column];
        sum += value * value;
    }
    sum = block_sum(sum);
    if (threadIdx.x == 0) {
        inverse_rms[row] = rsqrtf(sum / static_cast<float>(hidden) + epsilon);
    }
}

__global__ void normalize_kernel(
    const float* scratch,
    const float* weight,
    const float* inverse_rms,
    float* output,
    int elements,
    int hidden) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
         index += blockDim.x * gridDim.x) {
        output[index] = scratch[index] * inverse_rms[index / hidden] * weight[index % hidden];
    }
}

__global__ void fused_scalar_kernel(
    const float* x,
    const float* residual,
    const float* weight,
    float* output,
    int hidden,
    float epsilon) {
    extern __shared__ float combined[];
    const int base = blockIdx.x * hidden;
    float sum = 0.0f;
    for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
        const float value = x[base + column] + residual[base + column];
        combined[column] = value;
        sum += value * value;
    }
    sum = block_sum(sum);
    const float inverse_rms = rsqrtf(sum / static_cast<float>(hidden) + epsilon);
    for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
        output[base + column] = combined[column] * inverse_rms * weight[column];
    }
}

__global__ void fused_vector_kernel(
    const float* x,
    const float* residual,
    const float* weight,
    float* output,
    int hidden,
    float epsilon) {
    extern __shared__ float combined[];
    const int row_base = blockIdx.x * hidden;
    const int packs = hidden / 4;
    const auto* x4 = reinterpret_cast<const float4*>(x + row_base);
    const auto* residual4 = reinterpret_cast<const float4*>(residual + row_base);
    auto* combined4 = reinterpret_cast<float4*>(combined);
    float sum = 0.0f;

    for (int pack = threadIdx.x; pack < packs; pack += blockDim.x) {
        const float4 left = x4[pack];
        const float4 right = residual4[pack];
        const float4 value{
            left.x + right.x,
            left.y + right.y,
            left.z + right.z,
            left.w + right.w,
        };
        combined4[pack] = value;
        sum += value.x * value.x + value.y * value.y + value.z * value.z + value.w * value.w;
    }
    sum = block_sum(sum);
    const float inverse_rms = rsqrtf(sum / static_cast<float>(hidden) + epsilon);

    const auto* weight4 = reinterpret_cast<const float4*>(weight);
    auto* output4 = reinterpret_cast<float4*>(output + row_base);
    for (int pack = threadIdx.x; pack < packs; pack += blockDim.x) {
        const float4 value = combined4[pack];
        const float4 scale = weight4[pack];
        output4[pack] = {
            value.x * inverse_rms * scale.x,
            value.y * inverse_rms * scale.y,
            value.z * inverse_rms * scale.z,
            value.w * inverse_rms * scale.w,
        };
    }
}

}  // namespace

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
    cudaStream_t stream) {
    const int elements = rows * hidden;
    const int blocks = (elements + kThreads - 1) / kThreads;
    add_residual_kernel<<<blocks, kThreads, 0, stream>>>(x, residual, scratch, elements);
    inverse_rms_kernel<<<rows, kThreads, 0, stream>>>(scratch, inverse_rms, hidden, epsilon);
    normalize_kernel<<<blocks, kThreads, 0, stream>>>(
        scratch, weight, inverse_rms, output, elements, hidden);
    return cudaGetLastError();
}

cudaError_t launch_fused_residual_rmsnorm(
    const float* x,
    const float* residual,
    const float* weight,
    float* output,
    int rows,
    int hidden,
    float epsilon,
    cudaStream_t stream) {
    const int shared_bytes = hidden * static_cast<int>(sizeof(float));
    int maximum_shared = 0;
    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return status;
    }
    status = cudaDeviceGetAttribute(
        &maximum_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status != cudaSuccess) {
        return status;
    }
    if (shared_bytes > maximum_shared) {
        return cudaErrorInvalidValue;
    }
    if (shared_bytes > 48 * 1024) {
        if (hidden % 4 == 0) {
            status = cudaFuncSetAttribute(
                fused_vector_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
        } else {
            status = cudaFuncSetAttribute(
                fused_scalar_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
        }
        if (status != cudaSuccess) {
            return status;
        }
    }
    if (hidden % 4 == 0) {
        fused_vector_kernel<<<rows, kThreads, shared_bytes, stream>>>(
            x, residual, weight, output, hidden, epsilon);
    } else {
        fused_scalar_kernel<<<rows, kThreads, shared_bytes, stream>>>(
            x, residual, weight, output, hidden, epsilon);
    }
    return cudaGetLastError();
}

}  // namespace kernelforge
