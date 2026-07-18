#include "kernelforge/rmsnorm.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(expression)                                                        \
    do {                                                                              \
        const cudaError_t error = (expression);                                        \
        if (error != cudaSuccess) {                                                    \
            throw std::runtime_error(std::string(cudaGetErrorName(error)) + ": " +     \
                                     cudaGetErrorString(error));                       \
        }                                                                             \
    } while (false)

namespace {

struct Options {
    int rows = 4096;
    int hidden = 4096;
    int iterations = 200;
    int warmup = 25;
    float epsilon = 1e-5f;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value for " + argument);
        }
        const std::string value = argv[++index];
        if (argument == "--rows") {
            options.rows = std::stoi(value);
        } else if (argument == "--hidden") {
            options.hidden = std::stoi(value);
        } else if (argument == "--iterations") {
            options.iterations = std::stoi(value);
        } else if (argument == "--warmup") {
            options.warmup = std::stoi(value);
        } else {
            throw std::invalid_argument("unknown option " + argument);
        }
    }
    if (options.rows <= 0 || options.hidden <= 0 || options.iterations <= 0) {
        throw std::invalid_argument("rows, hidden, and iterations must be positive");
    }
    return options;
}

std::vector<float> cpu_reference(
    const std::vector<float>& x,
    const std::vector<float>& residual,
    const std::vector<float>& weight,
    int rows,
    int hidden,
    float epsilon) {
    std::vector<float> output(x.size());
    for (int row = 0; row < rows; ++row) {
        double square_sum = 0.0;
        const int base = row * hidden;
        for (int column = 0; column < hidden; ++column) {
            const double value = static_cast<double>(x[base + column]) + residual[base + column];
            square_sum += value * value;
        }
        const double inverse_rms = 1.0 / std::sqrt(square_sum / hidden + epsilon);
        for (int column = 0; column < hidden; ++column) {
            output[base + column] =
                (x[base + column] + residual[base + column]) *
                static_cast<float>(inverse_rms) * weight[column];
        }
    }
    return output;
}

template <typename Launcher>
float time_milliseconds(Launcher&& launch, int warmup, int iterations) {
    for (int iteration = 0; iteration < warmup; ++iteration) {
        CUDA_CHECK(launch());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int iteration = 0; iteration < iterations; ++iteration) {
        CUDA_CHECK(launch());
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed / iterations;
}

std::pair<float, float> errors(
    const std::vector<float>& actual, const std::vector<float>& expected) {
    float maximum_absolute = 0.0f;
    float maximum_relative = 0.0f;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float absolute = std::abs(actual[index] - expected[index]);
        const float relative = absolute / std::max(std::abs(expected[index]), 1e-6f);
        maximum_absolute = std::max(maximum_absolute, absolute);
        maximum_relative = std::max(maximum_relative, relative);
    }
    return {maximum_absolute, maximum_relative};
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::size_t elements =
            static_cast<std::size_t>(options.rows) * options.hidden;
        const std::size_t tensor_bytes = elements * sizeof(float);

        std::mt19937 generator(12345);
        std::normal_distribution<float> normal(0.0f, 0.5f);
        std::uniform_real_distribution<float> scales(0.8f, 1.2f);
        std::vector<float> host_x(elements);
        std::vector<float> host_residual(elements);
        std::vector<float> host_weight(options.hidden);
        std::generate(host_x.begin(), host_x.end(), [&] { return normal(generator); });
        std::generate(
            host_residual.begin(), host_residual.end(), [&] { return normal(generator); });
        std::generate(host_weight.begin(), host_weight.end(), [&] { return scales(generator); });
        const std::vector<float> reference = cpu_reference(
            host_x,
            host_residual,
            host_weight,
            options.rows,
            options.hidden,
            options.epsilon);

        float *x = nullptr, *residual = nullptr, *weight = nullptr;
        float *scratch = nullptr, *inverse_rms = nullptr, *unfused_output = nullptr;
        float* fused_output = nullptr;
        CUDA_CHECK(cudaMalloc(&x, tensor_bytes));
        CUDA_CHECK(cudaMalloc(&residual, tensor_bytes));
        CUDA_CHECK(cudaMalloc(&weight, options.hidden * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&scratch, tensor_bytes));
        CUDA_CHECK(cudaMalloc(&inverse_rms, options.rows * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&unfused_output, tensor_bytes));
        CUDA_CHECK(cudaMalloc(&fused_output, tensor_bytes));
        CUDA_CHECK(cudaMemcpy(x, host_x.data(), tensor_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(residual, host_residual.data(), tensor_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            weight, host_weight.data(), options.hidden * sizeof(float), cudaMemcpyHostToDevice));

        const auto launch_unfused = [&] {
            return kernelforge::launch_unfused_residual_rmsnorm(
                x,
                residual,
                weight,
                scratch,
                inverse_rms,
                unfused_output,
                options.rows,
                options.hidden,
                options.epsilon);
        };
        const auto launch_fused = [&] {
            return kernelforge::launch_fused_residual_rmsnorm(
                x,
                residual,
                weight,
                fused_output,
                options.rows,
                options.hidden,
                options.epsilon);
        };

        CUDA_CHECK(launch_unfused());
        CUDA_CHECK(launch_fused());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> host_unfused(elements);
        std::vector<float> host_fused(elements);
        CUDA_CHECK(cudaMemcpy(
            host_unfused.data(), unfused_output, tensor_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(
            cudaMemcpy(host_fused.data(), fused_output, tensor_bytes, cudaMemcpyDeviceToHost));
        const auto [unfused_absolute, unfused_relative] = errors(host_unfused, reference);
        const auto [fused_absolute, fused_relative] = errors(host_fused, reference);
        if (unfused_absolute > 2e-5f && unfused_relative > 2e-4f) {
            throw std::runtime_error("unfused correctness tolerance exceeded");
        }
        if (fused_absolute > 2e-5f && fused_relative > 2e-4f) {
            throw std::runtime_error("fused correctness tolerance exceeded");
        }

        const float unfused_ms =
            time_milliseconds(launch_unfused, options.warmup, options.iterations);
        const float fused_ms = time_milliseconds(launch_fused, options.warmup, options.iterations);
        const double unfused_bytes =
            28.0 * elements + 2.0 * options.rows * sizeof(float);
        const double fused_bytes = 16.0 * elements;
        const double unfused_gbps = unfused_bytes / unfused_ms / 1e6;
        const double fused_gbps = fused_bytes / fused_ms / 1e6;

        cudaDeviceProp properties{};
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        std::string device_name = properties.name;
        std::replace(device_name.begin(), device_name.end(), '"', '\'');
        std::cout << std::fixed << std::setprecision(6)
                  << "{\n"
                  << "  \"device\": \"" << device_name << "\",\n"
                  << "  \"rows\": " << options.rows << ",\n"
                  << "  \"hidden\": " << options.hidden << ",\n"
                  << "  \"iterations\": " << options.iterations << ",\n"
                  << "  \"unfused_ms\": " << unfused_ms << ",\n"
                  << "  \"fused_ms\": " << fused_ms << ",\n"
                  << "  \"speedup\": " << unfused_ms / fused_ms << ",\n"
                  << "  \"unfused_effective_gbps\": " << unfused_gbps << ",\n"
                  << "  \"fused_effective_gbps\": " << fused_gbps << ",\n"
                  << "  \"fused_max_absolute_error\": " << fused_absolute << ",\n"
                  << "  \"fused_max_relative_error\": " << fused_relative << ",\n"
                  << "  \"unfused_max_absolute_error\": " << unfused_absolute << ",\n"
                  << "  \"unfused_max_relative_error\": " << unfused_relative << "\n"
                  << "}\n";

        CUDA_CHECK(cudaFree(x));
        CUDA_CHECK(cudaFree(residual));
        CUDA_CHECK(cudaFree(weight));
        CUDA_CHECK(cudaFree(scratch));
        CUDA_CHECK(cudaFree(inverse_rms));
        CUDA_CHECK(cudaFree(unfused_output));
        CUDA_CHECK(cudaFree(fused_output));
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "kernelforge_bench: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
