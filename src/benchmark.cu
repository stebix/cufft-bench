#include "benchmark.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cufft.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            throw std::runtime_error(                                          \
                std::string("CUDA error at ") + __FILE__ + ":" +              \
                std::to_string(__LINE__) + ": " + cudaGetErrorString(err));    \
        }                                                                      \
    } while (0)

#define CUFFT_CHECK(call)                                                      \
    do {                                                                       \
        cufftResult err = (call);                                              \
        if (err != CUFFT_SUCCESS) {                                            \
            throw std::runtime_error(                                          \
                std::string("cuFFT error at ") + __FILE__ + ":" +             \
                std::to_string(__LINE__) + ": code " + std::to_string(err));   \
        }                                                                      \
    } while (0)

static void compute_stats(BenchResult& result) {
    const auto& t = result.timings_ms;
    int n = static_cast<int>(t.size());

    result.min_ms = *std::min_element(t.begin(), t.end());
    result.mean_ms = std::accumulate(t.begin(), t.end(), 0.0) / n;

    std::vector<double> sorted(t);
    std::sort(sorted.begin(), sorted.end());
    if (n % 2 == 0)
        result.median_ms = (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    else
        result.median_ms = sorted[n / 2];

    if (n <= 1) {
        result.stddev_ms = 0.0;
    } else {
        double sq_sum = 0.0;
        for (double v : t) {
            double d = v - result.mean_ms;
            sq_sum += d * d;
        }
        result.stddev_ms = std::sqrt(sq_sum / (n - 1));
    }
}

static bool is_real_transform(DataType dtype) {
    return dtype == DataType::Float32 || dtype == DataType::Float64;
}

static size_t element_size_input(DataType dtype) {
    switch (dtype) {
    case DataType::Float32:    return sizeof(float);
    case DataType::Float64:    return sizeof(double);
    case DataType::Complex64:  return 2 * sizeof(float);
    case DataType::Complex128: return 2 * sizeof(double);
    }
    return 0;
}

static size_t element_size_output(DataType dtype) {
    switch (dtype) {
    case DataType::Float32:    return 2 * sizeof(float);   // cufftComplex
    case DataType::Float64:    return 2 * sizeof(double);  // cufftDoubleComplex
    case DataType::Complex64:  return 2 * sizeof(float);
    case DataType::Complex128: return 2 * sizeof(double);
    }
    return 0;
}

static void compute_buffer_sizes(const BenchConfig& config,
                                 size_t& in_bytes, size_t& out_bytes) {
    size_t total_elems = 1;
    size_t out_elems = 1;

    int dims[3] = {config.nx, config.ny, config.nz};
    int ndim = config.dim;

    for (int i = 0; i < ndim; i++)
        total_elems *= dims[i];

    if (is_real_transform(config.dtype)) {
        // R2C: output last dimension is (n_last/2 + 1) complex elements
        out_elems = 1;
        for (int i = 0; i < ndim - 1; i++)
            out_elems *= dims[i];
        out_elems *= (dims[ndim - 1] / 2 + 1);
    } else {
        out_elems = total_elems;
    }

    in_bytes = total_elems * element_size_input(config.dtype);
    out_bytes = out_elems * element_size_output(config.dtype);
}

static cufftType get_cufft_type(DataType dtype) {
    switch (dtype) {
    case DataType::Float32:    return CUFFT_R2C;
    case DataType::Float64:    return CUFFT_D2Z;
    case DataType::Complex64:  return CUFFT_C2C;
    case DataType::Complex128: return CUFFT_Z2Z;
    }
    return CUFFT_R2C;
}

static cufftHandle create_plan(const BenchConfig& config) {
    cufftHandle plan;
    cufftType type = get_cufft_type(config.dtype);

    switch (config.dim) {
    case 1:
        CUFFT_CHECK(cufftPlan1d(&plan, config.nx, type, 1));
        break;
    case 2:
        CUFFT_CHECK(cufftPlan2d(&plan, config.nx, config.ny, type));
        break;
    case 3:
        CUFFT_CHECK(cufftPlan3d(&plan, config.nx, config.ny, config.nz, type));
        break;
    default:
        throw std::runtime_error("Invalid dimension: " + std::to_string(config.dim));
    }

    return plan;
}

static void exec_fft(cufftHandle plan, void* d_in, void* d_out, DataType dtype) {
    switch (dtype) {
    case DataType::Float32:
        CUFFT_CHECK(cufftExecR2C(plan,
            static_cast<cufftReal*>(d_in),
            static_cast<cufftComplex*>(d_out)));
        break;
    case DataType::Float64:
        CUFFT_CHECK(cufftExecD2Z(plan,
            static_cast<cufftDoubleReal*>(d_in),
            static_cast<cufftDoubleComplex*>(d_out)));
        break;
    case DataType::Complex64:
        CUFFT_CHECK(cufftExecC2C(plan,
            static_cast<cufftComplex*>(d_in),
            static_cast<cufftComplex*>(d_out),
            CUFFT_FORWARD));
        break;
    case DataType::Complex128:
        CUFFT_CHECK(cufftExecZ2Z(plan,
            static_cast<cufftDoubleComplex*>(d_in),
            static_cast<cufftDoubleComplex*>(d_out),
            CUFFT_FORWARD));
        break;
    }
}

template <typename T>
static void fill_random(T* buf, size_t count, std::mt19937& rng) {
    std::uniform_real_distribution<T> dist(-1.0, 1.0);
    for (size_t i = 0; i < count; i++)
        buf[i] = dist(rng);
}

BenchResult run_benchmark(const BenchConfig& config) {
    size_t in_bytes, out_bytes;
    compute_buffer_sizes(config, in_bytes, out_bytes);

    // Generate random input on host
    std::mt19937 rng(42);
    std::vector<char> h_input(in_bytes);

    bool is_double = (config.dtype == DataType::Float64 ||
                      config.dtype == DataType::Complex128);
    if (is_double) {
        fill_random(reinterpret_cast<double*>(h_input.data()),
                    in_bytes / sizeof(double), rng);
    } else {
        fill_random(reinterpret_cast<float*>(h_input.data()),
                    in_bytes / sizeof(float), rng);
    }

    std::vector<char> h_output(out_bytes);
    BenchResult result;

    if (config.mode == TimingMode::Kernel) {
        // Allocate once, copy once, time only the FFT exec
        void *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, in_bytes));
        CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
        CUDA_CHECK(cudaMemcpy(d_in, h_input.data(), in_bytes,
                               cudaMemcpyHostToDevice));

        cufftHandle plan = create_plan(config);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Warmup
        for (int i = 0; i < config.warmup; i++)
            exec_fft(plan, d_in, d_out, config.dtype);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed iterations
        for (int i = 0; i < config.iters; i++) {
            CUDA_CHECK(cudaEventRecord(start));
            exec_fft(plan, d_in, d_out, config.dtype);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            result.timings_ms.push_back(static_cast<double>(ms));
        }

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        cufftDestroy(plan);
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));

    } else {
        // E2E mode: plan created once, each iteration includes alloc/H2D/exec/D2H/free
        cufftHandle plan = create_plan(config);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Warmup
        for (int i = 0; i < config.warmup; i++) {
            void *d_in, *d_out;
            CUDA_CHECK(cudaMalloc(&d_in, in_bytes));
            CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
            CUDA_CHECK(cudaMemcpy(d_in, h_input.data(), in_bytes,
                                   cudaMemcpyHostToDevice));
            exec_fft(plan, d_in, d_out, config.dtype);
            CUDA_CHECK(cudaMemcpy(h_output.data(), d_out, out_bytes,
                                   cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_in));
            CUDA_CHECK(cudaFree(d_out));
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed iterations
        for (int i = 0; i < config.iters; i++) {
            CUDA_CHECK(cudaEventRecord(start));

            void *d_in, *d_out;
            CUDA_CHECK(cudaMalloc(&d_in, in_bytes));
            CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
            CUDA_CHECK(cudaMemcpy(d_in, h_input.data(), in_bytes,
                                   cudaMemcpyHostToDevice));
            exec_fft(plan, d_in, d_out, config.dtype);
            CUDA_CHECK(cudaMemcpy(h_output.data(), d_out, out_bytes,
                                   cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_in));
            CUDA_CHECK(cudaFree(d_out));

            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            result.timings_ms.push_back(static_cast<double>(ms));
        }

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        cufftDestroy(plan);
    }

    compute_stats(result);
    return result;
}
