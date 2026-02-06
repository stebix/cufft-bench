#pragma once

#include <cstddef>
#include <vector>

enum class DataType { Float32, Float64, Complex64, Complex128 };
enum class TimingMode { Kernel, E2E };

struct BenchConfig {
    DataType dtype;
    int dim;
    int nx;
    int ny;
    int nz;
    TimingMode mode;
    int warmup;
    int iters;
};

struct BenchResult {
    std::vector<double> timings_ms;
    double min_ms;
    double mean_ms;
    double median_ms;
    double stddev_ms;
};

BenchResult run_benchmark(const BenchConfig& config);
