#include "benchmark.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

static void print_help(const char* prog) {
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  --device <int>           CUDA device index (default: 0)\n");
    printf("  --list-devices           List available CUDA devices and exit\n");
    printf("  --dtype <float32|float64|complex64|complex128>  Data type (required)\n");
    printf("  --dim <1|2|3>            Transform dimensionality (required)\n");
    printf("  --size <int>             Set all axes to this size (convenience)\n");
    printf("  --nx <int>               Size along x-axis\n");
    printf("  --ny <int>               Size along y-axis (2D/3D)\n");
    printf("  --nz <int>               Size along z-axis (3D)\n");
    printf("  --mode <kernel|e2e>      Timing mode (default: kernel)\n");
    printf("  --warmup <int>           Warmup iterations (default: 5)\n");
    printf("  --iters <int>            Timed iterations (default: 20)\n");
    printf("  --format <human|json>    Output format (default: human)\n");
    printf("  --help                   Show this help\n");
    printf("\n");
    printf("Either --size or --nx must be provided.\n");
    printf("--size sets the default for all axes; --nx/--ny/--nz override individually.\n");
}

static void list_devices() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("No CUDA devices found.\n");
        return;
    }
    printf("Available CUDA devices:\n\n");
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("  Device %d: %s\n", i, prop.name);
        printf("    Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("    Global memory:      %.0f MiB\n",
               static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0));
        printf("    SM count:           %d\n", prop.multiProcessorCount);
        printf("    Clock rate:         %d MHz\n", prop.clockRate / 1000);
        printf("\n");
    }
}

static const char* dtype_name(DataType dtype) {
    switch (dtype) {
    case DataType::Float32:    return "float32 (R2C)";
    case DataType::Float64:    return "float64 (D2Z)";
    case DataType::Complex64:  return "complex64 (C2C)";
    case DataType::Complex128: return "complex128 (Z2Z)";
    }
    return "unknown";
}

static const char* mode_name(TimingMode mode) {
    return mode == TimingMode::Kernel ? "kernel" : "e2e";
}

static const char* dtype_short_name(DataType dtype) {
    switch (dtype) {
    case DataType::Float32:    return "float32";
    case DataType::Float64:    return "float64";
    case DataType::Complex64:  return "complex64";
    case DataType::Complex128: return "complex128";
    }
    return "unknown";
}

static const char* transform_name(DataType dtype) {
    switch (dtype) {
    case DataType::Float32:    return "R2C";
    case DataType::Float64:    return "D2Z";
    case DataType::Complex64:  return "C2C";
    case DataType::Complex128: return "Z2Z";
    }
    return "unknown";
}

static void print_json(const BenchConfig& config, const BenchResult& result,
                        const char* gpu_name, int device_id,
                        int cuda_major, int cuda_minor) {
    printf("{\n");
    printf("  \"gpu\": \"%s\",\n", gpu_name);
    printf("  \"device_id\": %d,\n", device_id);
    printf("  \"cuda_version\": \"%d.%d\",\n", cuda_major, cuda_minor);
    printf("  \"dtype\": \"%s\",\n", dtype_short_name(config.dtype));
    printf("  \"transform\": \"%s\",\n", transform_name(config.dtype));
    printf("  \"dimensions\": %d,\n", config.dim);
    printf("  \"nx\": %d,\n", config.nx);
    printf("  \"ny\": %d,\n", config.ny);
    printf("  \"nz\": %d,\n", config.nz);
    printf("  \"mode\": \"%s\",\n", mode_name(config.mode));
    printf("  \"warmup\": %d,\n", config.warmup);
    printf("  \"iterations\": %d,\n", config.iters);
    printf("  \"timings_ms\": [");
    for (size_t i = 0; i < result.timings_ms.size(); i++) {
        if (i > 0) printf(", ");
        printf("%.4f", result.timings_ms[i]);
    }
    printf("],\n");
    printf("  \"stats\": {\n");
    printf("    \"min_ms\": %.4f,\n", result.min_ms);
    printf("    \"mean_ms\": %.4f,\n", result.mean_ms);
    printf("    \"median_ms\": %.4f,\n", result.median_ms);
    printf("    \"stddev_ms\": %.4f\n", result.stddev_ms);
    printf("  }\n");
    printf("}\n");
}

int main(int argc, char* argv[]) {
    // Defaults
    DataType dtype = DataType::Float32;
    bool dtype_set = false;
    int dim = 0;
    int size = -1;
    int nx = -1, ny = -1, nz = -1;
    TimingMode mode = TimingMode::Kernel;
    OutputFormat format = OutputFormat::Human;
    int warmup = 5;
    int iters = 20;
    int device_id = -1;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        auto next_arg = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires a value\n", flag);
                exit(1);
            }
            return argv[++i];
        };

        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_help(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--list-devices") == 0) {
            list_devices();
            return 0;
        } else if (strcmp(argv[i], "--device") == 0) {
            device_id = atoi(next_arg("--device"));
        } else if (strcmp(argv[i], "--dtype") == 0) {
            const char* v = next_arg("--dtype");
            if (strcmp(v, "float32") == 0)         dtype = DataType::Float32;
            else if (strcmp(v, "float64") == 0)     dtype = DataType::Float64;
            else if (strcmp(v, "complex64") == 0)   dtype = DataType::Complex64;
            else if (strcmp(v, "complex128") == 0)   dtype = DataType::Complex128;
            else {
                fprintf(stderr, "Error: unknown dtype '%s' (expected float32|float64|complex64|complex128)\n", v);
                return 1;
            }
            dtype_set = true;
        } else if (strcmp(argv[i], "--dim") == 0) {
            dim = atoi(next_arg("--dim"));
        } else if (strcmp(argv[i], "--size") == 0) {
            size = atoi(next_arg("--size"));
        } else if (strcmp(argv[i], "--nx") == 0) {
            nx = atoi(next_arg("--nx"));
        } else if (strcmp(argv[i], "--ny") == 0) {
            ny = atoi(next_arg("--ny"));
        } else if (strcmp(argv[i], "--nz") == 0) {
            nz = atoi(next_arg("--nz"));
        } else if (strcmp(argv[i], "--mode") == 0) {
            const char* v = next_arg("--mode");
            if (strcmp(v, "kernel") == 0)      mode = TimingMode::Kernel;
            else if (strcmp(v, "e2e") == 0)    mode = TimingMode::E2E;
            else {
                fprintf(stderr, "Error: unknown mode '%s' (expected kernel|e2e)\n", v);
                return 1;
            }
        } else if (strcmp(argv[i], "--warmup") == 0) {
            warmup = atoi(next_arg("--warmup"));
        } else if (strcmp(argv[i], "--iters") == 0) {
            iters = atoi(next_arg("--iters"));
        } else if (strcmp(argv[i], "--format") == 0) {
            const char* v = next_arg("--format");
            if (strcmp(v, "human") == 0)       format = OutputFormat::Human;
            else if (strcmp(v, "json") == 0)   format = OutputFormat::Json;
            else {
                fprintf(stderr, "Error: unknown format '%s' (expected human|json)\n", v);
                return 1;
            }
        } else {
            fprintf(stderr, "Error: unknown option '%s'\n", argv[i]);
            return 1;
        }
    }

    // Validation
    if (!dtype_set) {
        fprintf(stderr, "Error: --dtype is required\n");
        return 1;
    }
    if (dim < 1 || dim > 3) {
        fprintf(stderr, "Error: --dim must be 1, 2, or 3\n");
        return 1;
    }
    if (size < 0 && nx < 0) {
        fprintf(stderr, "Error: either --size or --nx must be provided\n");
        return 1;
    }

    // Resolve sizes: --size provides defaults, --nx/ny/nz override
    int default_n = (size > 0) ? size : nx;
    if (nx < 0) nx = default_n;
    if (ny < 0) ny = default_n;
    if (nz < 0) nz = default_n;

    if (nx <= 0) {
        fprintf(stderr, "Error: nx must be positive\n");
        return 1;
    }
    if (dim >= 2 && ny <= 0) {
        fprintf(stderr, "Error: ny must be positive for %dD transforms\n", dim);
        return 1;
    }
    if (dim >= 3 && nz <= 0) {
        fprintf(stderr, "Error: nz must be positive for 3D transforms\n");
        return 1;
    }
    if (warmup < 0) {
        fprintf(stderr, "Error: --warmup must be non-negative\n");
        return 1;
    }
    if (iters < 1) {
        fprintf(stderr, "Error: --iters must be at least 1\n");
        return 1;
    }

    // Select and validate CUDA device
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        fprintf(stderr, "Error: no CUDA devices found\n");
        return 1;
    }
    if (device_id < 0) device_id = 0;
    if (device_id >= device_count) {
        fprintf(stderr, "Error: device %d not found (available: 0-%d)\n",
                device_id, device_count - 1);
        return 1;
    }
    cudaSetDevice(device_id);

    // Pre-initialize CUDA context
    cudaFree(0);

    // Print GPU info
    cudaDeviceProp prop;
    int device = device_id;
    cudaGetDeviceProperties(&prop, device);

    int cuda_major, cuda_minor;
    cudaRuntimeGetVersion(&cuda_major);
    // cudaRuntimeGetVersion returns version as major*1000 + minor*10
    cuda_minor = (cuda_major % 100) / 10;
    cuda_major = cuda_major / 1000;

    if (format == OutputFormat::Human) {
        printf("=== cuFFT Benchmark ===\n");
        printf("Device:     %d - %s\n", device, prop.name);
        printf("CUDA:       %d.%d\n", cuda_major, cuda_minor);
        printf("Transform:  %s\n", dtype_name(dtype));
        printf("Dimensions: %dD", dim);
        if (dim == 1)      printf(" [%d]\n", nx);
        else if (dim == 2) printf(" [%d x %d]\n", nx, ny);
        else               printf(" [%d x %d x %d]\n", nx, ny, nz);
        printf("Mode:       %s\n", mode_name(mode));
        printf("Warmup:     %d\n", warmup);
        printf("Iterations: %d\n", iters);
        printf("\n");
    }

    BenchConfig config;
    config.dtype = dtype;
    config.dim = dim;
    config.nx = nx;
    config.ny = ny;
    config.nz = nz;
    config.mode = mode;
    config.format = format;
    config.warmup = warmup;
    config.iters = iters;

    try {
        BenchResult result = run_benchmark(config);

        if (format == OutputFormat::Json) {
            print_json(config, result, prop.name, device, cuda_major, cuda_minor);
        } else {
            printf("--- Results (ms) ---\n");
            printf("  Min:    %10.4f\n", result.min_ms);
            printf("  Mean:   %10.4f\n", result.mean_ms);
            printf("  Median: %10.4f\n", result.median_ms);
            printf("  Stddev: %10.4f\n", result.stddev_ms);
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
}
