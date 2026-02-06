# cuFFT Benchmark

A command-line utility for benchmarking [cuFFT](https://docs.nvidia.com/cuda/cufft/) performance across different transform types, dimensionalities, and problem sizes.

## Building

Requires CMake 3.18+, a C++17-capable host compiler, and a CUDA toolkit installation (tested with CUDA 12.4).

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc
make -j$(nproc)
```

The default GPU architecture target is sm_80 (A100). To build for a different architecture, pass it at configure time:

```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90   # H100
cmake .. -DCMAKE_CUDA_ARCHITECTURES=89   # RTX 4090
```

## Usage

```
cufft-bench --dtype <type> --dim <1|2|3> --size <N> [options]
```

### Required options

| Option | Description |
|--------|-------------|
| `--dtype <type>` | Data type for the transform (see [Data types](#data-types)) |
| `--dim <1\|2\|3>` | Dimensionality of the transform |
| `--size <N>` or `--nx <N>` | Problem size (at least one must be provided) |

### Optional flags

| Option | Default | Description |
|--------|---------|-------------|
| `--size <N>` | — | Convenience flag: sets all axes to N |
| `--nx <N>` | — | Size along the x-axis |
| `--ny <N>` | same as `--size` or `--nx` | Size along the y-axis (2D and 3D only) |
| `--nz <N>` | same as `--size` or `--nx` | Size along the z-axis (3D only) |
| `--mode <kernel\|e2e>` | `kernel` | Timing mode (see [Timing modes](#timing-modes)) |
| `--warmup <N>` | `5` | Number of untimed warmup iterations |
| `--iters <N>` | `20` | Number of timed iterations |
| `--help` | — | Print usage information and exit |

### Specifying problem sizes

There are two ways to set the transform dimensions:

**Symmetric (all axes the same):** Use `--size N`. For a 2D transform this gives an N x N problem; for 3D, N x N x N.

```bash
# 2D 512x512
cufft-bench --dtype float32 --dim 2 --size 512

# 3D 128x128x128
cufft-bench --dtype complex64 --dim 3 --size 128
```

**Asymmetric (per-axis control):** Use `--nx`, `--ny`, and `--nz` to set each axis independently. If `--size` is also provided, it serves as the default for any axis not explicitly set.

```bash
# 2D 1024x512
cufft-bench --dtype float64 --dim 2 --nx 1024 --ny 512

# 3D 256x256x64 (--size sets x and y, --nz overrides z)
cufft-bench --dtype complex128 --dim 3 --size 256 --nz 64
```

Only the axes relevant to `--dim` are used: 1D uses only nx, 2D uses nx and ny, 3D uses all three.

## Data types

The `--dtype` flag selects both the input precision and the type of cuFFT transform:

| `--dtype` | Precision | Transform | cuFFT type | cuFFT function |
|-----------|-----------|-----------|------------|----------------|
| `float32` | 32-bit real | Real-to-Complex | `CUFFT_R2C` | `cufftExecR2C` |
| `float64` | 64-bit real | Real-to-Complex | `CUFFT_D2Z` | `cufftExecD2Z` |
| `complex64` | 32-bit complex | Complex-to-Complex | `CUFFT_C2C` | `cufftExecC2C` |
| `complex128` | 64-bit complex | Complex-to-Complex | `CUFFT_Z2Z` | `cufftExecZ2Z` |

For real-to-complex (R2C / D2Z) transforms, the output exploits Hermitian symmetry and has a reduced last dimension of `(N/2 + 1)` complex elements, following cuFFT conventions.

## Timing modes

The `--mode` flag controls what is included in the timing measurement. In both modes, the cuFFT plan is created once before timing begins (plan creation involves autotuning and is considered a one-time setup cost).

### `kernel` (default)

Times **only** the `cufftExec*` call. Memory allocation and host-device transfers happen once before the timed loop. This isolates the raw FFT compute performance.

```
Setup (not timed):     cudaMalloc -> H2D copy -> plan creation
Warmup (not timed):    exec_fft x warmup
Timed loop:            [CUDA event start] exec_fft [CUDA event stop]  x iters
Teardown (not timed):  cudaFree
```

### `e2e` (end-to-end)

Times the full data lifecycle per iteration: `cudaMalloc` -> host-to-device copy -> FFT execution -> device-to-host copy -> `cudaFree`. This captures the overhead of memory management and PCIe transfers.

```
Setup (not timed):     plan creation
Warmup (not timed):    full cycle x warmup
Timed loop:            [CUDA event start] cudaMalloc -> H2D -> exec -> D2H -> cudaFree [CUDA event stop]  x iters
```

Use `kernel` mode to measure pure compute throughput. Use `e2e` mode to understand realistic end-to-end latency when data must be moved to and from the GPU each time.

## Output

The tool prints a header with GPU and configuration details, followed by timing statistics:

```
=== cuFFT Benchmark ===
GPU:        NVIDIA A100-SXM4-40GB
CUDA:       12.4
Transform:  float32 (R2C)
Dimensions: 2D [512 x 512]
Mode:       kernel
Warmup:     5
Iterations: 20

--- Results (ms) ---
  Min:        0.0195
  Mean:       0.0198
  Median:     0.0195
  Stddev:     0.0004
```

| Statistic | Description |
|-----------|-------------|
| Min | Fastest iteration (least affected by system noise) |
| Mean | Arithmetic mean of all timed iterations |
| Median | Middle value when sorted (robust to outliers) |
| Stddev | Sample standard deviation (N-1 denominator) |

All times are reported in milliseconds. Each iteration is timed individually using CUDA events, which measure GPU-side elapsed time with sub-microsecond resolution.

## Examples

```bash
# Quick 1D benchmark, 1M points, single precision
cufft-bench --dtype float32 --dim 1 --size 1048576

# 2D double-precision R2C, 4096x4096, minimal warmup
cufft-bench --dtype float64 --dim 2 --size 4096 --warmup 2 --iters 50

# 3D complex single-precision, small cube, end-to-end timing
cufft-bench --dtype complex64 --dim 3 --size 64 --mode e2e

# 2D complex double-precision, asymmetric, many iterations for stable stats
cufft-bench --dtype complex128 --dim 2 --nx 2048 --ny 1024 --iters 100

# 1D large transform with no warmup (measures cold-start kernel performance)
cufft-bench --dtype complex64 --dim 1 --size 16777216 --warmup 0 --iters 5
```
