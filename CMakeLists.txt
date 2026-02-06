cmake_minimum_required(VERSION 3.18)
project(cufft-bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(cufft-bench
    src/main.cu
    src/benchmark.cu
)

target_include_directories(cufft-bench PRIVATE src)
target_link_libraries(cufft-bench PRIVATE CUDA::cufft)
