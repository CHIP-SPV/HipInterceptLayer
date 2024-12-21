#pragma once
#include <hip/hip_runtime.h>
#include <vector>

namespace test_kernels {

// GPU Kernels
__global__ void vectorAdd(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c, int n) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void scalarKernel(float* __restrict__ out, int scalar1, float scalar2) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    out[i] = scalar1 * scalar2;
}

__global__ void simpleKernel(float* __restrict__ data) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    data[i] *= 2.0f;
}

__global__ void simpleKernelWithN(float* __restrict__ data, int n) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        data[i] *= 2.0f;
    }
}

__global__ void vectorKernel(float4* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = make_float4(
        data[idx].x * 2.0f,
        data[idx].y * 2.0f,
        data[idx].z * 2.0f,
        data[idx].w * 2.0f
    );
}

// String versions of kernels for testing code generation
namespace kernel_strings {

constexpr const char* vector_add = R"(
    __global__ void vectorAdd(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c, int n) {
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < n) {
            c[i] = a[i] + b[i];
        }
    }
)";

constexpr const char* scalar_kernel = R"(
    __global__ void scalarKernel(float* __restrict__ out, int scalar1, float scalar2) {
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;
        out[i] = scalar1 * scalar2;
    }
)";

constexpr const char* simple_kernel = R"(
    __global__ void simpleKernel(float* __restrict__ data) {
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;
        data[i] *= 2.0f;
    }
)";

constexpr const char* simple_kernel_with_n = R"(
    __global__ void simpleKernel(float* __restrict__ data, int n) {
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < n) {
            data[i] *= 2.0f;
        }
    }
)";

constexpr const char* invalid_kernel = R"(
    __global__ void invalidKernel(float* __restrict__ data) {
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;
        undefined_function();  // This should cause a compilation error
    }
)";

} // namespace kernel_strings

// CPU Reference Functions
inline void vectorAdd_cpu(const std::vector<float>& a, const std::vector<float>& b, 
                         std::vector<float>& c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

inline void scalarKernel_cpu(std::vector<float>& out, int scalar1, float scalar2, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = scalar1 * scalar2;
    }
}

inline void simpleKernel_cpu(std::vector<float>& data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] *= 2.0f;
    }
}

inline void simpleKernelWithN_cpu(std::vector<float>& data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] *= 2.0f;
    }
}

inline void vectorKernel_cpu(std::vector<float4>& data, int N) {
    for (int i = 0; i < N; i++) {
        data[i].x *= 2.0f;
        data[i].y *= 2.0f;
        data[i].z *= 2.0f;
        data[i].w *= 2.0f;
    }
}

// Verification Functions
inline bool verify_vectorAdd(const std::vector<float>& gpu_result, 
                           const std::vector<float>& a,
                           const std::vector<float>& b, 
                           int n) {
    std::vector<float> cpu_result(n);
    vectorAdd_cpu(a, b, cpu_result, n);
    
    for (int i = 0; i < n; i++) {
        if (gpu_result[i] != cpu_result[i]) return false;
    }
    return true;
}

inline bool verify_scalarKernel(const std::vector<float>& gpu_result,
                               int scalar1, float scalar2, 
                               int n) {
    std::vector<float> cpu_result(n);
    scalarKernel_cpu(cpu_result, scalar1, scalar2, n);
    
    for (int i = 0; i < n; i++) {
        if (gpu_result[i] != cpu_result[i]) return false;
    }
    return true;
}

inline bool verify_simpleKernel(const std::vector<float>& gpu_result,
                               const std::vector<float>& original,
                               int n) {
    std::vector<float> cpu_result = original;
    simpleKernel_cpu(cpu_result, n);
    
    for (int i = 0; i < n; i++) {
        if (gpu_result[i] != cpu_result[i]) return false;
    }
    return true;
}

inline bool verify_simpleKernelWithN(const std::vector<float>& gpu_result,
                                    const std::vector<float>& original,
                                    int n) {
    std::vector<float> cpu_result = original;
    simpleKernelWithN_cpu(cpu_result, n);
    
    for (int i = 0; i < n; i++) {
        if (gpu_result[i] != cpu_result[i]) return false;
    }
    return true;
}

} // namespace test_kernels
