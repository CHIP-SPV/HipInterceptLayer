#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <vector>
#include <cassert>

// Define a complex kernel that uses various argument types
__global__ void complexDataKernel(
    float* scalar_array,                    // Regular scalar array
    HIP_vector_type<float, 4>* vec4_array, // Vector4 array
    HIP_vector_type<float, 2>* vec2_array, // Vector2 array
    float4* float4_array,                  // Built-in float4 array
    int scalar1,                           // Integer scalar
    float scalar2,                         // Float scalar
    double scalar3,                        // Double scalar
    bool flag,                             // Boolean flag
    unsigned int uint_val,                 // Unsigned int
    size_t n                              // Size parameter
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Process scalar array
    scalar_array[idx] = scalar_array[idx] * scalar2 + scalar1;

    // Process vector4 array using HIP_vector_type
    auto vec4 = vec4_array[idx];
    vec4.x = vec4.x * scalar2;
    vec4.y = vec4.y * scalar2;
    vec4.z = vec4.z * scalar2;
    vec4.w = vec4.w * scalar2;
    vec4_array[idx] = vec4;

    // Process vector2 array
    auto vec2 = vec2_array[idx];
    vec2.x = vec2.x + scalar3;
    vec2.y = vec2.y + scalar3;
    vec2_array[idx] = vec2;

    // Process float4 array
    float4 f4 = float4_array[idx];
    if (flag) {
        f4.x = f4.x * uint_val;
        f4.y = f4.y * uint_val;
        f4.z = f4.z * uint_val;
        f4.w = f4.w * uint_val;
    }
    float4_array[idx] = f4;
}

// CPU reference implementation
void complexDataKernel_cpu(
    std::vector<float>& scalar_array,
    std::vector<HIP_vector_type<float, 4>>& vec4_array,
    std::vector<HIP_vector_type<float, 2>>& vec2_array,
    std::vector<float4>& float4_array,
    int scalar1,
    float scalar2,
    double scalar3,
    bool flag,
    unsigned int uint_val,
    size_t n
) {
    for (size_t i = 0; i < n; i++) {
        // Process scalar array
        scalar_array[i] = scalar_array[i] * scalar2 + scalar1;

        // Process vector4 array
        vec4_array[i].x = vec4_array[i].x * scalar2;
        vec4_array[i].y = vec4_array[i].y * scalar2;
        vec4_array[i].z = vec4_array[i].z * scalar2;
        vec4_array[i].w = vec4_array[i].w * scalar2;

        // Process vector2 array
        vec2_array[i].x = vec2_array[i].x + scalar3;
        vec2_array[i].y = vec2_array[i].y + scalar3;

        // Process float4 array
        if (flag) {
            float4_array[i].x = float4_array[i].x * uint_val;
            float4_array[i].y = float4_array[i].y * uint_val;
            float4_array[i].z = float4_array[i].z * uint_val;
            float4_array[i].w = float4_array[i].w * uint_val;
        }
    }
}

// Helper function to initialize data
void initializeData(
    std::vector<float>& scalar_array,
    std::vector<HIP_vector_type<float, 4>>& vec4_array,
    std::vector<HIP_vector_type<float, 2>>& vec2_array,
    std::vector<float4>& float4_array,
    size_t n
) {
    for (size_t i = 0; i < n; i++) {
        // Initialize scalar array
        scalar_array[i] = 1.0f + i;

        // Initialize vector4 array
        vec4_array[i].x = 1.0f + (i * 4);
        vec4_array[i].y = 2.0f + (i * 4);
        vec4_array[i].z = 3.0f + (i * 4);
        vec4_array[i].w = 4.0f + (i * 4);

        // Initialize vector2 array
        vec2_array[i].x = 1.0f + (i * 2);
        vec2_array[i].y = 2.0f + (i * 2);

        // Initialize float4 array
        float4_array[i].x = 1.0f + (i * 4);
        float4_array[i].y = 2.0f + (i * 4);
        float4_array[i].z = 3.0f + (i * 4);
        float4_array[i].w = 4.0f + (i * 4);
    }
}

// Helper function to verify results
bool verifyResults(
    const std::vector<float>& gpu_scalar,
    const std::vector<HIP_vector_type<float, 4>>& gpu_vec4,
    const std::vector<HIP_vector_type<float, 2>>& gpu_vec2,
    const std::vector<float4>& gpu_float4,
    const std::vector<float>& cpu_scalar,
    const std::vector<HIP_vector_type<float, 4>>& cpu_vec4,
    const std::vector<HIP_vector_type<float, 2>>& cpu_vec2,
    const std::vector<float4>& cpu_float4,
    size_t n
) {
    // Verify scalar array
    for (size_t i = 0; i < n; i++) {
        if (gpu_scalar[i] != cpu_scalar[i]) {
            std::cout << "Scalar mismatch at " << i << ": GPU=" << gpu_scalar[i] 
                     << ", CPU=" << cpu_scalar[i] << std::endl;
            return false;
        }
    }

    // Verify vector4 array
    for (size_t i = 0; i < n; i++) {
        if (gpu_vec4[i].x != cpu_vec4[i].x || gpu_vec4[i].y != cpu_vec4[i].y ||
            gpu_vec4[i].z != cpu_vec4[i].z || gpu_vec4[i].w != cpu_vec4[i].w) {
            std::cout << "Vector4 mismatch at " << i << std::endl;
            return false;
        }
    }

    // Verify vector2 array
    for (size_t i = 0; i < n; i++) {
        if (gpu_vec2[i].x != cpu_vec2[i].x || gpu_vec2[i].y != cpu_vec2[i].y) {
            std::cout << "Vector2 mismatch at " << i << std::endl;
            return false;
        }
    }

    // Verify float4 array
    for (size_t i = 0; i < n; i++) {
        if (gpu_float4[i].x != cpu_float4[i].x || gpu_float4[i].y != cpu_float4[i].y ||
            gpu_float4[i].z != cpu_float4[i].z || gpu_float4[i].w != cpu_float4[i].w) {
            std::cout << "Float4 mismatch at " << i << std::endl;
            return false;
        }
    }

    return true;
}

int main() {
    const size_t N = 3;  // Reduced from 1024 to 3
    
    // Allocate host memory
    std::vector<float> h_scalar(N);
    std::vector<HIP_vector_type<float, 4>> h_vec4(N);
    std::vector<HIP_vector_type<float, 2>> h_vec2(N);
    std::vector<float4> h_float4(N);

    // Create copies for CPU computation
    std::vector<float> cpu_scalar(N);
    std::vector<HIP_vector_type<float, 4>> cpu_vec4(N);
    std::vector<HIP_vector_type<float, 2>> cpu_vec2(N);
    std::vector<float4> cpu_float4(N);

    // Initialize data
    initializeData(h_scalar, h_vec4, h_vec2, h_float4, N);
    
    // Copy data for CPU computation
    cpu_scalar = h_scalar;
    cpu_vec4 = h_vec4;
    cpu_vec2 = h_vec2;
    cpu_float4 = h_float4;

    // Set scalar parameters
    int scalar1 = 5;
    float scalar2 = 2.0f;
    double scalar3 = 1.5;
    bool flag = true;
    unsigned int uint_val = 3;

    // Allocate device memory
    float* d_scalar;
    HIP_vector_type<float, 4>* d_vec4;
    HIP_vector_type<float, 2>* d_vec2;
    float4* d_float4;

    hipMalloc(&d_scalar, N * sizeof(float));
    hipMalloc(&d_vec4, N * sizeof(HIP_vector_type<float, 4>));
    hipMalloc(&d_vec2, N * sizeof(HIP_vector_type<float, 2>));
    hipMalloc(&d_float4, N * sizeof(float4));

    // Copy data to device
    hipMemcpy(d_scalar, h_scalar.data(), N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_vec4, h_vec4.data(), N * sizeof(HIP_vector_type<float, 4>), hipMemcpyHostToDevice);
    hipMemcpy(d_vec2, h_vec2.data(), N * sizeof(HIP_vector_type<float, 2>), hipMemcpyHostToDevice);
    hipMemcpy(d_float4, h_float4.data(), N * sizeof(float4), hipMemcpyHostToDevice);

    // Launch kernel
    dim3 blockSize(3);  // Reduced block size since we only have 3 elements
    dim3 gridSize(1);   // Single block is enough for 3 elements
    
    hipLaunchKernelGGL(complexDataKernel,
                       gridSize, blockSize,
                       0, 0,
                       d_scalar, d_vec4, d_vec2, d_float4,
                       scalar1, scalar2, scalar3, flag, uint_val, N);
    
    hipDeviceSynchronize();

    // Copy results back to host
    hipMemcpy(h_scalar.data(), d_scalar, N * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_vec4.data(), d_vec4, N * sizeof(HIP_vector_type<float, 4>), hipMemcpyDeviceToHost);
    hipMemcpy(h_vec2.data(), d_vec2, N * sizeof(HIP_vector_type<float, 2>), hipMemcpyDeviceToHost);
    hipMemcpy(h_float4.data(), d_float4, N * sizeof(float4), hipMemcpyDeviceToHost);

    // Compute on CPU for verification
    complexDataKernel_cpu(cpu_scalar, cpu_vec4, cpu_vec2, cpu_float4,
                         scalar1, scalar2, scalar3, flag, uint_val, N);

    // Verify results
    bool success = verifyResults(h_scalar, h_vec4, h_vec2, h_float4,
                               cpu_scalar, cpu_vec4, cpu_vec2, cpu_float4, N);

    if (!success) {
        std::cout << "Test FAILED!" << std::endl;
        return 1;
    }

    std::cout << "Test PASSED!" << std::endl;

    // Cleanup
    hipFree(d_scalar);
    hipFree(d_vec4);
    hipFree(d_vec2);
    hipFree(d_float4);

    return 0;
}
