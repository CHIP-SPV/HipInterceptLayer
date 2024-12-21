#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <vector>
#include <cassert>
#include "test_kernels.hh"

// Helper function to initialize data with known pattern
void initializeData(std::vector<float>& data, float start_val = 1.0f) {
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = start_val + i;
    }
}

// Helper function to verify data matches expected pattern
bool verifyData(const std::vector<float>& data, const std::vector<float>& expected) {
    for (size_t i = 0; i < data.size(); i++) {
        if (data[i] != expected[i]) {
            std::cout << "Mismatch at index " << i << ": "
                     << "actual=" << data[i] << ", expected=" << expected[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    const int N = 1024;
    std::vector<float> h_input(N);
    std::vector<float> h_output(N);
    std::vector<float> expected_output(N);

    // Initialize input with known pattern
    initializeData(h_input);
    
    // Calculate expected output
    expected_output = h_input;
    test_kernels::simpleKernel_cpu(expected_output, N);

    // Allocate device memory
    float *d_data;
    hipMalloc(&d_data, N * sizeof(float));

    // Copy input to device - this will be captured as a memory operation
    hipMemcpy(d_data, h_input.data(), N * sizeof(float), hipMemcpyHostToDevice);

    // Launch kernel - this will capture pre and post states
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    hipLaunchKernelGGL(test_kernels::simpleKernel, 
                       gridSize, blockSize, 
                       0, 0, d_data);
    hipDeviceSynchronize();

    // Copy result back - this will be captured as another memory operation
    hipMemcpy(h_output.data(), d_data, N * sizeof(float), hipMemcpyDeviceToHost);

    // Verify results
    bool success = verifyData(h_output, expected_output);
    if (!success) {
        std::cout << "Data verification failed!" << std::endl;
        return 1;
    }

    std::cout << "Data verification successful!" << std::endl;
    
    // Cleanup
    hipFree(d_data);

    // After the first test, add vector test
    const int VEC_N = 256;
    std::vector<float4> h_vec_input(VEC_N);
    std::vector<float4> h_vec_output(VEC_N);
    std::vector<float4> expected_vec_output(VEC_N);

    // Initialize vector input with known pattern
    for (int i = 0; i < VEC_N; i++) {
        h_vec_input[i].x = 1.0f + (i * 4);
        h_vec_input[i].y = 1.0f + (i * 4) + 1;
        h_vec_input[i].z = 1.0f + (i * 4) + 2;
        h_vec_input[i].w = 1.0f + (i * 4) + 3;
    }

    // Calculate expected vector output
    expected_vec_output = h_vec_input;
    test_kernels::vectorKernel_cpu(expected_vec_output, VEC_N);

    // Allocate device memory for vectors
    float4 *d_vec_data;
    hipMalloc(&d_vec_data, VEC_N * sizeof(float4));

    // Copy vector input to device
    hipMemcpy(d_vec_data, h_vec_input.data(), VEC_N * sizeof(float4), hipMemcpyHostToDevice);

    // Launch vector kernel
    dim3 vec_blockSize(256);
    dim3 vec_gridSize((VEC_N + vec_blockSize.x - 1) / vec_blockSize.x);
    hipLaunchKernelGGL(test_kernels::vectorKernel,
                       vec_gridSize, vec_blockSize,
                       0, 0, d_vec_data);
    hipDeviceSynchronize();

    // Copy vector result back
    hipMemcpy(h_vec_output.data(), d_vec_data, VEC_N * sizeof(float4), hipMemcpyDeviceToHost);

    // Verify vector results
    bool vec_success = true;
    for (int i = 0; i < VEC_N; i++) {
        if (h_vec_output[i].x != expected_vec_output[i].x ||
            h_vec_output[i].y != expected_vec_output[i].y ||
            h_vec_output[i].z != expected_vec_output[i].z ||
            h_vec_output[i].w != expected_vec_output[i].w) {
            std::cout << "Vector mismatch at index " << i << std::endl;
            vec_success = false;
            break;
        }
    }

    if (!vec_success) {
        std::cout << "Vector data verification failed!" << std::endl;
        return 1;
    }

    std::cout << "Vector data verification successful!" << std::endl;

    // Cleanup vector data
    hipFree(d_vec_data);

    return 0;
}
