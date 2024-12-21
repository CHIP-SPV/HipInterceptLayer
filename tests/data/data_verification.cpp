#include <hip/hip_runtime.h>
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
    return 0;
}
