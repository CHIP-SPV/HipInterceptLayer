#include "test_kernels.hh"
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

#define HIP_CHECK(call)                                                          \
    do {                                                                         \
        hipError_t err = call;                                                  \
        if (err != hipSuccess) {                                                \
            printf("HIP error %s:%d '%s'\n", __FILE__, __LINE__,               \
                   hipGetErrorString(err));                                     \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

int main() {
    const int N = 1024;
    const size_t size = N * sizeof(float);

    // Allocate and initialize host memory
    std::vector<float> h_a(N, 1.0f);
    std::vector<float> h_b(N, 2.0f);
    std::vector<float> h_c(N);
    std::vector<float> h_scalar(N);
    std::vector<float> h_simple(N, 3.0f);
    std::vector<float> h_simpleN(N, 4.0f);

    // Save original data for verification
    std::vector<float> orig_simple = h_simple;
    std::vector<float> orig_simpleN = h_simpleN;

    // Allocate device memory
    float *d_a, *d_b, *d_c, *d_scalar, *d_simple, *d_simpleN;
    HIP_CHECK(hipMalloc(&d_a, size));
    HIP_CHECK(hipMalloc(&d_b, size));
    HIP_CHECK(hipMalloc(&d_c, size));
    HIP_CHECK(hipMalloc(&d_scalar, size));
    HIP_CHECK(hipMalloc(&d_simple, size));
    HIP_CHECK(hipMalloc(&d_simpleN, size));

    // Copy data to device
    HIP_CHECK(hipMemcpy(d_a, h_a.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_simple, h_simple.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_simpleN, h_simpleN.data(), size, hipMemcpyHostToDevice));

    // Launch configuration
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    // Launch and verify kernels
    std::cout << "Running and verifying kernels..." << std::endl;

    // vectorAdd
    test_kernels::vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    HIP_CHECK(hipMemcpy(h_c.data(), d_c, size, hipMemcpyDeviceToHost));
    std::cout << "vectorAdd: " 
              << (test_kernels::verify_vectorAdd(h_c, h_a, h_b, N) ? "PASSED" : "FAILED") 
              << std::endl;

    // scalarKernel
    int scalar1 = 2;
    float scalar2 = 3.0f;
    test_kernels::scalarKernel<<<gridSize, blockSize>>>(d_scalar, scalar1, scalar2);
    HIP_CHECK(hipMemcpy(h_scalar.data(), d_scalar, size, hipMemcpyDeviceToHost));
    std::cout << "scalarKernel: " 
              << (test_kernels::verify_scalarKernel(h_scalar, scalar1, scalar2, N) ? "PASSED" : "FAILED") 
              << std::endl;

    // simpleKernel
    test_kernels::simpleKernel<<<gridSize, blockSize>>>(d_simple);
    HIP_CHECK(hipMemcpy(h_simple.data(), d_simple, size, hipMemcpyDeviceToHost));
    std::cout << "simpleKernel: " 
              << (test_kernels::verify_simpleKernel(h_simple, orig_simple, N) ? "PASSED" : "FAILED") 
              << std::endl;

    // simpleKernelWithN
    test_kernels::simpleKernelWithN<<<gridSize, blockSize>>>(d_simpleN, N);
    HIP_CHECK(hipMemcpy(h_simpleN.data(), d_simpleN, size, hipMemcpyDeviceToHost));
    std::cout << "simpleKernelWithN: " 
              << (test_kernels::verify_simpleKernelWithN(h_simpleN, orig_simpleN, N) ? "PASSED" : "FAILED") 
              << std::endl;

    // Cleanup
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
    HIP_CHECK(hipFree(d_scalar));
    HIP_CHECK(hipFree(d_simple));
    HIP_CHECK(hipFree(d_simpleN));

    return 0;
}
    