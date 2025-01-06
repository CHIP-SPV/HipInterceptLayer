#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Vector increment kernel using float4
__global__ void vectorIncrementKernel(float4 input_vec4, float input_scalar, 
                                    float4* output_vec4, float* output_scalar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Increment float4 component using the input value (not array)
    output_vec4[idx].x = input_vec4.x + input_scalar;
    output_vec4[idx].y = input_vec4.y + input_scalar;
    output_vec4[idx].z = input_vec4.z + input_scalar;
    output_vec4[idx].w = input_vec4.w + input_scalar;
    
    // Increment scalar component
    output_scalar[idx] = input_scalar * 2.0f;  // Double the scalar value
}

int main() {
    const int N = 1024;
    const size_t vec4_bytes = N * sizeof(float4);
    const size_t scalar_bytes = N * sizeof(float);

    // Host arrays for output
    float4 *h_output_vec4;
    float *h_output_scalar;
    
    // Input values (not arrays)
    float4 h_input_vec4;
    float h_input_scalar;
    
    // Device arrays (only for output)
    float4 *d_output_vec4;
    float *d_output_scalar;

    // Initialize input values with non-zero values
    h_input_vec4.x = 1.0f;
    h_input_vec4.y = 2.0f;
    h_input_vec4.z = 3.0f;
    h_input_vec4.w = 4.0f;
    h_input_scalar = 0.5f;

    // Allocate host memory for output
    h_output_vec4 = (float4*)malloc(vec4_bytes);
    h_output_scalar = (float*)malloc(scalar_bytes);

    // Allocate device memory (only for output)
    hipMalloc(&d_output_vec4, vec4_bytes);
    hipMalloc(&d_output_scalar, scalar_bytes);

    // Launch kernel
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    hipLaunchKernelGGL(vectorIncrementKernel, gridSize, blockSize, 0, 0,
                       h_input_vec4, h_input_scalar, d_output_vec4, d_output_scalar);

    // Copy results back to host
    hipMemcpy(h_output_vec4, d_output_vec4, vec4_bytes, hipMemcpyDeviceToHost);
    hipMemcpy(h_output_scalar, d_output_scalar, scalar_bytes, hipMemcpyDeviceToHost);

    // Verify results (check first few elements)
    printf("Input values:\n");
    printf("  Input vec4: (%f, %f, %f, %f), scalar: %f\n",
           h_input_vec4.x, h_input_vec4.y, h_input_vec4.z, h_input_vec4.w,
           h_input_scalar);
    
    printf("\nFirst few elements of the result:\n");
    for(int i = 0; i < 5; i++) {
        printf("Element %d:\n", i);
        printf("  Output vec4: (%f, %f, %f, %f), scalar: %f\n",
               h_output_vec4[i].x, h_output_vec4[i].y, h_output_vec4[i].z, h_output_vec4[i].w,
               h_output_scalar[i]);
    }

    // Cleanup
    hipFree(d_output_vec4);
    hipFree(d_output_scalar);
    free(h_output_vec4);
    free(h_output_scalar);

    return 0;
}
