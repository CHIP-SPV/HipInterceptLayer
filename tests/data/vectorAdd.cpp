#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Vector increment kernel using float4
__global__ void vectorIncrementKernel(float4 in_scalar_float4, float in_scalar_float, 
                                    float4* inout_vector_float4, float* inout_vector_float) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Increment float4 component using the input value (not array)
    inout_vector_float4[idx].x = in_scalar_float4.x + in_scalar_float;
    inout_vector_float4[idx].y = in_scalar_float4.y + in_scalar_float;
    inout_vector_float4[idx].z = in_scalar_float4.z + in_scalar_float;
    inout_vector_float4[idx].w = in_scalar_float4.w + in_scalar_float;
    
    // Increment scalar component
    inout_vector_float[idx] = in_scalar_float * 2.0f;  // Double the scalar value
}

int main() {
    const int N = 1024;
    const size_t vec4_bytes = N * sizeof(float4);
    const size_t scalar_bytes = N * sizeof(float);

    // Host arrays for output
    float4 *h_inout_vector_float4;
    float *h_inout_vector_float;
    
    // Input values (not arrays)
    float4 h_in_scalar_float4;
    float h_in_scalar_float;
    
    // Device arrays (only for output)
    float4 *d_inout_vector_float4;
    float *d_inout_vector_float;

    // Initialize input values with non-zero values
    h_in_scalar_float4.x = 1.0f;
    h_in_scalar_float4.y = 2.0f;
    h_in_scalar_float4.z = 3.0f;
    h_in_scalar_float4.w = 4.0f;
    h_in_scalar_float = 0.5f;

    // Allocate host memory for output
    h_inout_vector_float4 = (float4*)malloc(vec4_bytes);
    h_inout_vector_float = (float*)malloc(scalar_bytes);

    // Allocate device memory (only for output)
    hipMalloc(&d_inout_vector_float4, vec4_bytes);
    hipMalloc(&d_inout_vector_float, scalar_bytes);

    // Launch kernel
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    hipLaunchKernelGGL(vectorIncrementKernel, gridSize, blockSize, 0, 0,
                       h_in_scalar_float4, h_in_scalar_float, 
                       d_inout_vector_float4, d_inout_vector_float);

    // Copy results back to host
    hipMemcpy(h_inout_vector_float4, d_inout_vector_float4, vec4_bytes, hipMemcpyDeviceToHost);
    hipMemcpy(h_inout_vector_float, d_inout_vector_float, scalar_bytes, hipMemcpyDeviceToHost);

    // Verify results (check first few elements)
    printf("Input values:\n");
    printf("  in_scalar_float4: (%f, %f, %f, %f), in_scalar_float: %f\n",
           h_in_scalar_float4.x, h_in_scalar_float4.y, h_in_scalar_float4.z, h_in_scalar_float4.w,
           h_in_scalar_float);
    
    printf("\nFirst few elements of the result:\n");
    for(int i = 0; i < 5; i++) {
        printf("Element %d:\n", i);
        printf("  inout_vector_float4: (%f, %f, %f, %f), inout_vector_float: %f\n",
               h_inout_vector_float4[i].x, h_inout_vector_float4[i].y, 
               h_inout_vector_float4[i].z, h_inout_vector_float4[i].w,
               h_inout_vector_float[i]);
    }

    // Cleanup
    hipFree(d_inout_vector_float4);
    hipFree(d_inout_vector_float);
    free(h_inout_vector_float4);
    free(h_inout_vector_float);

    return 0;
}
