#include <hip/hip_runtime.h>
#include <iostream>
#include <cassert>

typedef unsigned int tileflags;

__launch_bounds__(64) __global__ void computeNonbonded(
    unsigned long long *__restrict__ forceBuffers,
    float *__restrict__ energyBuffer, const float4 *__restrict__ posq,
    const tileflags *__restrict__ exclusions,
    const int2 *__restrict__ exclusionTiles, unsigned int startTileIndex,
    unsigned long long numTileIndices,
    const int *__restrict__ tiles,
    const unsigned int *__restrict__ interactionCount, float4 periodicBoxSize,
    float4 invPeriodicBoxSize, float4 periodicBoxVecX, float4 periodicBoxVecY,
    float4 periodicBoxVecZ, unsigned int maxTiles,
    const float4 *__restrict__ blockCenter, const float4 *__restrict__ blockSize,
    const unsigned int *__restrict__ interactingAtoms,
    unsigned int maxSinglePairs, const int2 *__restrict__ singlePairs,
    const float2 *__restrict__ global_nonbonded0_sigmaEpsilon) {
    
    // Get thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Increment arrays
    if (idx < maxTiles) {
        // Increment force buffers
        if (forceBuffers) forceBuffers[idx]++;
        
        // Increment energy buffer
        if (energyBuffer) energyBuffer[idx]++;
        
        // Increment tiles
        if (tiles) atomicAdd((int*)&tiles[idx], 1);
        
        // Increment interaction count
        if (interactionCount) atomicAdd((unsigned int*)&interactionCount[idx], 1);
        
        // Increment interacting atoms
        if (interactingAtoms) atomicAdd((unsigned int*)&interactingAtoms[idx], 1);
    }
}

int main() {
    const int N = 128; // Number of elements
    const int BLOCK_SIZE = 64;
    const int NUM_BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Allocate and initialize host data
    unsigned long long* h_forceBuffers = new unsigned long long[N];
    float* h_energyBuffer = new float[N];
    float4* h_posq = new float4[N];
    tileflags* h_exclusions = new tileflags[N];
    int2* h_exclusionTiles = new int2[N];
    int* h_tiles = new int[N];
    unsigned int* h_interactionCount = new unsigned int[N];
    float4* h_blockCenter = new float4[N];
    float4* h_blockSize = new float4[N];
    unsigned int* h_interactingAtoms = new unsigned int[N];
    int2* h_singlePairs = new int2[N];
    float2* h_sigmaEpsilon = new float2[N];
    
    // Initialize with known values
    for (int i = 0; i < N; i++) {
        h_forceBuffers[i] = i;
        h_energyBuffer[i] = (float)i;
        h_posq[i] = make_float4(i, i+0.1f, i+0.2f, i+0.3f);
        h_exclusions[i] = i;
        h_exclusionTiles[i] = make_int2(i, i+1);
        h_tiles[i] = i;
        h_interactionCount[i] = i;
        h_blockCenter[i] = make_float4(i+0.5f, i+0.6f, i+0.7f, i+0.8f);
        h_blockSize[i] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        h_interactingAtoms[i] = i;
        h_singlePairs[i] = make_int2(i, i+2);
        h_sigmaEpsilon[i] = make_float2(0.5f + i, 1.0f + i);
    }
    
    // Allocate device memory
    unsigned long long* d_forceBuffers;
    float* d_energyBuffer;
    float4* d_posq;
    tileflags* d_exclusions;
    int2* d_exclusionTiles;
    int* d_tiles;
    unsigned int* d_interactionCount;
    float4* d_blockCenter;
    float4* d_blockSize;
    unsigned int* d_interactingAtoms;
    int2* d_singlePairs;
    float2* d_sigmaEpsilon;
    
    hipError_t err;
    err = hipMalloc(&d_forceBuffers, N * sizeof(unsigned long long));
    assert(err == hipSuccess);
    err = hipMalloc(&d_energyBuffer, N * sizeof(float));
    assert(err == hipSuccess);
    err = hipMalloc(&d_posq, N * sizeof(float4));
    assert(err == hipSuccess);
    err = hipMalloc(&d_exclusions, N * sizeof(tileflags));
    assert(err == hipSuccess);
    err = hipMalloc(&d_exclusionTiles, N * sizeof(int2));
    assert(err == hipSuccess);
    err = hipMalloc(&d_tiles, N * sizeof(int));
    assert(err == hipSuccess);
    err = hipMalloc(&d_interactionCount, N * sizeof(unsigned int));
    assert(err == hipSuccess);
    err = hipMalloc(&d_blockCenter, N * sizeof(float4));
    assert(err == hipSuccess);
    err = hipMalloc(&d_blockSize, N * sizeof(float4));
    assert(err == hipSuccess);
    err = hipMalloc(&d_interactingAtoms, N * sizeof(unsigned int));
    assert(err == hipSuccess);
    err = hipMalloc(&d_singlePairs, N * sizeof(int2));
    assert(err == hipSuccess);
    err = hipMalloc(&d_sigmaEpsilon, N * sizeof(float2));
    assert(err == hipSuccess);
    
    // Copy data to device
    err = hipMemcpy(d_forceBuffers, h_forceBuffers, N * sizeof(unsigned long long), hipMemcpyHostToDevice);
    assert(err == hipSuccess);
    err = hipMemcpy(d_energyBuffer, h_energyBuffer, N * sizeof(float), hipMemcpyHostToDevice);
    assert(err == hipSuccess);
    err = hipMemcpy(d_posq, h_posq, N * sizeof(float4), hipMemcpyHostToDevice);
    assert(err == hipSuccess);
    err = hipMemcpy(d_exclusions, h_exclusions, N * sizeof(tileflags), hipMemcpyHostToDevice);
    assert(err == hipSuccess);
    err = hipMemcpy(d_exclusionTiles, h_exclusionTiles, N * sizeof(int2), hipMemcpyHostToDevice);
    assert(err == hipSuccess);
    err = hipMemcpy(d_tiles, h_tiles, N * sizeof(int), hipMemcpyHostToDevice);
    assert(err == hipSuccess);
    err = hipMemcpy(d_interactionCount, h_interactionCount, N * sizeof(unsigned int), hipMemcpyHostToDevice);
    assert(err == hipSuccess);
    err = hipMemcpy(d_blockCenter, h_blockCenter, N * sizeof(float4), hipMemcpyHostToDevice);
    assert(err == hipSuccess);
    err = hipMemcpy(d_blockSize, h_blockSize, N * sizeof(float4), hipMemcpyHostToDevice);
    assert(err == hipSuccess);
    err = hipMemcpy(d_interactingAtoms, h_interactingAtoms, N * sizeof(unsigned int), hipMemcpyHostToDevice);
    assert(err == hipSuccess);
    err = hipMemcpy(d_singlePairs, h_singlePairs, N * sizeof(int2), hipMemcpyHostToDevice);
    assert(err == hipSuccess);
    err = hipMemcpy(d_sigmaEpsilon, h_sigmaEpsilon, N * sizeof(float2), hipMemcpyHostToDevice);
    assert(err == hipSuccess);
    
    // Launch kernel with all parameters initialized
    dim3 gridDim(NUM_BLOCKS, 1, 1);
    dim3 blockDim(BLOCK_SIZE, 1, 1);
    
    const unsigned int startTileIndex = 0;
    const unsigned long long numTileIndices = N;
    const float4 periodicBoxSize = make_float4(10.0f, 10.0f, 10.0f, 0.0f);
    const float4 invPeriodicBoxSize = make_float4(0.1f, 0.1f, 0.1f, 0.0f);
    const float4 periodicBoxVecX = make_float4(10.0f, 0.0f, 0.0f, 0.0f);
    const float4 periodicBoxVecY = make_float4(0.0f, 10.0f, 0.0f, 0.0f);
    const float4 periodicBoxVecZ = make_float4(0.0f, 0.0f, 10.0f, 0.0f);
    
    hipLaunchKernelGGL(computeNonbonded, gridDim, blockDim, 0, 0,
        d_forceBuffers, d_energyBuffer, d_posq,
        d_exclusions, d_exclusionTiles, startTileIndex, numTileIndices,
        d_tiles, d_interactionCount, periodicBoxSize,
        invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY,
        periodicBoxVecZ, N, d_blockCenter, d_blockSize,
        d_interactingAtoms, N, d_singlePairs, d_sigmaEpsilon);
    
    err = hipGetLastError();
    assert(err == hipSuccess && "Kernel launch failed");
    
    err = hipDeviceSynchronize();
    assert(err == hipSuccess && "Kernel execution failed");
    
    // Copy results back
    err = hipMemcpy(h_forceBuffers, d_forceBuffers, N * sizeof(unsigned long long), hipMemcpyDeviceToHost);
    assert(err == hipSuccess);
    err = hipMemcpy(h_energyBuffer, d_energyBuffer, N * sizeof(float), hipMemcpyDeviceToHost);
    assert(err == hipSuccess);
    err = hipMemcpy(h_tiles, d_tiles, N * sizeof(int), hipMemcpyDeviceToHost);
    assert(err == hipSuccess);
    err = hipMemcpy(h_interactionCount, d_interactionCount, N * sizeof(unsigned int), hipMemcpyDeviceToHost);
    assert(err == hipSuccess);
    err = hipMemcpy(h_interactingAtoms, d_interactingAtoms, N * sizeof(unsigned int), hipMemcpyDeviceToHost);
    assert(err == hipSuccess);
    
    // Verify results - each value should be incremented by 1
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_forceBuffers[i] != (unsigned long long)(i + 1)) {
            std::cerr << "Error: forceBuffers[" << i << "] = " << h_forceBuffers[i] 
                      << ", expected " << (i + 1) << std::endl;
            success = false;
        }
        if (h_energyBuffer[i] != (float)(i + 1)) {
            std::cerr << "Error: energyBuffer[" << i << "] = " << h_energyBuffer[i] 
                      << ", expected " << (i + 1) << std::endl;
            success = false;
        }
        if (h_tiles[i] != i + 1) {
            std::cerr << "Error: tiles[" << i << "] = " << h_tiles[i] 
                      << ", expected " << (i + 1) << std::endl;
            success = false;
        }
        if (h_interactionCount[i] != (unsigned int)(i + 1)) {
            std::cerr << "Error: interactionCount[" << i << "] = " << h_interactionCount[i] 
                      << ", expected " << (i + 1) << std::endl;
            success = false;
        }
        if (h_interactingAtoms[i] != (unsigned int)(i + 1)) {
            std::cerr << "Error: interactingAtoms[" << i << "] = " << h_interactingAtoms[i] 
                      << ", expected " << (i + 1) << std::endl;
            success = false;
        }
    }
    
    // Cleanup
    hipFree(d_forceBuffers);
    hipFree(d_energyBuffer);
    hipFree(d_posq);
    hipFree(d_exclusions);
    hipFree(d_exclusionTiles);
    hipFree(d_tiles);
    hipFree(d_interactionCount);
    hipFree(d_blockCenter);
    hipFree(d_blockSize);
    hipFree(d_interactingAtoms);
    hipFree(d_singlePairs);
    hipFree(d_sigmaEpsilon);
    
    delete[] h_forceBuffers;
    delete[] h_energyBuffer;
    delete[] h_posq;
    delete[] h_exclusions;
    delete[] h_exclusionTiles;
    delete[] h_tiles;
    delete[] h_interactionCount;
    delete[] h_blockCenter;
    delete[] h_blockSize;
    delete[] h_interactingAtoms;
    delete[] h_singlePairs;
    delete[] h_sigmaEpsilon;
    
    if (success) {
        std::cout << "Test passed" << std::endl;
    } else {
        std::cout << "Test failed" << std::endl;
    }
    return success ? 0 : 1;  // Return 0 on success, 1 on failure
}