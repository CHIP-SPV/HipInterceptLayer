#include <gtest/gtest.h>
#include "../Interceptor.hh"
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>

class InterceptorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up code if needed
    }
};

TEST_F(InterceptorTest, VariousDataTypesMemoryStateTest) {
    const size_t N = 4; // Test with 4 elements for vector types
    
    // Allocate host memory for different data types
    float* h_float = new float[N];
    float4* h_float4 = new float4[N];
    uint32_t* h_uint = new uint32_t[N];
    float2* h_vec2 = new float2[N];
    
    // Initialize host data with test values
    for (size_t i = 0; i < N; i++) {
        h_float[i] = 1.0f + i;
        h_float4[i] = make_float4(i, i + 0.5f, i + 1.0f, i + 1.5f);
        h_uint[i] = i + 42;
        h_vec2[i] = make_float2(i + 0.1f, i + 0.2f);
    }
    
    // Allocate device memory
    float* d_float;
    float4* d_float4;
    uint32_t* d_uint;
    float2* d_vec2;
    
    ASSERT_EQ(hipMalloc(&d_float, N * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_float4, N * sizeof(float4)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_uint, N * sizeof(uint32_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_vec2, N * sizeof(float2)), hipSuccess);
    
    // Copy data to device
    ASSERT_EQ(hipMemcpy(d_float, h_float, N * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_float4, h_float4, N * sizeof(float4), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_uint, h_uint, N * sizeof(uint32_t), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_vec2, h_vec2, N * sizeof(float2), hipMemcpyHostToDevice), hipSuccess);
    
    // Create arrays for verification
    float* verify_float = new float[N];
    float4* verify_float4 = new float4[N];
    uint32_t* verify_uint = new uint32_t[N];
    float2* verify_vec2 = new float2[N];
    
    // Copy data back to host for verification
    ASSERT_EQ(hipMemcpy(verify_float, d_float, N * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(verify_float4, d_float4, N * sizeof(float4), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(verify_uint, d_uint, N * sizeof(uint32_t), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(verify_vec2, d_vec2, N * sizeof(float2), hipMemcpyDeviceToHost), hipSuccess);
    
    // Verify data matches
    for (size_t i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(verify_float[i], h_float[i]);
        
        EXPECT_FLOAT_EQ(verify_float4[i].x, h_float4[i].x);
        EXPECT_FLOAT_EQ(verify_float4[i].y, h_float4[i].y);
        EXPECT_FLOAT_EQ(verify_float4[i].z, h_float4[i].z);
        EXPECT_FLOAT_EQ(verify_float4[i].w, h_float4[i].w);
        
        EXPECT_EQ(verify_uint[i], h_uint[i]);
        
        EXPECT_FLOAT_EQ(verify_vec2[i].x, h_vec2[i].x);
        EXPECT_FLOAT_EQ(verify_vec2[i].y, h_vec2[i].y);
    }
    
    // Cleanup
    delete[] h_float;
    delete[] h_float4;
    delete[] h_uint;
    delete[] h_vec2;
    delete[] verify_float;
    delete[] verify_float4;
    delete[] verify_uint;
    delete[] verify_vec2;
    
    ASSERT_EQ(hipFree(d_float), hipSuccess);
    ASSERT_EQ(hipFree(d_float4), hipSuccess);
    ASSERT_EQ(hipFree(d_uint), hipSuccess);
    ASSERT_EQ(hipFree(d_vec2), hipSuccess);
}

TEST_F(InterceptorTest, DataVerificationTraceTest) {
    // Load the trace file generated during build
    std::string trace_file = std::string(CMAKE_BINARY_DIR) + "/tests/data_verification-0.trace";
    
    // Create a Tracer instance to read the trace file
    Tracer tracer(trace_file);
    std::cout << tracer << std::endl;
    tracer.setSerializeTrace(false);
    
    // We expect 9 operations:
    // 4 H2D copies
    // 1 kernel
    // 4 D2H copies
    ASSERT_EQ(tracer.getNumOperations(), 9) << "Expected 9 operations in trace";
    
    const size_t N = 3;  // Same size as in data_verification.cpp
    
    // Get the kernel operation (should be operation 4)
    auto kernel_op = tracer.getOperation(4);
    ASSERT_TRUE(kernel_op->isKernel()) << "Operation 4 should be a kernel";
    auto kernel = static_cast<const KernelExecution*>(kernel_op.get());
    
    // Verify kernel name and configuration
    EXPECT_EQ(kernel->kernel_name, "complexDataKernel") << "Incorrect kernel name";
    EXPECT_EQ(kernel->grid_dim.x, 1) << "Incorrect grid dimension x";
    EXPECT_EQ(kernel->grid_dim.y, 1) << "Incorrect grid dimension y";
    EXPECT_EQ(kernel->grid_dim.z, 1) << "Incorrect grid dimension z";
    EXPECT_EQ(kernel->block_dim.x, 3) << "Incorrect block dimension x";
    EXPECT_EQ(kernel->block_dim.y, 1) << "Incorrect block dimension y";
    EXPECT_EQ(kernel->block_dim.z, 1) << "Incorrect block dimension z";
    
    // Get memory operations for scalar array
    auto scalar_h2d = tracer.getOperation(0);
    auto scalar_d2h = tracer.getOperation(5);
    
    ASSERT_TRUE(scalar_h2d->isMemory()) << "Operation 0 should be memory operation";
    ASSERT_TRUE(scalar_d2h->isMemory()) << "Operation 5 should be memory operation";
    
    // Verify scalar array data
    auto scalar_h2d_op = static_cast<const MemoryOperation*>(scalar_h2d.get());
    auto scalar_d2h_op = static_cast<const MemoryOperation*>(scalar_d2h.get());
    
    ASSERT_EQ(scalar_h2d_op->size, N * sizeof(float)) << "Incorrect scalar array size";
    ASSERT_EQ(scalar_d2h_op->size, N * sizeof(float)) << "Incorrect scalar array size";
    
    // Get pre and post state data for scalar array
    auto pre_scalar_data = scalar_h2d_op->pre_state.getData();
    auto post_scalar_data = scalar_d2h_op->post_state.getData();
    ASSERT_NE(pre_scalar_data, nullptr);
    ASSERT_NE(post_scalar_data, nullptr);
    
    // Ensure proper alignment for scalar array
    std::vector<float> pre_scalar_aligned(N);
    std::vector<float> post_scalar_aligned(N);
    std::memcpy(pre_scalar_aligned.data(), pre_scalar_data.get(), N * sizeof(float));
    std::memcpy(post_scalar_aligned.data(), post_scalar_data.get(), N * sizeof(float));
    
    // Verify scalar array values
    for (size_t i = 0; i < N; i++) {
        // Pre-state: initialized as 1.0f + i
        EXPECT_FLOAT_EQ(pre_scalar_aligned[i], 1.0f + i) 
            << "Pre-state scalar mismatch at index " << i;
        
        // Post-state: scalar[i] = scalar[i] * scalar2 + scalar1
        // where scalar2 = 2.0f and scalar1 = 5
        float expected = (1.0f + i) * 2.0f + 5.0f;
        EXPECT_FLOAT_EQ(post_scalar_aligned[i], expected)
            << "Post-state scalar mismatch at index " << i;
    }
    
    // Get memory operations for vec4 array
    auto vec4_h2d = tracer.getOperation(1);
    auto vec4_d2h = tracer.getOperation(6);
    
    ASSERT_TRUE(vec4_h2d->isMemory()) << "Operation 1 should be memory operation";
    ASSERT_TRUE(vec4_d2h->isMemory()) << "Operation 6 should be memory operation";
    
    // Verify vec4 array data
    auto vec4_h2d_op = static_cast<const MemoryOperation*>(vec4_h2d.get());
    auto vec4_d2h_op = static_cast<const MemoryOperation*>(vec4_d2h.get());
    
    ASSERT_EQ(vec4_h2d_op->size, N * sizeof(HIP_vector_type<float, 4>)) << "Incorrect vec4 array size";
    ASSERT_EQ(vec4_d2h_op->size, N * sizeof(HIP_vector_type<float, 4>)) << "Incorrect vec4 array size";
    
    // Get pre and post state data for vec4 array
    auto pre_vec4_data = vec4_h2d_op->pre_state.getData();
    auto post_vec4_data = vec4_d2h_op->post_state.getData();
    ASSERT_NE(pre_vec4_data, nullptr);
    ASSERT_NE(post_vec4_data, nullptr);
    
    // Ensure proper alignment for vec4 array
    std::vector<HIP_vector_type<float, 4>> pre_vec4_aligned(N);
    std::vector<HIP_vector_type<float, 4>> post_vec4_aligned(N);
    std::memcpy(pre_vec4_aligned.data(), pre_vec4_data.get(), N * sizeof(HIP_vector_type<float, 4>));
    std::memcpy(post_vec4_aligned.data(), post_vec4_data.get(), N * sizeof(HIP_vector_type<float, 4>));
    
    // Verify vec4 array values
    for (size_t i = 0; i < N; i++) {
        // Pre-state: initialized as 1.0f + (i * 4), 2.0f + (i * 4), etc.
        EXPECT_FLOAT_EQ(pre_vec4_aligned[i].x, 1.0f + (i * 4)) << "Pre-state vec4 x mismatch at index " << i;
        EXPECT_FLOAT_EQ(pre_vec4_aligned[i].y, 2.0f + (i * 4)) << "Pre-state vec4 y mismatch at index " << i;
        EXPECT_FLOAT_EQ(pre_vec4_aligned[i].z, 3.0f + (i * 4)) << "Pre-state vec4 z mismatch at index " << i;
        EXPECT_FLOAT_EQ(pre_vec4_aligned[i].w, 4.0f + (i * 4)) << "Pre-state vec4 w mismatch at index " << i;
        
        // Post-state: vec4[i] = vec4[i] * scalar2 where scalar2 = 2.0f
        EXPECT_FLOAT_EQ(post_vec4_aligned[i].x, (1.0f + (i * 4)) * 2.0f) << "Post-state vec4 x mismatch at index " << i;
        EXPECT_FLOAT_EQ(post_vec4_aligned[i].y, (2.0f + (i * 4)) * 2.0f) << "Post-state vec4 y mismatch at index " << i;
        EXPECT_FLOAT_EQ(post_vec4_aligned[i].z, (3.0f + (i * 4)) * 2.0f) << "Post-state vec4 z mismatch at index " << i;
        EXPECT_FLOAT_EQ(post_vec4_aligned[i].w, (4.0f + (i * 4)) * 2.0f) << "Post-state vec4 w mismatch at index " << i;
    }
    
    // Get memory operations for vec2 array
    auto vec2_h2d = tracer.getOperation(2);
    auto vec2_d2h = tracer.getOperation(7);
    
    ASSERT_TRUE(vec2_h2d->isMemory()) << "Operation 2 should be memory operation";
    ASSERT_TRUE(vec2_d2h->isMemory()) << "Operation 7 should be memory operation";
    
    // Verify vec2 array data
    auto vec2_h2d_op = static_cast<const MemoryOperation*>(vec2_h2d.get());
    auto vec2_d2h_op = static_cast<const MemoryOperation*>(vec2_d2h.get());
    
    ASSERT_EQ(vec2_h2d_op->size, N * sizeof(HIP_vector_type<float, 2>)) << "Incorrect vec2 array size";
    ASSERT_EQ(vec2_d2h_op->size, N * sizeof(HIP_vector_type<float, 2>)) << "Incorrect vec2 array size";
    
    // Get pre and post state data for vec2 array
    auto pre_vec2_data = vec2_h2d_op->pre_state.getData();
    auto post_vec2_data = vec2_d2h_op->post_state.getData();
    ASSERT_NE(pre_vec2_data, nullptr);
    ASSERT_NE(post_vec2_data, nullptr);
    
    // Ensure proper alignment for vec2 array
    std::vector<HIP_vector_type<float, 2>> pre_vec2_aligned(N);
    std::vector<HIP_vector_type<float, 2>> post_vec2_aligned(N);
    std::memcpy(pre_vec2_aligned.data(), pre_vec2_data.get(), N * sizeof(HIP_vector_type<float, 2>));
    std::memcpy(post_vec2_aligned.data(), post_vec2_data.get(), N * sizeof(HIP_vector_type<float, 2>));
    
    // Verify vec2 array values
    for (size_t i = 0; i < N; i++) {
        // Pre-state: initialized as 1.0f + (i * 2), 2.0f + (i * 2)
        EXPECT_FLOAT_EQ(pre_vec2_aligned[i].x, 1.0f + (i * 2)) << "Pre-state vec2 x mismatch at index " << i;
        EXPECT_FLOAT_EQ(pre_vec2_aligned[i].y, 2.0f + (i * 2)) << "Pre-state vec2 y mismatch at index " << i;
        
        // Post-state: vec2[i] = vec2[i] + scalar3 where scalar3 = 1.5
        EXPECT_FLOAT_EQ(post_vec2_aligned[i].x, (1.0f + (i * 2)) + 1.5f) << "Post-state vec2 x mismatch at index " << i;
        EXPECT_FLOAT_EQ(post_vec2_aligned[i].y, (2.0f + (i * 2)) + 1.5f) << "Post-state vec2 y mismatch at index " << i;
    }
    
    // Get memory operations for float4 array
    auto float4_h2d = tracer.getOperation(3);
    auto float4_d2h = tracer.getOperation(8);
    
    ASSERT_TRUE(float4_h2d->isMemory()) << "Operation 3 should be memory operation";
    ASSERT_TRUE(float4_d2h->isMemory()) << "Operation 8 should be memory operation";
    
    // Verify float4 array data
    auto float4_h2d_op = static_cast<const MemoryOperation*>(float4_h2d.get());
    auto float4_d2h_op = static_cast<const MemoryOperation*>(float4_d2h.get());
    
    ASSERT_EQ(float4_h2d_op->size, N * sizeof(float4)) << "Incorrect float4 array size";
    ASSERT_EQ(float4_d2h_op->size, N * sizeof(float4)) << "Incorrect float4 array size";
    
    // Get pre and post state data for float4 array
    auto pre_float4_data = float4_h2d_op->pre_state.getData();
    auto post_float4_data = float4_d2h_op->post_state.getData();
    ASSERT_NE(pre_float4_data, nullptr);
    ASSERT_NE(post_float4_data, nullptr);
    
    // Ensure proper alignment for float4 array
    std::vector<float4> pre_float4_aligned(N);
    std::vector<float4> post_float4_aligned(N);
    std::memcpy(pre_float4_aligned.data(), pre_float4_data.get(), N * sizeof(float4));
    std::memcpy(post_float4_aligned.data(), post_float4_data.get(), N * sizeof(float4));
    
    // Verify float4 array values
    for (size_t i = 0; i < N; i++) {
        // Pre-state: initialized as 1.0f + (i * 4), 2.0f + (i * 4), etc.
        EXPECT_FLOAT_EQ(pre_float4_aligned[i].x, 1.0f + (i * 4)) << "Pre-state float4 x mismatch at index " << i;
        EXPECT_FLOAT_EQ(pre_float4_aligned[i].y, 2.0f + (i * 4)) << "Pre-state float4 y mismatch at index " << i;
        EXPECT_FLOAT_EQ(pre_float4_aligned[i].z, 3.0f + (i * 4)) << "Pre-state float4 z mismatch at index " << i;
        EXPECT_FLOAT_EQ(pre_float4_aligned[i].w, 4.0f + (i * 4)) << "Pre-state float4 w mismatch at index " << i;
        
        // Post-state: float4[i] = float4[i] * uint_val where uint_val = 3 and flag = true
        EXPECT_FLOAT_EQ(post_float4_aligned[i].x, (1.0f + (i * 4)) * 3.0f) << "Post-state float4 x mismatch at index " << i;
        EXPECT_FLOAT_EQ(post_float4_aligned[i].y, (2.0f + (i * 4)) * 3.0f) << "Post-state float4 y mismatch at index " << i;
        EXPECT_FLOAT_EQ(post_float4_aligned[i].z, (3.0f + (i * 4)) * 3.0f) << "Post-state float4 z mismatch at index " << i;
        EXPECT_FLOAT_EQ(post_float4_aligned[i].w, (4.0f + (i * 4)) * 3.0f) << "Post-state float4 w mismatch at index " << i;
    }
}

TEST_F(InterceptorTest, SimpleMemoryStateTest) {
    // Create a simple array of floats
    const size_t size = 3 * sizeof(float);
    float input[3] = {1.0f, 2.0f, 3.0f};
    
    // Create a MemoryState and capture the data
    MemoryState state;
    state.captureHostMemory(input, size);
    
    // Get the data back and verify
    auto data = state.getData();
    ASSERT_NE(data, nullptr);
    
    float* output = reinterpret_cast<float*>(data.get());
    for (int i = 0; i < 3; i++) {
        EXPECT_FLOAT_EQ(output[i], input[i]) << "Mismatch at index " << i;
    }
}

TEST_F(InterceptorTest, MemoryStateHashTest) {
    // Create test data
    const size_t size = 1024;
    std::vector<float> data(size);
    for (size_t i = 0; i < size; i++) {
        data[i] = static_cast<float>(i);
    }

    // Allocate GPU memory
    float* d_data;
    ASSERT_EQ(hipMalloc(&d_data, size * sizeof(float)), hipSuccess);

    // Copy data to GPU
    ASSERT_EQ(hipMemcpy(d_data, data.data(), size * sizeof(float), hipMemcpyHostToDevice), hipSuccess);

    // Create two MemoryState objects and capture the same GPU memory
    MemoryState state1, state2;
    state1.captureGpuMemory(d_data, size * sizeof(float));
    state2.captureGpuMemory(d_data, size * sizeof(float));

    // Get the data and hashes
    auto data1 = state1.getData();
    auto data2 = state2.getData();
    
    std::string hash1 = state1.calculateHash(data1.get(), size * sizeof(float));
    std::string hash2 = state2.calculateHash(data2.get(), size * sizeof(float));

    // Verify hashes match
    EXPECT_EQ(hash1, hash2) << "Memory state hashes don't match for identical GPU memory";

    // Verify the actual data matches
    ASSERT_NE(data1, nullptr);
    ASSERT_NE(data2, nullptr);
    EXPECT_EQ(memcmp(data1.get(), data2.get(), size * sizeof(float)), 0) 
        << "Memory contents don't match for identical GPU memory";

    // Clean up
    ASSERT_EQ(hipFree(d_data), hipSuccess);
}

TEST_F(InterceptorTest, MemoryStateCaptureReplayTest) {
    // Create test data
    const size_t size = 1024;
    std::vector<float> data(size);
    for (size_t i = 0; i < size; i++) {
        data[i] = static_cast<float>(i);
    }

    // First run: Capture the memory state
    float* d_data1;
    ASSERT_EQ(hipMalloc(&d_data1, size * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_data1, data.data(), size * sizeof(float), hipMemcpyHostToDevice), hipSuccess);

    MemoryState capture_state;
    capture_state.captureGpuMemory(d_data1, size * sizeof(float));
    auto capture_data = capture_state.getData();
    std::string capture_hash = capture_state.calculateHash(capture_data.get(), size * sizeof(float));

    // Second run: Replay with new allocation
    float* d_data2;
    ASSERT_EQ(hipMalloc(&d_data2, size * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_data2, data.data(), size * sizeof(float), hipMemcpyHostToDevice), hipSuccess);

    MemoryState replay_state;
    replay_state.captureGpuMemory(d_data2, size * sizeof(float));
    auto replay_data = replay_state.getData();
    std::string replay_hash = replay_state.calculateHash(replay_data.get(), size * sizeof(float));

    // Verify hashes match between capture and replay
    EXPECT_EQ(capture_hash, replay_hash) 
        << "Memory state hashes don't match between capture and replay\n"
        << "Capture hash: " << capture_hash << "\n"
        << "Replay hash: " << replay_hash;

    // Verify the actual data matches
    ASSERT_NE(capture_data, nullptr);
    ASSERT_NE(replay_data, nullptr);
    EXPECT_EQ(memcmp(capture_data.get(), replay_data.get(), size * sizeof(float)), 0)
        << "Memory contents don't match between capture and replay";

    // Clean up
    ASSERT_EQ(hipFree(d_data1), hipSuccess);
    ASSERT_EQ(hipFree(d_data2), hipSuccess);
}
