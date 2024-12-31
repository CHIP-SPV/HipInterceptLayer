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
    const float* pre_scalar = reinterpret_cast<const float*>(scalar_h2d_op->pre_state->getData().get());
    const float* post_scalar = reinterpret_cast<const float*>(scalar_d2h_op->post_state->getData().get());
    
    // Verify scalar array values
    for (size_t i = 0; i < N; i++) {
        // Pre-state: initialized as 1.0f + i
        EXPECT_FLOAT_EQ(pre_scalar[i], 1.0f + i) 
            << "Pre-state scalar mismatch at index " << i;
        
        // Post-state: scalar[i] = scalar[i] * scalar2 + scalar1
        // where scalar2 = 2.0f and scalar1 = 5
        float expected = (1.0f + i) * 2.0f + 5.0f;
        EXPECT_FLOAT_EQ(post_scalar[i], expected)
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
    const HIP_vector_type<float, 4>* pre_vec4 = reinterpret_cast<const HIP_vector_type<float, 4>*>(vec4_h2d_op->pre_state->getData().get());
    const HIP_vector_type<float, 4>* post_vec4 = reinterpret_cast<const HIP_vector_type<float, 4>*>(vec4_d2h_op->post_state->getData().get());
    
    // Verify vec4 array values
    for (size_t i = 0; i < N; i++) {
        // Pre-state: initialized as 1.0f + (i * 4), 2.0f + (i * 4), etc.
        EXPECT_FLOAT_EQ(pre_vec4[i].x, 1.0f + (i * 4)) << "Pre-state vec4 x mismatch at index " << i;
        EXPECT_FLOAT_EQ(pre_vec4[i].y, 2.0f + (i * 4)) << "Pre-state vec4 y mismatch at index " << i;
        EXPECT_FLOAT_EQ(pre_vec4[i].z, 3.0f + (i * 4)) << "Pre-state vec4 z mismatch at index " << i;
        EXPECT_FLOAT_EQ(pre_vec4[i].w, 4.0f + (i * 4)) << "Pre-state vec4 w mismatch at index " << i;
        
        // Post-state: vec4[i] = vec4[i] * scalar2 where scalar2 = 2.0f
        EXPECT_FLOAT_EQ(post_vec4[i].x, (1.0f + (i * 4)) * 2.0f) << "Post-state vec4 x mismatch at index " << i;
        EXPECT_FLOAT_EQ(post_vec4[i].y, (2.0f + (i * 4)) * 2.0f) << "Post-state vec4 y mismatch at index " << i;
        EXPECT_FLOAT_EQ(post_vec4[i].z, (3.0f + (i * 4)) * 2.0f) << "Post-state vec4 z mismatch at index " << i;
        EXPECT_FLOAT_EQ(post_vec4[i].w, (4.0f + (i * 4)) * 2.0f) << "Post-state vec4 w mismatch at index " << i;
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
    const HIP_vector_type<float, 2>* pre_vec2 = reinterpret_cast<const HIP_vector_type<float, 2>*>(vec2_h2d_op->pre_state->getData().get());
    const HIP_vector_type<float, 2>* post_vec2 = reinterpret_cast<const HIP_vector_type<float, 2>*>(vec2_d2h_op->post_state->getData().get());
    
    // Verify vec2 array values
    for (size_t i = 0; i < N; i++) {
        // Pre-state: initialized as 1.0f + (i * 2), 2.0f + (i * 2)
        EXPECT_FLOAT_EQ(pre_vec2[i].x, 1.0f + (i * 2)) << "Pre-state vec2 x mismatch at index " << i;
        EXPECT_FLOAT_EQ(pre_vec2[i].y, 2.0f + (i * 2)) << "Pre-state vec2 y mismatch at index " << i;
        
        // Post-state: vec2[i] = vec2[i] + scalar3 where scalar3 = 1.5
        EXPECT_FLOAT_EQ(post_vec2[i].x, (1.0f + (i * 2)) + 1.5f) << "Post-state vec2 x mismatch at index " << i;
        EXPECT_FLOAT_EQ(post_vec2[i].y, (2.0f + (i * 2)) + 1.5f) << "Post-state vec2 y mismatch at index " << i;
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
    const float4* pre_float4 = reinterpret_cast<const float4*>(float4_h2d_op->pre_state->getData().get());
    const float4* post_float4 = reinterpret_cast<const float4*>(float4_d2h_op->post_state->getData().get());
    
    // Verify float4 array values
    for (size_t i = 0; i < N; i++) {
        // Pre-state: initialized as 1.0f + (i * 4), 2.0f + (i * 4), etc.
        EXPECT_FLOAT_EQ(pre_float4[i].x, 1.0f + (i * 4)) << "Pre-state float4 x mismatch at index " << i;
        EXPECT_FLOAT_EQ(pre_float4[i].y, 2.0f + (i * 4)) << "Pre-state float4 y mismatch at index " << i;
        EXPECT_FLOAT_EQ(pre_float4[i].z, 3.0f + (i * 4)) << "Pre-state float4 z mismatch at index " << i;
        EXPECT_FLOAT_EQ(pre_float4[i].w, 4.0f + (i * 4)) << "Pre-state float4 w mismatch at index " << i;
        
        // Post-state: float4[i] = float4[i] * uint_val where uint_val = 3 and flag = true
        EXPECT_FLOAT_EQ(post_float4[i].x, (1.0f + (i * 4)) * 3.0f) << "Post-state float4 x mismatch at index " << i;
        EXPECT_FLOAT_EQ(post_float4[i].y, (2.0f + (i * 4)) * 3.0f) << "Post-state float4 y mismatch at index " << i;
        EXPECT_FLOAT_EQ(post_float4[i].z, (3.0f + (i * 4)) * 3.0f) << "Post-state float4 z mismatch at index " << i;
        EXPECT_FLOAT_EQ(post_float4[i].w, (4.0f + (i * 4)) * 3.0f) << "Post-state float4 w mismatch at index " << i;
    }
}
