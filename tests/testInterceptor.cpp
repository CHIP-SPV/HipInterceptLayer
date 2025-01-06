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

// Test to verify vectorAdd kernel behavior from trace
TEST_F(InterceptorTest, CompareVectorAddTraceWithSource) {
    // Load and verify the trace file
    std::string trace_file = std::string(CMAKE_BINARY_DIR) + "/tests/vectorAdd-0.trace";
    Tracer tracer(trace_file);
    
    // Find the kernel operation by name
    auto kernel_indices = tracer.getOperationsIdxByName("vectorIncrementKernel");
    ASSERT_FALSE(kernel_indices.empty()) << "No vectorIncrementKernel found in trace";
    
    // Get the first instance of the kernel
    auto kernel_op = std::dynamic_pointer_cast<KernelExecution>(tracer.getOperation(kernel_indices[0]));
    ASSERT_NE(kernel_op, nullptr);
    std::cout << "Found kernel operation at index " << kernel_indices[0] << std::endl;
    
    // Add debug output for kernel operation details
    std::cout << "Kernel operation details:" << std::endl;
    std::cout << "  Name: " << kernel_op->kernel_name << std::endl;
    std::cout << "  Grid dim: " << kernel_op->grid_dim.x << "," << kernel_op->grid_dim.y << "," << kernel_op->grid_dim.z << std::endl;
    std::cout << "  Block dim: " << kernel_op->block_dim.x << "," << kernel_op->block_dim.y << "," << kernel_op->block_dim.z << std::endl;
    std::cout << "  Shared mem: " << kernel_op->shared_mem << std::endl;
    std::cout << "  Number of arguments: " << kernel_op->arg_ptrs.size() << std::endl;
    std::cout << "  Number of scalar values: " << kernel_op->scalar_values.size() << std::endl;
    
    // Verify kernel launch parameters
    EXPECT_EQ(kernel_op->kernel_name, "vectorIncrementKernel");
    EXPECT_EQ(kernel_op->block_dim.x, 256);  // From vectorAdd.cpp
    EXPECT_EQ(kernel_op->block_dim.y, 1);
    EXPECT_EQ(kernel_op->block_dim.z, 1);
    EXPECT_EQ(kernel_op->shared_mem, 0);

    // Verify scalar values
    ASSERT_EQ(kernel_op->scalar_values.size(), 2) << "Expected 2 scalar values (float4 and float)";
    
    // First scalar value should be float4 (16 bytes)
    ASSERT_EQ(kernel_op->scalar_values[0].size(), 16) << "Expected float4 to be 16 bytes";
    const float4* input_vec4 = reinterpret_cast<const float4*>(kernel_op->scalar_values[0].data());
    std::cout << "Input float4: (" << input_vec4->x << ", " << input_vec4->y << ", " 
              << input_vec4->z << ", " << input_vec4->w << ")" << std::endl;
    
    // Second scalar value should be float (4 bytes)
    ASSERT_EQ(kernel_op->scalar_values[1].size(), 8) << "Expected float to be 8 bytes";
    const float* input_scalar = reinterpret_cast<const float*>(kernel_op->scalar_values[1].data());
    std::cout << "Input scalar: " << *input_scalar << std::endl;

    // Expected values from vectorAdd.cpp
    const float4 expected_input_vec4 = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    const float expected_input_scalar = 0.5f;

    // Verify input values
    EXPECT_FLOAT_EQ(input_vec4->x, expected_input_vec4.x);
    EXPECT_FLOAT_EQ(input_vec4->y, expected_input_vec4.y);
    EXPECT_FLOAT_EQ(input_vec4->z, expected_input_vec4.z);
    EXPECT_FLOAT_EQ(input_vec4->w, expected_input_vec4.w);
    EXPECT_FLOAT_EQ(*input_scalar, expected_input_scalar);

    // Pre-execution state
    auto pre_state = kernel_op->pre_state;
    std::cout << "Pre-state total_size: " << pre_state.total_size << std::endl;
    std::cout << "Pre-state chunks size: " << pre_state.chunks.size() << std::endl;
    ASSERT_GT(pre_state.total_size, 0);
    ASSERT_GT(pre_state.chunks.size(), 0) << "No chunks in pre_state";

    // Get raw data from the first chunk of pre_state
    const auto& pre_chunk = pre_state.chunks[0];
    const float4* pre_vec4 = reinterpret_cast<const float4*>(pre_chunk.data.get());
    const float* pre_scalar = reinterpret_cast<const float*>(pre_vec4 + 1024);  // N=1024

    // Print first few pre-state values
    std::cout << "\nFirst few pre-state values:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "  Index " << i << ": vec4=("
                  << pre_vec4[i].x << ", " 
                  << pre_vec4[i].y << ", "
                  << pre_vec4[i].z << ", "
                  << pre_vec4[i].w << "), "
                  << "scalar=" << pre_scalar[i] << std::endl;
    }

    // Verify all pre-state values (should be uninitialized or zero)
    for (int i = 0; i < 1024; i++) {
        // We don't verify exact values since they're uninitialized
        // but we can verify they exist and are accessible
        volatile float x = pre_vec4[i].x;
        volatile float y = pre_vec4[i].y;
        volatile float z = pre_vec4[i].z;
        volatile float w = pre_vec4[i].w;
        volatile float s = pre_scalar[i];
        (void)x; (void)y; (void)z; (void)w; (void)s;  // Prevent unused variable warnings
    }

    // Post-execution state
    auto post_state = kernel_op->post_state;
    std::cout << "Post-state total_size: " << post_state.total_size << std::endl;
    std::cout << "Post-state chunks size: " << post_state.chunks.size() << std::endl;
    ASSERT_GT(post_state.total_size, 0);
    ASSERT_GT(post_state.chunks.size(), 0) << "No chunks in post_state";

    // Get raw data from the first chunk of post_state
    const auto& post_chunk = post_state.chunks[0];
    const float4* output_vec4 = reinterpret_cast<const float4*>(post_chunk.data.get());
    const float* output_scalar = reinterpret_cast<const float*>(output_vec4 + 1024);  // N=1024

    // Expected output values
    const float4 expected_output_vec4 = make_float4(
        expected_input_vec4.x + expected_input_scalar,
        expected_input_vec4.y + expected_input_scalar,
        expected_input_vec4.z + expected_input_scalar,
        expected_input_vec4.w + expected_input_scalar
    );
    const float expected_output_scalar = expected_input_scalar * 2.0f;

    // Print first few output values
    std::cout << "\nFirst few post-state values:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "  Index " << i << ": vec4=("
                  << output_vec4[i].x << ", " 
                  << output_vec4[i].y << ", "
                  << output_vec4[i].z << ", "
                  << output_vec4[i].w << "), "
                  << "scalar=" << output_scalar[i] << std::endl;
    }

    std::cout << "\nExpected output vec4: ("
              << expected_output_vec4.x << ", "
              << expected_output_vec4.y << ", "
              << expected_output_vec4.z << ", "
              << expected_output_vec4.w << ")" << std::endl;
    std::cout << "Expected output scalar: " << expected_output_scalar << std::endl;

    // Verify all output values
    for (int i = 0; i < 1024; i++) {
        // Verify vec4 components
        EXPECT_FLOAT_EQ(output_vec4[i].x, expected_output_vec4.x)
            << "Vec4.x mismatch at index " << i;
        EXPECT_FLOAT_EQ(output_vec4[i].y, expected_output_vec4.y)
            << "Vec4.y mismatch at index " << i;
        EXPECT_FLOAT_EQ(output_vec4[i].z, expected_output_vec4.z)
            << "Vec4.z mismatch at index " << i;
        EXPECT_FLOAT_EQ(output_vec4[i].w, expected_output_vec4.w)
            << "Vec4.w mismatch at index " << i;

        // Verify scalar value
        EXPECT_FLOAT_EQ(output_scalar[i], expected_output_scalar)
            << "Scalar mismatch at index " << i;
    }
}

