#include <gtest/gtest.h>
#include "../Comparator.hh"
#include <sstream>
#include <filesystem>

class ComparatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup code if needed
    }
};

TEST_F(ComparatorTest, IdenticalTracesCompareEqual) {
    std::string trace_file = std::string(CMAKE_BINARY_DIR) + "/tests/test_kernels-0.trace";

    // Create temporary trace files
    {
        Tracer tracer1(trace_file);
        Tracer tracer2(trace_file);
        
        // Record identical kernel executions
        KernelExecution kernel;
        kernel.kernel_name = "testKernel";
        kernel.grid_dim = {1, 1, 1};
        kernel.block_dim = {256, 1, 1};
        kernel.shared_mem = 0;
        
        // Create and initialize memory states
        auto pre_state = std::make_shared<MemoryState>(1024);  // 1KB pre state
        auto post_state = std::make_shared<MemoryState>(2048); // 2KB post state
        
        // Fill with recognizable patterns
        {
            auto pre_data = pre_state->getData();
            auto post_data = post_state->getData();
            for (size_t i = 0; i < 1024; i++) {
                pre_data.get()[i] = static_cast<char>(i & 0xFF);
            }
            for (size_t i = 0; i < 2048; i++) {
                post_data.get()[i] = static_cast<char>((i * 2) & 0xFF);
            }
        }
        
        kernel.pre_state = pre_state;
        kernel.post_state = post_state;
        
        tracer1.trace_.addOperation(std::make_shared<KernelExecution>(kernel));
        tracer2.trace_.addOperation(std::make_shared<KernelExecution>(kernel));

        tracer1.finalizeTrace();
        tracer2.finalizeTrace();
    }

    // Compare the traces
    std::stringstream output;
    Comparator comparator(trace_file, trace_file);
    comparator.compare(output);
    
    // Verify no differences were reported
    std::string result = output.str();
    EXPECT_EQ(result.find("Traces differ"), std::string::npos);
}

TEST_F(ComparatorTest, DifferentKernelConfigsReported) {
    std::string trace_file = std::string(CMAKE_BINARY_DIR) + "/tests/test_kernels-0.trace";
    std::string new_trace_file = std::string(CMAKE_BINARY_DIR) + "/tests/test_kernels-1.trace";
    int num_ops = 0;

    // Create temporary trace files
    {
        // First ensure any existing files are removed
        std::filesystem::remove(trace_file);
        std::filesystem::remove(new_trace_file);
        
        Tracer tracer1(trace_file);
        Tracer tracer2(trace_file);
        num_ops = tracer1.getNumOperations();
        
        // Record kernel executions with different configurations
        KernelExecution kernel1;
        kernel1.kernel_name = "testKernel";
        kernel1.grid_dim = {1, 1, 1};
        kernel1.block_dim = {256, 1, 1};
        kernel1.shared_mem = 0;
        
        KernelExecution kernel2;
        kernel2.kernel_name = "testKernel";
        kernel2.grid_dim = {2, 1, 1};  // Different grid dimension
        kernel2.block_dim = {256, 1, 1};
        kernel2.shared_mem = 0;

        // Initialize memory states for kernel1
        auto pre_state1 = std::make_shared<MemoryState>(1024);  // 1KB pre state
        auto post_state1 = std::make_shared<MemoryState>(2048); // 2KB post state
        
        // Fill with recognizable patterns
        {
            auto pre_data = pre_state1->getData();
            auto post_data = post_state1->getData();
            for (size_t i = 0; i < 1024; i++) {
                pre_data.get()[i] = static_cast<char>(i & 0xFF);
            }
            for (size_t i = 0; i < 2048; i++) {
                post_data.get()[i] = static_cast<char>((i * 2) & 0xFF);
            }
        }
        
        kernel1.pre_state = pre_state1;
        kernel1.post_state = post_state1;

        // Initialize memory states for kernel2
        auto pre_state2 = std::make_shared<MemoryState>(1024);  // 1KB pre state
        auto post_state2 = std::make_shared<MemoryState>(2048); // 2KB post state
        
        // Fill with different patterns
        {
            auto pre_data = pre_state2->getData();
            auto post_data = post_state2->getData();
            for (size_t i = 0; i < 1024; i++) {
                pre_data.get()[i] = static_cast<char>((i * 3) & 0xFF);
            }
            for (size_t i = 0; i < 2048; i++) {
                post_data.get()[i] = static_cast<char>((i * 4) & 0xFF);
            }
        }
        
        kernel2.pre_state = pre_state2;
        kernel2.post_state = post_state2;

        tracer1.trace_.addOperation(std::make_shared<KernelExecution>(kernel1));

        tracer2.setFilePath(new_trace_file);
        tracer2.trace_.addOperation(std::make_shared<KernelExecution>(kernel2));

        tracer1.finalizeTrace();
        tracer2.finalizeTrace();
    }

    // Compare the traces
    std::stringstream output;
    Comparator comparator(trace_file, new_trace_file);
    // assert that number of executions increased by 1
    EXPECT_EQ(comparator.tracer1.getNumOperations(), num_ops + 1);
    EXPECT_EQ(comparator.tracer2.getNumOperations(), num_ops + 1);
    comparator.compare(output);
    
    // Verify differences were reported
    std::string result = output.str();
    EXPECT_NE(result.find("Traces differ"), std::string::npos);
    EXPECT_NE(result.find("Kernel(testKernel)"), std::string::npos);
}

TEST_F(ComparatorTest, DifferentMemoryOperationsReported) {
    std::string trace_file = std::string(CMAKE_BINARY_DIR) + "/tests/test_kernels-0.trace";
    std::string new_trace_file = std::string(CMAKE_BINARY_DIR) + "/tests/test_kernels-1.trace";

    // Create temporary trace files
    {
        // First ensure any existing files are removed
        std::filesystem::remove(trace_file);
        std::filesystem::remove(new_trace_file);
        
        Tracer tracer1(trace_file);
        Tracer tracer2(trace_file);
        
        // Record memory operations with different sizes
        MemoryOperation mem1;
        mem1.type = MemoryOpType::COPY;
        mem1.size = 1024;
        mem1.kind = hipMemcpyHostToDevice;
        
        MemoryOperation mem2;
        mem2.type = MemoryOpType::COPY;
        mem2.size = 2048;  // Different size
        mem2.kind = hipMemcpyHostToDevice;
        
        // Initialize memory states for mem1
        auto pre_state1 = std::make_shared<MemoryState>(1024);  // 1KB pre state
        auto post_state1 = std::make_shared<MemoryState>(1024); // Same size post state
        
        // Fill with recognizable patterns
        {
            auto pre_data = pre_state1->getData();
            auto post_data = post_state1->getData();
            for (size_t i = 0; i < 1024; i++) {
                pre_data.get()[i] = static_cast<char>(i & 0xFF);
                post_data.get()[i] = static_cast<char>((i * 2) & 0xFF);
            }
        }
        
        mem1.pre_state = pre_state1;
        mem1.post_state = post_state1;

        // Initialize memory states for mem2
        auto pre_state2 = std::make_shared<MemoryState>(2048);  // 2KB pre state
        auto post_state2 = std::make_shared<MemoryState>(2048); // Same size post state
        
        // Fill with different patterns
        {
            auto pre_data = pre_state2->getData();
            auto post_data = post_state2->getData();
            for (size_t i = 0; i < 2048; i++) {
                pre_data.get()[i] = static_cast<char>((i * 3) & 0xFF);
                post_data.get()[i] = static_cast<char>((i * 4) & 0xFF);
            }
        }
        
        mem2.pre_state = pre_state2;
        mem2.post_state = post_state2;
        
        tracer1.trace_.addOperation(std::make_shared<MemoryOperation>(mem1));

        tracer2.setFilePath(new_trace_file);
        tracer2.trace_.addOperation(std::make_shared<MemoryOperation>(mem2));

        tracer1.finalizeTrace();
        tracer2.finalizeTrace();
    }

    // Compare the traces
    std::stringstream output;
    Comparator comparator(trace_file, new_trace_file);
    comparator.compare(output);
    
    // Verify differences were reported
    std::string result = output.str();
    EXPECT_NE(result.find("Traces differ"), std::string::npos);
    EXPECT_NE(result.find("hipMemcpyAsync call"), std::string::npos);
}

TEST_F(ComparatorTest, VerifyStateCapture) {
    std::string trace_file = std::string(CMAKE_BINARY_DIR) + "/tests/data_verification-0.trace";
    
    // Load the trace
    Tracer tracer(trace_file);
    std::cout << tracer;
    
    // We expect 13 operations:
    // 4 mallocs (scalar, vec4, vec2, float4)
    // 4 H2D copies
    // 1 kernel
    // 4 D2H copies
    ASSERT_EQ(tracer.getNumOperations(), 13) << "Expected 13 operations in trace";
    
    // Get operations for the kernel and surrounding memory operations
    auto kernel_op = tracer.getOperation(8); // kernel should be in the middle
    
    // Verify kernel execution
    ASSERT_TRUE(kernel_op->isKernel()) << "Operation is not a kernel";
    auto kernel = static_cast<const KernelExecution*>(kernel_op.get());
    EXPECT_EQ(kernel->kernel_name, "complexDataKernel") << "Kernel name mismatch";
    
    // Verify kernel's pre and post states
    ASSERT_TRUE(kernel->pre_state != nullptr) << "Pre-state is null";
    ASSERT_TRUE(kernel->post_state != nullptr) << "Post-state is null";
    
    // Calculate total size for all arrays
    size_t N = 3;  // Reduced from 1024 to 3
    size_t total_size = N * (sizeof(float) + sizeof(float4) + 
                            sizeof(HIP_vector_type<float, 4>) + 
                            sizeof(HIP_vector_type<float, 2>));
    
    std::cout << "Pre-state size: " << kernel->pre_state->total_size << std::endl;
    std::cout << "Post-state size: " << kernel->post_state->total_size << std::endl;
    
    EXPECT_EQ(kernel->pre_state->total_size, total_size);
    EXPECT_EQ(kernel->post_state->total_size, total_size);
    
    // Get the data for comparison
    auto pre_data = kernel->pre_state->getData();
    auto post_data = kernel->post_state->getData();
    float* pre_scalar = reinterpret_cast<float*>(pre_data.get());
    float* post_scalar = reinterpret_cast<float*>(post_data.get());
    
    auto* pre_vec4 = reinterpret_cast<HIP_vector_type<float, 4>*>(pre_scalar + N);
    auto* post_vec4 = reinterpret_cast<HIP_vector_type<float, 4>*>(post_scalar + N);
    
    auto* pre_vec2 = reinterpret_cast<HIP_vector_type<float, 2>*>(pre_vec4 + N);
    auto* post_vec2 = reinterpret_cast<HIP_vector_type<float, 2>*>(post_vec4 + N);
    
    auto* pre_float4 = reinterpret_cast<float4*>(pre_vec2 + N);
    auto* post_float4 = reinterpret_cast<float4*>(post_vec2 + N);
    
    // Print all values for debugging since we only have 3 elements
    std::cout << "\nScalar array values:" << std::endl;
    for (size_t i = 0; i < N; i++) {
        std::cout << "pre[" << i << "]=" << pre_scalar[i] 
                 << ", post[" << i << "]=" << post_scalar[i] << std::endl;
    }
    
    std::cout << "\nVector4 array values:" << std::endl;
    for (size_t i = 0; i < N; i++) {
        std::cout << "pre[" << i << "]=(" << pre_vec4[i].x << "," << pre_vec4[i].y << ","
                 << pre_vec4[i].z << "," << pre_vec4[i].w << ")" << std::endl;
        std::cout << "post[" << i << "]=(" << post_vec4[i].x << "," << post_vec4[i].y << ","
                 << post_vec4[i].z << "," << post_vec4[i].w << ")" << std::endl;
    }
    
    // Verify scalar array
    for (size_t i = 0; i < N; i++) {
        // Verify pre-state matches initialization pattern
        EXPECT_FLOAT_EQ(pre_scalar[i], 1.0f + i)
            << "Pre-state scalar mismatch at index " << i;
        
        // Verify post-state matches expected computation
        float expected = pre_scalar[i] * 2.0f + 5; // scalar2 = 2.0f, scalar1 = 5
        EXPECT_FLOAT_EQ(post_scalar[i], expected)
            << "Post-state scalar mismatch at index " << i;
    }
    
    // Verify vector4 array
    for (size_t i = 0; i < N; i++) {
        // Verify pre-state matches initialization pattern
        EXPECT_FLOAT_EQ(pre_vec4[i].x, 1.0f + (i * 4))
            << "Pre-state vec4 x mismatch at index " << i;
        EXPECT_FLOAT_EQ(pre_vec4[i].y, 2.0f + (i * 4))
            << "Pre-state vec4 y mismatch at index " << i;
        EXPECT_FLOAT_EQ(pre_vec4[i].z, 3.0f + (i * 4))
            << "Pre-state vec4 z mismatch at index " << i;
        EXPECT_FLOAT_EQ(pre_vec4[i].w, 4.0f + (i * 4))
            << "Pre-state vec4 w mismatch at index " << i;
        
        // Verify post-state matches expected computation (multiply by scalar2 = 2.0f)
        EXPECT_FLOAT_EQ(post_vec4[i].x, pre_vec4[i].x * 2.0f)
            << "Post-state vec4 x mismatch at index " << i;
        EXPECT_FLOAT_EQ(post_vec4[i].y, pre_vec4[i].y * 2.0f)
            << "Post-state vec4 y mismatch at index " << i;
        EXPECT_FLOAT_EQ(post_vec4[i].z, pre_vec4[i].z * 2.0f)
            << "Post-state vec4 z mismatch at index " << i;
        EXPECT_FLOAT_EQ(post_vec4[i].w, pre_vec4[i].w * 2.0f)
            << "Post-state vec4 w mismatch at index " << i;
    }
    
    // Verify vector2 array
    for (size_t i = 0; i < N; i++) {
        // Verify pre-state matches initialization pattern
        EXPECT_FLOAT_EQ(pre_vec2[i].x, 1.0f + (i * 2))
            << "Pre-state vec2 x mismatch at index " << i;
        EXPECT_FLOAT_EQ(pre_vec2[i].y, 2.0f + (i * 2))
            << "Pre-state vec2 y mismatch at index " << i;
        
        // Verify post-state matches expected computation (add scalar3 = 1.5)
        EXPECT_FLOAT_EQ(post_vec2[i].x, pre_vec2[i].x + 1.5)
            << "Post-state vec2 x mismatch at index " << i;
        EXPECT_FLOAT_EQ(post_vec2[i].y, pre_vec2[i].y + 1.5)
            << "Post-state vec2 y mismatch at index " << i;
    }
    
    // Verify float4 array
    for (size_t i = 0; i < N; i++) {
        // Verify pre-state matches initialization pattern
        EXPECT_FLOAT_EQ(pre_float4[i].x, 1.0f + (i * 4))
            << "Pre-state float4 x mismatch at index " << i;
        EXPECT_FLOAT_EQ(pre_float4[i].y, 2.0f + (i * 4))
            << "Pre-state float4 y mismatch at index " << i;
        EXPECT_FLOAT_EQ(pre_float4[i].z, 3.0f + (i * 4))
            << "Pre-state float4 z mismatch at index " << i;
        EXPECT_FLOAT_EQ(pre_float4[i].w, 4.0f + (i * 4))
            << "Pre-state float4 w mismatch at index " << i;
        
        // Verify post-state matches expected computation (multiply by uint_val = 3)
        EXPECT_FLOAT_EQ(post_float4[i].x, pre_float4[i].x * 3)
            << "Post-state float4 x mismatch at index " << i;
        EXPECT_FLOAT_EQ(post_float4[i].y, pre_float4[i].y * 3)
            << "Post-state float4 y mismatch at index " << i;
        EXPECT_FLOAT_EQ(post_float4[i].z, pre_float4[i].z * 3)
            << "Post-state float4 z mismatch at index " << i;
        EXPECT_FLOAT_EQ(post_float4[i].w, pre_float4[i].w * 3)
            << "Post-state float4 w mismatch at index " << i;
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
