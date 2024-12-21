#include <gtest/gtest.h>
#include "../Comparator.hh"
#include <sstream>

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
        
        tracer1.trace_.operations.push_back(std::make_unique<KernelExecution>(kernel));
        tracer2.trace_.operations.push_back(std::make_unique<KernelExecution>(kernel));
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

        tracer1.trace_.addOperation(std::make_unique<KernelExecution>(kernel1));

        tracer2.setFilePath(new_trace_file);
        tracer2.trace_.addOperation(std::make_unique<KernelExecution>(kernel2));
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
        
        tracer1.trace_.operations.push_back(std::make_unique<MemoryOperation>(mem1));

        tracer2.setFilePath(new_trace_file);
        tracer2.trace_.operations.push_back(std::make_unique<MemoryOperation>(mem2));
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
    
    // We expect 3 operations in order:
    // 0. hipMalloc
    // 1. hipMemcpy (Host to Device)
    // 2. Kernel execution (simpleKernel)
    // 3. hipMemcpy (Device to Host)
    ASSERT_EQ(tracer.getNumOperations(), 4);
    
    // Get operations
    auto op1 = tracer.getOperation(0);
    auto op2 = tracer.getOperation(1);
    auto op3 = tracer.getOperation(2);
    auto op4 = tracer.getOperation(3);
    
    // Verify first memcpy (Host to Device)
    ASSERT_TRUE(op1->isMemory());
    auto memcpy1 = static_cast<const MemoryOperation*>(op1.get());
    EXPECT_EQ(memcpy1->kind, hipMemcpyHostToDevice);
    
    // Verify kernel execution
    ASSERT_TRUE(op2->isKernel());
    auto kernel = static_cast<const KernelExecution*>(op2.get());
    EXPECT_EQ(kernel->kernel_name, "simpleKernel");
    
    // Verify kernel's pre and post states
    ASSERT_TRUE(kernel->pre_state != nullptr);
    ASSERT_TRUE(kernel->post_state != nullptr);
    EXPECT_EQ(kernel->pre_state->size, 1024 * sizeof(float));
    EXPECT_EQ(kernel->post_state->size, 1024 * sizeof(float));
    
    // Verify the data in pre_state matches input pattern
    float* pre_data = reinterpret_cast<float*>(kernel->pre_state->data.get());
    for (int i = 0; i < 1024; i++) {
        EXPECT_FLOAT_EQ(pre_data[i], 1.0f + i) 
            << "Pre-state mismatch at index " << i;
    }
    
    // Verify the data in post_state matches expected output
    float* post_data = reinterpret_cast<float*>(kernel->post_state->data.get());
    for (int i = 0; i < 1024; i++) {
        EXPECT_FLOAT_EQ(post_data[i], (1.0f + i) * 2.0f) 
            << "Post-state mismatch at index " << i;
    }
    
    // Verify final memcpy (Device to Host)
    ASSERT_TRUE(op4->isMemory());
    auto memcpy2 = static_cast<const MemoryOperation*>(op4.get());
    EXPECT_EQ(memcpy2->kind, hipMemcpyDeviceToHost);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
