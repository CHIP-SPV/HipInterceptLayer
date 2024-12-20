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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
