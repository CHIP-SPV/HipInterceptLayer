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
        auto pre_state = MemoryState(1024);  // 1KB pre state
        auto post_state = MemoryState(2048); // 2KB post state
        
        // Fill with recognizable patterns
        {
            auto pre_data = pre_state.getData();
            auto post_data = post_state.getData();
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
    auto result = comparator.compare(output);
    EXPECT_TRUE(result);
}

TEST_F(ComparatorTest, DifferentKernelConfigsReported) {
    std::string trace_file = std::string(CMAKE_BINARY_DIR) + "/tests/test_kernels-0.trace";
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

        // Initialize memory states for kernel1
        auto pre_state1 = MemoryState(1024);  // 1KB pre state
        auto post_state1 = MemoryState(2048); // 2KB post state
        
        // Fill with recognizable patterns
        {
            auto pre_data = pre_state1.getData();
            auto post_data = post_state1.getData();
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
        auto pre_state2 = MemoryState(1024);  // 1KB pre state
        auto post_state2 = MemoryState(2048); // 2KB post state
        
        // Fill with different patterns
        {
            auto pre_data = pre_state2.getData();
            auto post_data = post_state2.getData();
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
        tracer2.trace_.addOperation(std::make_shared<KernelExecution>(kernel2));

        tracer1.finalizeTrace(trace_file + "-1");
        tracer2.finalizeTrace(trace_file + "-2");
    }

    // Compare the traces
    std::stringstream output;
    Comparator comparator(trace_file + "-1", trace_file + "-2");
    // assert that number of executions increased by 1
    EXPECT_EQ(comparator.tracer1.getNumOperations(), num_ops + 1);
    EXPECT_EQ(comparator.tracer2.getNumOperations(), num_ops + 1);
    auto result = comparator.compare(output);
    EXPECT_FALSE(result);
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
        
        // Initialize memory states for mem1
        auto pre_state1 = MemoryState(1024);  // 1KB pre state
        auto post_state1 = MemoryState(1024); // Same size post state
        
        // Fill with recognizable patterns
        {
            auto pre_data = pre_state1.getData();
            auto post_data = post_state1.getData();
            for (size_t i = 0; i < 1024; i++) {
                pre_data.get()[i] = static_cast<char>(i & 0xFF);
                post_data.get()[i] = static_cast<char>((i * 2) & 0xFF);
            }
        }
        
        mem1.pre_state = pre_state1;
        mem1.post_state = post_state1;

        // Initialize memory states for mem2
        auto pre_state2 = MemoryState(2048);  // 2KB pre state
        auto post_state2 = MemoryState(2048); // Same size post state
        
        // Fill with different patterns
        {
            auto pre_data = pre_state2.getData();
            auto post_data = post_state2.getData();
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
    auto result = comparator.compare(output);
    EXPECT_FALSE(result);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
