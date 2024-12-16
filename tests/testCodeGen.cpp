#include <gtest/gtest.h>
#include "CodeGen.hh"
#include "KernelManager.hh"
#include "Tracer.hh"

using namespace hip_intercept;

TEST(CodeGenTest, BasicCodeGeneration) {
    // Create a simple kernel
    std::string kernel_source = R"(
        __global__ void vectorAdd(float* a, float* b, float* c, int n) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i < n) {
                c[i] = a[i] + b[i];
            }
        }
    )";

    // Setup KernelManager
    KernelManager kernel_manager;
    kernel_manager.addFromModuleSource(kernel_source);

    // Create a mock trace
    Trace trace;
    KernelExecution exec;
    exec.kernel_name = "vectorAdd";
    exec.grid_dim = dim3(1, 1, 1);
    exec.block_dim = dim3(256, 1, 1);
    exec.shared_mem = 0;
    
    // Add mock memory states
    size_t array_size = 1024 * sizeof(float);
    exec.pre_state.push_back(MemoryState(array_size));  // a
    exec.pre_state.push_back(MemoryState(array_size));  // b
    exec.pre_state.push_back(MemoryState(array_size));  // c
    
    trace.kernel_executions.push_back(exec);

    // Generate code
    CodeGen code_gen(trace, kernel_manager);
    std::string generated_code = code_gen.generateReproducer();

    // Basic verification
    EXPECT_TRUE(generated_code.find("hipMalloc") != std::string::npos);
    EXPECT_TRUE(generated_code.find("vectorAdd<<<") != std::string::npos);
    EXPECT_TRUE(generated_code.find("dim3 grid") != std::string::npos);
    EXPECT_TRUE(generated_code.find("dim3 block") != std::string::npos);
    EXPECT_TRUE(generated_code.find("hipFree") != std::string::npos);
    
    // Verify kernel parameters
    EXPECT_TRUE(generated_code.find("float* arg_0_vectorAdd") != std::string::npos);
    EXPECT_TRUE(generated_code.find("float* arg_1_vectorAdd") != std::string::npos);
    EXPECT_TRUE(generated_code.find("float* arg_2_vectorAdd") != std::string::npos);
    
    // Verify memory operations
    EXPECT_TRUE(generated_code.find("hipMemcpy") != std::string::npos);
    EXPECT_TRUE(generated_code.find("hipMemcpyHostToDevice") != std::string::npos);
}

TEST(CodeGenTest, KernelWithScalarArguments) {
    std::string kernel_source = R"(
        __global__ void scalarKernel(float* out, int scalar1, float scalar2) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            out[i] = scalar1 * scalar2;
        }
    )";

    KernelManager kernel_manager;
    kernel_manager.addFromModuleSource(kernel_source);

    Trace trace;
    KernelExecution exec;
    exec.kernel_name = "scalarKernel";
    exec.grid_dim = dim3(1, 1, 1);
    exec.block_dim = dim3(128, 1, 1);
    exec.shared_mem = 0;
    
    size_t array_size = 128 * sizeof(float);
    exec.pre_state.push_back(MemoryState(array_size));
    
    trace.kernel_executions.push_back(exec);

    CodeGen code_gen(trace, kernel_manager);
    std::string generated_code = code_gen.generateReproducer();

    // Verify scalar parameter declarations
    EXPECT_TRUE(generated_code.find("int arg_1_scalarKernel;") != std::string::npos);
    EXPECT_TRUE(generated_code.find("float arg_2_scalarKernel;") != std::string::npos);
}

TEST(CodeGenTest, MultipleKernelLaunches) {
    std::string kernel_source = R"(
        __global__ void simpleKernel(float* data) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            data[i] *= 2.0f;
        }
    )";

    KernelManager kernel_manager;
    kernel_manager.addFromModuleSource(kernel_source);

    Trace trace;
    
    // First kernel execution
    KernelExecution exec1;
    exec1.kernel_name = "simpleKernel";
    exec1.grid_dim = dim3(1, 1, 1);
    exec1.block_dim = dim3(64, 1, 1);
    exec1.shared_mem = 0;
    exec1.pre_state.push_back(MemoryState(64 * sizeof(float)));
    
    // Second kernel execution
    KernelExecution exec2 = exec1;  // Same kernel, different launch config
    exec2.block_dim = dim3(128, 1, 1);
    
    trace.kernel_executions.push_back(exec1);
    trace.kernel_executions.push_back(exec2);

    CodeGen code_gen(trace, kernel_manager);
    std::string generated_code = code_gen.generateReproducer();

    // Verify multiple kernel launches
    size_t first_launch = generated_code.find("simpleKernel<<<");
    size_t second_launch = generated_code.find("simpleKernel<<<", first_launch + 1);
    EXPECT_NE(first_launch, std::string::npos);
    EXPECT_NE(second_launch, std::string::npos);
    EXPECT_NE(first_launch, second_launch);
}

TEST(CodeGenTest, GenerateAndCompile) {
    std::string kernel_source = R"(
        __global__ void simpleKernel(float* data, int n) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i < n) {
                data[i] *= 2.0f;
            }
        }
    )";

    KernelManager kernel_manager;
    kernel_manager.addFromModuleSource(kernel_source);

    Trace trace;
    KernelExecution exec;
    exec.kernel_name = "simpleKernel";
    exec.grid_dim = dim3(1, 1, 1);
    exec.block_dim = dim3(64, 1, 1);
    exec.shared_mem = 0;
    
    // Create sample data for the float array
    size_t array_size = 64 * sizeof(float);
    std::vector<float> sample_data(64, 1.0f);
    for (size_t i = 0; i < sample_data.size(); i++) {
        sample_data[i] = static_cast<float>(i);  // More varied test data
    }
    exec.pre_state.push_back(MemoryState(reinterpret_cast<const char*>(sample_data.data()), array_size));
    
    // Add scalar argument
    int n_value = 64;
    exec.pre_state.push_back(MemoryState(reinterpret_cast<const char*>(&n_value), sizeof(int)));
    exec.arg_sizes.push_back(array_size);
    exec.arg_sizes.push_back(sizeof(int));
    
    trace.kernel_executions.push_back(exec);

    // Create a temporary directory for test outputs
    std::string test_dir = "/tmp/hip_test_" + std::to_string(
        std::chrono::system_clock::now().time_since_epoch().count());
    std::filesystem::create_directories(test_dir);

    CodeGen code_gen(trace, kernel_manager);
    
    // Test file generation
    std::string generated_file = code_gen.generateFile(test_dir);
    EXPECT_TRUE(std::filesystem::exists(generated_file));
    
    // Print generated code for debugging
    std::cout << "\nGenerated code:\n" << std::ifstream(generated_file).rdbuf() << std::endl;
    
    // Test compilation
    EXPECT_TRUE(code_gen.compileFile(generated_file, test_dir));
    
    // Verify the executable was created
    std::string expected_executable = 
        (std::filesystem::path(test_dir) / 
         std::filesystem::path(generated_file).stem()).string();
    EXPECT_TRUE(std::filesystem::exists(expected_executable));
    
    // Cleanup
    std::filesystem::remove_all(test_dir);
}

TEST(CodeGenTest, GenerateAndCompileInvalidKernel) {
    std::string kernel_source = R"(
        __global__ void invalidKernel(float* data) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            undefined_function();  // This should cause a compilation error
        }
    )";

    KernelManager kernel_manager;
    kernel_manager.addFromModuleSource(kernel_source);

    Trace trace;
    KernelExecution exec;
    exec.kernel_name = "invalidKernel";
    exec.grid_dim = dim3(1, 1, 1);
    exec.block_dim = dim3(64, 1, 1);
    exec.pre_state.push_back(MemoryState(64 * sizeof(float)));
    
    trace.kernel_executions.push_back(exec);

    CodeGen code_gen(trace, kernel_manager);
    EXPECT_FALSE(code_gen.generateAndCompile("/tmp"));
}
