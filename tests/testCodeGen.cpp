#include <gtest/gtest.h>
#include "CodeGen.hh"
#include "KernelManager.hh"
#include "Tracer.hh"
#include "data/Kernels.hh"
#include <fstream>

using namespace hip_intercept;

// Helper function to save and print generated code
void saveAndPrintCode(const std::string& test_name, const std::string& generated_code) {
    std::string test_output = "/tmp/test_" + test_name + ".cpp";
    std::ofstream out(test_output);
    out << generated_code;
    out.close();
    
    std::cout << "\nGenerated code for " << test_name << " saved to: " << test_output << "\n";
    std::cout << "Generated code:\n" << generated_code << std::endl;
}

TEST(CodeGenTest, BasicCodeGeneration) {
    // Load the trace file generated during build
    std::string trace_file = std::string(CMAKE_BINARY_DIR) + "/tests/test_kernels-0.trace";
    Tracer tracer(trace_file);
    
    // Get the first vectorAdd kernel execution from the trace
    const auto& trace = tracer.instance().trace_;
    const auto& kernel_manager = tracer.getKernelManager();
    std::cout << trace;
    
    // Find the first vectorAdd execution
    auto it = std::find_if(trace.kernel_executions.begin(), trace.kernel_executions.end(),
        [](const KernelExecution& exec) { 
            std::cout << "Kernel name: " << exec.kernel_name << std::endl;
            return exec.kernel_name == "vectorAdd"; 
        });
    
    ASSERT_NE(it, trace.kernel_executions.end()) << "vectorAdd kernel execution not found in trace";
    
    // Generate code using the traced execution
    CodeGen code_gen(trace, kernel_manager);
    std::string generated_code = code_gen.generateReproducer();
    saveAndPrintCode("basic_codegen", generated_code);
    
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
    std::string kernel_source = test_kernels::kernel_strings::scalar_kernel;

    KernelManager kernel_manager;
    kernel_manager.addFromModuleSource(kernel_source);

    Trace trace;
    KernelExecution exec;
    exec.kernel_name = "scalarKernel";
    exec.grid_dim = dim3(1, 1, 1);
    exec.block_dim = dim3(128, 1, 1);
    exec.shared_mem = 0;
    
    // Add memory states for all arguments
    size_t array_size = 128 * sizeof(float);
    exec.pre_state.push_back(MemoryState(array_size));  // out array
    
    // Add scalar argument values
    int scalar1_value = 42;
    float scalar2_value = 3.14f;
    exec.pre_state.push_back(MemoryState(reinterpret_cast<const char*>(&scalar1_value), sizeof(int)));
    exec.pre_state.push_back(MemoryState(reinterpret_cast<const char*>(&scalar2_value), sizeof(float)));
    
    // Add argument sizes
    exec.arg_sizes.push_back(array_size);
    exec.arg_sizes.push_back(sizeof(int));
    exec.arg_sizes.push_back(sizeof(float));
    
    trace.kernel_executions.push_back(exec);

    CodeGen code_gen(trace, kernel_manager);
    std::string generated_code = code_gen.generateReproducer();
    saveAndPrintCode("scalar_arguments", generated_code);

    // Verify scalar parameter declarations and initialization
    EXPECT_TRUE(generated_code.find("int arg_1_scalarKernel;") != std::string::npos);
    EXPECT_TRUE(generated_code.find("float arg_2_scalarKernel;") != std::string::npos);
    EXPECT_TRUE(generated_code.find("memcpy(&arg_1_scalarKernel, trace_data_1,") != std::string::npos);
    EXPECT_TRUE(generated_code.find("memcpy(&arg_2_scalarKernel, trace_data_2,") != std::string::npos);
}

TEST(CodeGenTest, MultipleKernelLaunches) {
    std::string kernel_source = test_kernels::kernel_strings::simple_kernel;

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
    saveAndPrintCode("multiple_launches", generated_code);

    // Verify multiple kernel launches
    size_t first_launch = generated_code.find("simpleKernel<<<");
    size_t second_launch = generated_code.find("simpleKernel<<<", first_launch + 1);
    EXPECT_NE(first_launch, std::string::npos);
    EXPECT_NE(second_launch, std::string::npos);
    EXPECT_NE(first_launch, second_launch);
}

TEST(CodeGenTest, GenerateAndCompile) {
    std::string kernel_source = test_kernels::kernel_strings::simple_kernel_with_n;

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
        sample_data[i] = static_cast<float>(i);
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
    
    // Generate and compile the code
    std::string generated_file = code_gen.generateFile(test_dir);
    EXPECT_TRUE(std::filesystem::exists(generated_file));

    std::cout << "\nGenerated code:\n" << std::ifstream(generated_file).rdbuf() << std::endl;

    EXPECT_TRUE(code_gen.compileFile(generated_file, test_dir));
    
    // Get path to the compiled binary
    std::string executable = 
        (std::filesystem::path(test_dir) / 
         std::filesystem::path(generated_file).stem()).string();
    EXPECT_TRUE(std::filesystem::exists(executable));
    
    // Run the compiled binary
    std::string cmd = executable + " 2>&1";
    FILE* pipe = popen(cmd.c_str(), "r");
    ASSERT_TRUE(pipe != nullptr);
    
    // Read the output
    char buffer[128];
    std::string output;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
    
    // Check the return code
    int status = pclose(pipe);
    EXPECT_EQ(WEXITSTATUS(status), 0) << "Program failed with output:\n" << output;
    
    // Cleanup
    std::filesystem::remove_all(test_dir);

    std::string generated_code = code_gen.generateReproducer();
    saveAndPrintCode("generate_and_compile", generated_code);
}

TEST(CodeGenTest, GenerateAndCompileInvalidKernel) {
    std::string kernel_source = test_kernels::kernel_strings::invalid_kernel;

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
    std::string generated_code = code_gen.generateReproducer();
    saveAndPrintCode("invalid_kernel", generated_code);

    EXPECT_FALSE(code_gen.generateAndCompile("/tmp"));
}
