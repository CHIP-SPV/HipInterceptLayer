#include <gtest/gtest.h>
#include "CodeGen.hh"
#include "KernelManager.hh"
#include "Tracer.hh"
#include "data/Kernels.hh"
#include <fstream>

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
    // Generate code using the traced execution
    CodeGen code_gen(trace_file);
    auto idx = code_gen.tracer.getOperationsIdxByName("vectorAdd")[0];
    std::string generated_code = code_gen.generateReproducer(idx);
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
    std::string trace_file = std::string(CMAKE_BINARY_DIR) + "/tests/test_kernels-0.trace";
    CodeGen code_gen(trace_file);
    
    auto idx = code_gen.tracer.getOperationsIdxByName("scalarKernel")[0];
    std::string generated_code = code_gen.generateReproducer(idx);
    saveAndPrintCode("scalar_arguments", generated_code);
    
    // Verify scalar parameter declarations and initialization
    EXPECT_TRUE(generated_code.find("int arg_1_scalarKernel;") != std::string::npos);
    EXPECT_TRUE(generated_code.find("float arg_2_scalarKernel;") != std::string::npos);
    
    // Verify data loading
    EXPECT_TRUE(generated_code.find("loadTraceData(trace_file,") != std::string::npos);
    EXPECT_TRUE(generated_code.find("sizeof(int), &arg_1_scalarKernel)") != std::string::npos);
    EXPECT_TRUE(generated_code.find("sizeof(float), &arg_2_scalarKernel)") != std::string::npos);
}

TEST(CodeGenTest, GenerateAndCompile) {
    std::string trace_file = std::string(CMAKE_BINARY_DIR) + "/tests/test_kernels-0.trace";
    CodeGen code_gen(trace_file);

    // Create a temporary directory for test outputs
    std::string test_dir = "/tmp/hip_test_" + std::to_string(
        std::chrono::system_clock::now().time_since_epoch().count());
    std::filesystem::create_directories(test_dir);

    
    // Generate and compile the code
    auto idx = code_gen.tracer.getOperationsIdxByName("vectorAdd")[0];
    std::string generated_file = code_gen.generateFile(idx, test_dir);
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

    std::string generated_code = code_gen.generateReproducer(idx);
    saveAndPrintCode("generate_and_compile", generated_code);
}

TEST(CodeGenTest, GenerateAndCompileInvalidKernel) {
    std::string trace_file = std::string(CMAKE_BINARY_DIR) + "/tests/test_kernels-0.trace";
    CodeGen code_gen(trace_file);
    auto idx = code_gen.tracer.getOperationsIdxByName("vectorAdd")[0];
    std::string generated_code = code_gen.generateReproducer(idx);
    saveAndPrintCode("invalid_kernel", generated_code);

    EXPECT_FALSE(code_gen.generateAndCompile(0, "/tmp"));
}

TEST(CodeGenTest, KernelDeclarationsIncluded) {
    std::string trace_file = std::string(CMAKE_BINARY_DIR) + "/tests/test_kernels-0.trace";
    CodeGen code_gen(trace_file);
    auto idx = code_gen.tracer.getOperationsIdxByName("vectorAdd")[0];
    std::string generated_code = code_gen.generateReproducer(idx);
    saveAndPrintCode("kernel_declarations", generated_code);
    
    // Verify kernel declarations are present
    const Kernel& kernel = code_gen.tracer.getKernelManager().getKernelByName("vectorAdd");
    std::string kernel_source = kernel.getSource();
    EXPECT_TRUE(generated_code.find(kernel_source) != std::string::npos) 
        << "Kernel source not found in generated code";
    
    // Verify kernel can be called with correct arguments
    EXPECT_TRUE(generated_code.find("vectorAdd<<<grid, block, 0>>>(arg_0_vectorAdd_d, "
                                  "arg_1_vectorAdd_d, arg_2_vectorAdd_d, arg_3_vectorAdd)") 
                != std::string::npos)
        << "Kernel call not found or incorrect in generated code";
}

/*

TEST(CodeGenTest, KernelWithoutSource) {
    std::string trace_file = std::string(CMAKE_BINARY_DIR) + "/tests/test_kernels-0.trace";
    Tracer tracer(trace_file);
    const auto& kernel_manager = tracer.getKernelManager();
    
    // Create a mock kernel without source
    Kernel mock_kernel("mockKernel");
    mock_kernel.addArgument(KernelArgument("float*", "input"));
    mock_kernel.addArgument(KernelArgument("float*", "output"));
    mock_kernel.addArgument(KernelArgument("int", "size"));
    
    // Add the mock kernel to the trace
    KernelExecution mock_exec;
    mock_exec.kernel_name = "mockKernel";
    mock_exec.grid_dim = {1, 1, 1};
    mock_exec.block_dim = {256, 1, 1};
    mock_exec.shared_mem = 0;
    
    Trace modified_trace = tracer.instance().trace_;
    modified_trace.kernel_executions.push_back(mock_exec);
    
    // Create CodeGen with the modified trace
    CodeGen code_gen(trace_file, kernel_manager);
    code_gen.trace_ = modified_trace;  // Note: This requires making trace_ protected instead of private
    
    std::string generated_code = code_gen.generateReproducer(
        modified_trace.kernel_executions.size() - 1);
    saveAndPrintCode("kernel_without_source", generated_code);
    
    // Verify generated kernel declaration
    std::string expected_declaration = 
        "__global__ void mockKernel(float* input, float* output, int size) {\n"
        "    // TODO: Original kernel source not available\n"
        "    // This is a placeholder implementation\n"
        "}\n";
    
    EXPECT_TRUE(generated_code.find(expected_declaration) != std::string::npos)
        << "Generated code does not contain expected kernel declaration";
}

*/
