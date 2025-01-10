#include <gtest/gtest.h>
#include "../CodeGen.hh"
#include "KernelManager.hh"
#include "Tracer.hh"
#include "data/test_kernels.hh"
#include <fstream>
#include <filesystem>

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

TEST(CodeGenTest, ComputeNonbondedReproducer) {
    std::string trace_file = std::string(CMAKE_BINARY_DIR) + "/tests/computeNonbondedRepro-0.trace";
    
    // Run HIPInterceptCompare with --gen-repro to generate and execute the reproducer
    std::string temp_dir = "/tmp/compute_nonbonded_reproducer";
    std::filesystem::create_directories(temp_dir);
    
    auto cmd = "HIP_TRACE_LOCATION=" + std::string(CMAKE_BINARY_DIR) + "/tests " + std::string(CMAKE_BINARY_DIR) + "/HIPInterceptCompare " + trace_file + " --gen-repro 12";
    std::cout << "Running command: " << cmd << std::endl;
    // The kernel operation is at index 12 in the trace
    FILE* pipe = popen(cmd.c_str(), "r");
    ASSERT_TRUE(pipe != nullptr);
    
    // Read the output
    char buffer[5000];
    std::string output;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
    
    // Check the return code
    int status = pclose(pipe);
    EXPECT_EQ(WEXITSTATUS(status), 0) << "HIPInterceptCompare failed with output:\n" << output;
    
    // Expected checksums from the original execution
    std::map<std::string, int> expected_checksums = {
        {"forceBuffers", 8000},
        {"energyBuffer", 5210},
        {"tiles", 8000},
        {"interactionCount", 8000},
        {"interactingAtoms", 8000}
    };
    
    // Verify the checksums in the output
    bool found_checksums = false;
    std::map<std::string, int> actual_checksums;
    std::istringstream output_stream(output);
    std::string line;
    
    while (std::getline(output_stream, line)) {
        if (line.find("POST-EXECUTION ARGUMENT VALUES:") != std::string::npos) {
            found_checksums = true;
            continue;
        }
        
        if (found_checksums && line.find("checksum:") != std::string::npos) {
            // Extract the argument name and checksum
            size_t arg_pos = line.find("Arg ");
            size_t checksum_pos = line.find("checksum: ");
            if (arg_pos != std::string::npos && checksum_pos != std::string::npos) {
                // Get the argument index
                int arg_idx = std::stoi(line.substr(arg_pos + 4));
                // Get the checksum value
                int checksum = std::stoi(line.substr(checksum_pos + 10));
                
                // Map argument indices to names based on the kernel signature
                std::string arg_name;
                switch (arg_idx) {
                    case 0: arg_name = "forceBuffers"; break;
                    case 1: arg_name = "energyBuffer"; break;
                    case 7: arg_name = "tiles"; break;
                    case 8: arg_name = "interactionCount"; break;
                    case 17: arg_name = "interactingAtoms"; break;
                }
                
                if (!arg_name.empty()) {
                    actual_checksums[arg_name] = checksum;
                }
            }
        }
    }
    
    // Verify all expected checksums are present and match
    for (const auto& [name, expected] : expected_checksums) {
        ASSERT_TRUE(actual_checksums.count(name) > 0) 
            << "Missing checksum for " << name;
        EXPECT_EQ(actual_checksums[name], expected) 
            << "Checksum mismatch for " << name 
            << ". Expected: " << expected 
            << ", Actual: " << actual_checksums[name];
    }
    
    // Clean up temporary files
    std::filesystem::remove_all(temp_dir);
}
