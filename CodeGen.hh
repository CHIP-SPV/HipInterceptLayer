#ifndef HIP_INTERCEPT_LAYER_CODE_GEN_HH
#define HIP_INTERCEPT_LAYER_CODE_GEN_HH

#include "KernelManager.hh"
#include "Tracer.hh"
#include <sstream>
#include <string>
#include <unordered_set>
#include <filesystem>
#include <cstdlib>
#include <chrono>
#include <fstream>

namespace hip_intercept {

class CodeGen {
public:
    CodeGen(const Trace& trace, const KernelManager& kernel_manager)
        : trace_(trace), kernel_manager_(kernel_manager) {}

    // Generate complete reproducer code
    std::string generateReproducer() {
        std::stringstream ss;
        
        // Generate includes and main function header
        generateHeader(ss);
        
        // Generate variable declarations
        generateDeclarations(ss);
        
        // Generate initialization code
        generateInitialization(ss);
        
        // Generate kernel launches
        generateKernelLaunches(ss);
        
        // Generate cleanup code
        generateCleanup(ss);
        
        return ss.str();
    }

    // Generate and write the code to a file
    std::string generateFile(const std::string& output_dir = "/tmp") {
        std::filesystem::path dir_path(output_dir);
        std::filesystem::create_directories(dir_path);
        
        // Generate a unique filename using timestamp
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
        
        std::string filename = dir_path / ("kernel_replay_" + std::to_string(timestamp) + ".hip");
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to create output file: " + filename);
        }
        
        file << generateReproducer();
        file.close();
        
        return filename;
    }

    // Compile the generated file using hipcc
    bool compileFile(const std::string& filename, const std::string& output_dir = "/tmp") {
        std::filesystem::path output_path(output_dir);
        std::filesystem::path input_path(filename);
        std::string output_file = (output_path / input_path.stem()).string();

        // Construct the hipcc command
        std::stringstream cmd;
        cmd << "hipcc -w -o " << output_file << " " << filename;
        
        std::cout << "Executing: " << cmd.str() << std::endl;
        
        // Execute the command
        int result = std::system(cmd.str().c_str());
        
        if (result != 0) {
            std::cerr << "Compilation failed with error code: " << result << std::endl;
            return false;
        }
        
        return true;
    }

    // Convenience method to generate and compile in one step
    bool generateAndCompile(const std::string& output_dir = "/tmp") {
        try {
            std::string filename = generateFile(output_dir);
            return compileFile(filename, output_dir);
        } catch (const std::exception& e) {
            std::cerr << "Error during generate and compile: " << e.what() << std::endl;
            return false;
        }
    }

private:
    const Trace& trace_;
    const KernelManager& kernel_manager_;
    std::unordered_set<std::string> declared_vars_;

    void generateHeader(std::stringstream& ss) {
        ss << "#include <hip/hip_runtime.h>\n"
           << "#include <iostream>\n"
           << "#include <cstring>\n\n";
        
        // Add kernel declarations from KernelManager
        for (const auto& exec : trace_.kernel_executions) {
            const Kernel& kernel = kernel_manager_.getKernelByName(exec.kernel_name);
            ss << kernel.getSource() << "\n\n";
        }
        
        // Add trace data declarations
        for (const auto& exec : trace_.kernel_executions) {
            for (size_t i = 0; i < exec.pre_state.size(); i++) {
                ss << "const unsigned char trace_data_" << i << "[] = {";
                const unsigned char* data = reinterpret_cast<const unsigned char*>(exec.pre_state[i].data.get());
                for (size_t j = 0; j < exec.pre_state[i].size; j++) {
                    if (j % 12 == 0) ss << "\n    ";
                    ss << static_cast<unsigned int>(data[j]);
                    if (j < exec.pre_state[i].size - 1) ss << ", ";
                }
                ss << "\n};\n\n";
            }
        }
        
        ss << "int main() {\n"
           << "    hipError_t err;\n\n";
    }

    void generateDeclarations(std::stringstream& ss) {
        for (const auto& exec : trace_.kernel_executions) {
            const Kernel& kernel = kernel_manager_.getKernelByName(exec.kernel_name);
            const auto& args = kernel.getArguments();
            
            for (size_t i = 0; i < args.size(); i++) {
                const auto& arg = args[i];
                std::string var_name = "arg_" + std::to_string(i) + "_" + exec.kernel_name;
                
                if (declared_vars_.find(var_name) != declared_vars_.end()) {
                    continue;
                }
                
                if (arg.isPointer()) {
                    ss << "    " << arg.getBaseType() << "* " << var_name << "_h = nullptr;\n";
                    ss << "    " << arg.getBaseType() << "* " << var_name << "_d = nullptr;\n";
                    declared_vars_.insert(var_name + "_h");
                    declared_vars_.insert(var_name + "_d");
                } else {
                    ss << "    " << arg.getBaseType() << " " << var_name << ";\n";
                    declared_vars_.insert(var_name);
                }
            }
            ss << "\n";
        }
    }

    void generateInitialization(std::stringstream& ss) {
        for (const auto& exec : trace_.kernel_executions) {
            const Kernel& kernel = kernel_manager_.getKernelByName(exec.kernel_name);
            const auto& args = kernel.getArguments();
            
            for (size_t i = 0; i < args.size(); i++) {
                const auto& arg = args[i];
                std::string var_name = "arg_" + std::to_string(i) + "_" + exec.kernel_name;
                
                if (arg.isPointer() && i < exec.pre_state.size()) {
                    size_t size = exec.pre_state[i].size;
                    ss << "    // Allocate and initialize " << var_name << "\n";
                    ss << "    " << var_name << "_h = (" << arg.getBaseType() << "*)malloc(" << size << ");\n";
                    ss << "    err = hipMalloc((void**)&" << var_name << "_d, " << size << ");\n";
                    ss << "    if (err != hipSuccess) { std::cerr << \"Failed to allocate memory\\n\"; return 1; }\n";
                    ss << "    memcpy(" << var_name << "_h, trace_data_" << i 
                       << ", " << size << ");\n";
                    ss << "    err = hipMemcpy(" << var_name << "_d, " << var_name 
                       << "_h, " << size << ", hipMemcpyHostToDevice);\n";
                    ss << "    if (err != hipSuccess) { std::cerr << \"Failed to copy memory\\n\"; return 1; }\n\n";
                } else if (!arg.isPointer() && i < exec.pre_state.size()) {
                    // For scalar arguments, copy from pre_state instead of arg_sizes
                    ss << "    memcpy(&" << var_name << ", " 
                       << "reinterpret_cast<const " << arg.getBaseType() << "*>(trace_data_" << i 
                       << "), sizeof(" << arg.getBaseType() << "));\n";
                }
            }
        }
    }

    void generateKernelLaunches(std::stringstream& ss) {
        for (const auto& exec : trace_.kernel_executions) {
            const Kernel& kernel = kernel_manager_.getKernelByName(exec.kernel_name);
            
            ss << "    // Launch kernel " << kernel.getName() << "\n";
            ss << "    dim3 grid(" << exec.grid_dim.x << ", " 
               << exec.grid_dim.y << ", " << exec.grid_dim.z << ");\n";
            ss << "    dim3 block(" << exec.block_dim.x << ", " 
               << exec.block_dim.y << ", " << exec.block_dim.z << ");\n";
            
            ss << "    " << kernel.getName() << "<<<grid, block, " 
               << exec.shared_mem << ">>>(";
            
            const auto& args = kernel.getArguments();
            for (size_t i = 0; i < args.size(); i++) {
                if (i > 0) ss << ", ";
                std::string var_name = "arg_" + std::to_string(i) + "_" + exec.kernel_name;
                ss << (args[i].isPointer() ? var_name + "_d" : var_name);
            }
            ss << ");\n\n";
        }
    }

    void generateCleanup(std::stringstream& ss) {
        ss << "\n    // Cleanup\n";
        for (const auto& var : declared_vars_) {
            if (var.find("_d") != std::string::npos) {
                ss << "    if (" << var << ") hipFree(" << var << ");\n";
            } else if (var.find("_h") != std::string::npos) {
                ss << "    if (" << var << ") free(" << var << ");\n";
            }
        }
        
        ss << "\n    return 0;\n"
           << "}\n";
    }
};

} // namespace hip_intercept

#endif // HIP_INTERCEPT_LAYER_CODE_GEN_HH
