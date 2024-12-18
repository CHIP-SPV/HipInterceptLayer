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
#include <iomanip>

class CodeGen {
public:
    CodeGen(const std::string& trace_file_path, const KernelManager& kernel_manager)
        : trace_file_path_(trace_file_path), kernel_manager_(kernel_manager), operation_index_(-1) {
        // Load trace from file
        Tracer tracer(trace_file_path);
        trace_ = tracer.instance().trace_;
    }

    std::string generateReproducer(std::string kernel_name, int instance_index) {
        // Find all operations for the given kernel name and instance index
        std::vector<int> operation_indices;
        for (size_t i = 0; i < trace_.kernel_executions.size(); i++) {
            if (trace_.kernel_executions[i].kernel_name == kernel_name) {
                operation_indices.push_back(i);
            }
        }
        if (operation_indices.empty()) {
            throw std::runtime_error("Kernel not found in trace");
        }
        return generateReproducer(operation_indices[instance_index]);
    }

    // Generate complete reproducer code
    std::string generateReproducer(int operation_index) {
        if (operation_index < 0 || operation_index >= trace_.kernel_executions.size()) {
            throw std::runtime_error("Operation index out of range: " + 
                std::to_string(operation_index));
        }

        operation_index_ = operation_index;
        std::stringstream ss;
        
        // Generate includes and main function header
        generateHeader(ss);
        
        // Generate variable declarations for specific operation
        generateDeclarations(ss, operation_index);
        
        // Generate initialization code for specific operation
        generateInitialization(ss, operation_index);
        
        // Generate single kernel launch
        generateKernelLaunches(ss, operation_index);
        
        // Generate cleanup code
        generateCleanup(ss);
        
        return ss.str();
    }

    // Generate and write the code to a file
    std::string generateFile(int operation_index, const std::string& output_dir = "/tmp") {
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
        
        file << generateReproducer(operation_index);
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
    bool generateAndCompile(int operation_index, const std::string& output_dir = "/tmp") {
        try {
            std::string filename = generateFile(operation_index, output_dir);
            return compileFile(filename, output_dir);
        } catch (const std::exception& e) {
            std::cerr << "Error during generate and compile: " << e.what() << std::endl;
            return false;
        }
    }

private:
    std::string trace_file_path_;
    const KernelManager& kernel_manager_;
    Trace trace_;  // Now owned by CodeGen
    std::unordered_set<std::string> declared_vars_;
    int operation_index_;

    void generateHeader(std::stringstream& ss) {
        ss << "#include <hip/hip_runtime.h>\n"
           << "#include <iostream>\n"
           << "#include <cstring>\n"
           << "#include <fstream>\n\n";
        
        // Add kernel declaration only for the current operation
        const KernelExecution& exec = trace_.kernel_executions[operation_index_];
        const Kernel& kernel = kernel_manager_.getKernelByName(exec.kernel_name);
        std::string source = kernel.getSource();
        
        if (!source.empty()) {
            ss << source << "\n\n";
        } else {
            // Generate kernel declaration with empty body
            ss << "__global__ void " << kernel.getName() << "(";
            
            const auto& args = kernel.getArguments();
            for (size_t i = 0; i < args.size(); i++) {
                if (i > 0) ss << ", ";
                ss << args[i].getType();
                if (!args[i].getName().empty()) {
                    ss << " " << args[i].getName();
                } else {
                    ss << " arg" << i;
                }
            }
            
            ss << ") {\n"
               << "    // TODO: Original kernel source not available\n"
               << "    // This is a placeholder implementation\n"
               << "}\n\n";
        }
        
        // Add helper function to load trace data
        ss << "bool loadTraceData(const char* filename, size_t offset, size_t size, void* dest) {\n"
           << "    std::ifstream file(filename, std::ios::binary);\n"
           << "    if (!file.is_open()) {\n"
           << "        std::cerr << \"Failed to open trace file: \" << filename << std::endl;\n"
           << "        return false;\n"
           << "    }\n"
           << "    file.seekg(offset);\n"
           << "    file.read(static_cast<char*>(dest), size);\n"
           << "    return file.good();\n"
           << "}\n\n";
        
        ss << "int main() {\n"
           << "    hipError_t err;\n"
           << "    const char* trace_file = \"" << trace_file_path_ << "\";\n\n";
    }

    void generateDeclarations(std::stringstream& ss, int operation_index) {
        auto process_execution = [&](const KernelExecution& exec) {
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
        };

        process_execution(trace_.kernel_executions[operation_index]);
    }

    void generateInitialization(std::stringstream& ss, int operation_index) {
        size_t current_offset = 0;

        auto process_execution = [&](const KernelExecution& exec) {
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
                    
                    // Load data from trace file
                    ss << "    if (!loadTraceData(trace_file, " << current_offset << ", " << size 
                       << ", " << var_name << "_h)) { return 1; }\n";
                    
                    ss << "    err = hipMemcpy(" << var_name << "_d, " << var_name 
                       << "_h, " << size << ", hipMemcpyHostToDevice);\n";
                    ss << "    if (err != hipSuccess) { std::cerr << \"Failed to copy memory\\n\"; return 1; }\n\n";
                    
                    current_offset += size;
                } else if (!arg.isPointer() && i < exec.pre_state.size()) {
                    size_t size = sizeof(arg.getBaseType());
                    ss << "    if (!loadTraceData(trace_file, " << current_offset << ", " 
                       << "sizeof(" << arg.getBaseType() << "), &" << var_name << ")) { return 1; }\n";
                    
                    current_offset += size;
                }
            }
        };

        process_execution(trace_.kernel_executions[operation_index]);
    }

    void generateKernelLaunches(std::stringstream& ss, int operation_index) {
        auto process_execution = [&](const KernelExecution& exec) {
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
        };

        process_execution(trace_.kernel_executions[operation_index]);
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

#endif // HIP_INTERCEPT_LAYER_CODE_GEN_HH
