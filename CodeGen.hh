#ifndef HIP_INTERCEPT_LAYER_CODE_GEN_HH
#define HIP_INTERCEPT_LAYER_CODE_GEN_HH

#include "KernelManager.hh"
#include "Tracer.hh"
#include <sstream>
#include <string>
#include <unordered_set>

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

private:
    const Trace& trace_;
    const KernelManager& kernel_manager_;
    std::unordered_set<std::string> declared_vars_;

    void generateHeader(std::stringstream& ss) {
        ss << "#include <hip/hip_runtime.h>\n"
           << "#include <iostream>\n"
           << "#include <cstring>\n\n"
           << "int main() {\n"
           << "    hipError_t err;\n\n";
    }

    void generateDeclarations(std::stringstream& ss) {
        for (const auto& exec : trace_.kernel_executions) {
            const Kernel& kernel = kernel_manager_.getKernelByName(exec.kernel_name);
            const auto& args = kernel.getArguments();
            
            for (size_t i = 0; i < args.size(); i++) {
                const auto& arg = args[i];
                std::string base_name = "arg_" + std::to_string(i) + "_" + exec.kernel_name;
                
                if (declared_vars_.find(base_name) != declared_vars_.end()) {
                    continue;
                }
                
                if (arg.isPointer()) {
                    std::string host_var = base_name + "_h";
                    std::string dev_var = base_name + "_d";
                    ss << "    " << arg.getType() << " " << host_var << " = nullptr;\n";
                    ss << "    " << arg.getType() << " " << dev_var << " = nullptr;\n";
                    declared_vars_.insert(host_var);
                    declared_vars_.insert(dev_var);
                } else {
                    ss << "    " << arg.getType() << " " << base_name << ";\n";
                    declared_vars_.insert(base_name);
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
                    ss << "    " << var_name << "_h = (" << arg.getType() << ")malloc(" << size << ");\n";
                    ss << "    err = hipMalloc(&" << var_name << "_d, " << size << ");\n";
                    ss << "    if (err != hipSuccess) { std::cerr << \"Failed to allocate memory\\n\"; return 1; }\n";
                    ss << "    memcpy(" << var_name << "_h, trace_data_" << i << ", " << size << ");\n";
                    ss << "    err = hipMemcpy(" << var_name << "_d, " << var_name << "_h, " << size 
                       << ", hipMemcpyHostToDevice);\n";
                    ss << "    if (err != hipSuccess) { std::cerr << \"Failed to copy memory\\n\"; return 1; }\n\n";
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
