#include "Comparator.hh"
#include "CodeGen.hh"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <algorithm>
#include <string>

namespace {
    const char* RED = "\033[1;31m";
    const char* YELLOW = "\033[1;33m";
    const char* CYAN = "\033[1;36m";
    const char* RESET = "\033[0m";

    void printUsage(const char* program) {
        std::cerr << "Usage:\n"
                  << "  " << program << " <trace1> [trace2]     Compare two traces\n"
                  << "  " << program << " <trace> --gen-repro <op#> [--debug]   Generate reproducer for operation\n"
                  << "  " << program << " <trace> --print-vals   Print checksums and scalar values\n";
    }

    void printKernelValues(const KernelExecution& kernel, const Tracer& tracer) {
        std::cout << "\nKernel: " << kernel.kernel_name << "\n";
        std::cout << "Grid: (" << kernel.grid_dim.x << "," << kernel.grid_dim.y << "," << kernel.grid_dim.z << ")\n";
        std::cout << "Block: (" << kernel.block_dim.x << "," << kernel.block_dim.y << "," << kernel.block_dim.z << ")\n";
        std::cout << "Shared Memory: " << kernel.shared_mem << " bytes\n";

        // Get kernel info from the tracer's KernelManager
        const auto& manager = tracer.getKernelManager();
        Kernel k = manager.getKernelByName(kernel.kernel_name);
        auto arguments = k.getArguments();

        std::cout << "PRE-EXECUTION ARGUMENT VALUES:\n";
        for (size_t i = 0; i < kernel.pre_args.size(); i++) {
            const auto& value = kernel.pre_args[i];
            const auto& arg = arguments[i];
            std::cout << "  Arg " << i << " (" << arg.getType() << "): size=" << value.total_size() << " ";
            if (arg.isPointer()) {
                // For array arguments, print the checksum
                float checksum = calculateChecksum(value.data.data(), value.total_size());
                std::cout << "checksum=" << std::hex << std::setprecision(8) << checksum << std::dec;
            } else {
                // For scalar arguments, print the value
                arg.printValue(std::cout, value.data.data());
            }
            std::cout << "\n";
        }

        std::cout << "POST-EXECUTION ARGUMENT VALUES:\n";
        for (size_t i = 0; i < kernel.post_args.size(); i++) {
            const auto& value = kernel.post_args[i];
            const auto& arg = arguments[i];
            std::cout << "  Arg " << i << " (" << arg.getType() << "): size=" << value.total_size() << " ";
            if (arg.isPointer()) {
                // For array arguments, print the checksum
                float checksum = calculateChecksum(value.data.data(), value.total_size());
                std::cout << "checksum=" << std::hex << std::setprecision(8) << checksum << std::dec;
            } else {
                // For scalar arguments, print the value
                arg.printValue(std::cout, value.data.data());
            }
            std::cout << "\n";
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2 && argc != 3 && argc != 4 && argc != 5) {
        printUsage(argv[0]);
        return 1;
    }

    if (argc == 2) {
        Tracer tracer1(argv[1]);
        tracer1.setSerializeTrace(false);
        std::cout << tracer1;
    } 
    else if (argc == 3 && std::string(argv[2]) == "--print-vals") {
        Tracer tracer(argv[1]);
        tracer.setSerializeTrace(false);
        
        // Print values for each kernel execution
        for (size_t i = 0; i < tracer.trace_.operations.size(); i++) {
            const auto& op = tracer.trace_.operations[i];
            if (op->type == OperationType::KERNEL) {
                const auto* kernel = dynamic_cast<const KernelExecution*>(op.get());
                if (kernel) {
                    std::cout << "\n=== Operation " << i << " ===\n";
                    printKernelValues(*kernel, tracer);
                }
            }
        }
    }
    else if (argc >= 4 && std::string(argv[2]) == "--gen-repro") {
        try {
            int op_index = std::stoi(argv[3]);
            CodeGen codegen(argv[1]);
            auto output_dir = std::filesystem::current_path().string();
            bool debug_mode = (argc == 5 && std::string(argv[4]) == "--debug");
            if (codegen.generateAndCompile(op_index, output_dir, debug_mode)) {
                std::cout << "Successfully generated and compiled reproducer for operation " << op_index << "\n";
                return 0;
            }
            return 1;
        } catch (const std::exception& e) {
            std::cerr << "Error generating reproducer: " << e.what() << "\n";
            return 1;
        }
    }
    else if (argc == 3) {
        Tracer tracer1(argv[1]);
        Tracer tracer2(argv[2]);
        tracer1.setSerializeTrace(false);
        tracer2.setSerializeTrace(false);
        
        if (tracer1.trace_.operations.size() != tracer2.trace_.operations.size()) {
            std::cerr << "Number of operations does not match: " 
                      << tracer1.trace_.operations.size() << " vs " 
                      << tracer2.trace_.operations.size() << std::endl;
            return 1;
        }

        bool all_match = true;
        for (size_t i = 0; i < tracer1.trace_.operations.size(); i++) {
            const auto& op1 = tracer1.trace_.operations[i];
            const auto& op2 = tracer2.trace_.operations[i];
            
            if (!Comparator::compare(*op1, *op2)) {
                std::cerr << "Operation " << i << " does not match" << std::endl;
                all_match = false;
            }
        }
        
        return all_match ? 0 : 1;
    }
    
    return 0;
}