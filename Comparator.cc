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
                  << "  " << program << " <trace> --gen-repro <op#>   Generate reproducer for operation\n"
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

        // Print scalar values with proper type information
        std::cout << "Scalar Arguments:\n";
        for (size_t i = 0; i < kernel.scalar_values.size(); i++) {
            const auto& value = kernel.scalar_values[i];
            const auto& arg = arguments[i];
            std::cout << "  Arg " << i << " (" << arg.getType() << "): ";
            arg.printValue(std::cout, value.data());
            std::cout << "\n";
        }

    }
}

int main(int argc, char* argv[]) {
    if (argc != 2 && argc != 3 && argc != 4) {
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
    else if (argc == 4 && std::string(argv[2]) == "--gen-repro") {
        try {
            int op_index = std::stoi(argv[3]);
            CodeGen codegen(argv[1]);
            auto output_dir = std::filesystem::current_path().string();
            if (codegen.generateAndCompile(op_index, output_dir)) {
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
        Comparator comparator(argv[1], argv[2]);
        std::cout << comparator;
    }
    
    return 0;
}