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

    void printKernelValues(const KernelExecution& kernel) {
        std::cout << "\nKernel: " << kernel.kernel_name << "\n";
        std::cout << "Grid: (" << kernel.grid_dim.x << "," << kernel.grid_dim.y << "," << kernel.grid_dim.z << ")\n";
        std::cout << "Block: (" << kernel.block_dim.x << "," << kernel.block_dim.y << "," << kernel.block_dim.z << ")\n";
        std::cout << "Shared Memory: " << kernel.shared_mem << " bytes\n";

        // Print scalar values
        std::cout << "Scalar Arguments:\n";
        for (size_t i = 0; i < kernel.scalar_values.size(); i++) {
            const auto& value = kernel.scalar_values[i];
            std::cout << "  Arg " << i << " (size=" << value.size() << "): ";
            
            // Try to interpret common types
            if (value.size() == sizeof(int)) {
                std::cout << *reinterpret_cast<const int*>(value.data()) << " (int)";
            } else if (value.size() == sizeof(float)) {
                std::cout << *reinterpret_cast<const float*>(value.data()) << " (float)";
            } else if (value.size() == sizeof(double)) {
                std::cout << *reinterpret_cast<const double*>(value.data()) << " (double)";
            } else if (value.size() == sizeof(float4)) {
                const float4* vec = reinterpret_cast<const float4*>(value.data());
                std::cout << "(" << vec->x << "," << vec->y << "," << vec->z << "," << vec->w << ") (float4)";
            } else if (value.size() == sizeof(int2)) {
                const int2* vec = reinterpret_cast<const int2*>(value.data());
                std::cout << "(" << vec->x << "," << vec->y << ") (int2)";
            } else if (value.size() == sizeof(float2)) {
                const float2* vec = reinterpret_cast<const float2*>(value.data());
                std::cout << "(" << vec->x << "," << vec->y << ") (float2)";
            } else {
                // Print raw bytes for unknown types
                std::cout << std::hex << std::setw(2) << std::setfill('0');
                for (size_t j = 0; j < value.size(); j++) {
                    std::cout << static_cast<int>(value[j]);
                }
                std::cout << std::dec;
            }
            std::cout << "\n";
        }

        // Print memory state checksums
        std::cout << "Memory State Checksums:\n";
        if (!kernel.pre_state.chunks.empty()) {
            float pre_checksum = calculateChecksum(kernel.pre_state.chunks[0].data.get(), kernel.pre_state.chunks[0].size);
            for (size_t i = 1; i < kernel.pre_state.chunks.size(); i++) {
                pre_checksum += calculateChecksum(kernel.pre_state.chunks[i].data.get(), kernel.pre_state.chunks[i].size);
            }
            std::cout << "  Pre-state: " << std::hex << std::setprecision(8) << pre_checksum << std::dec << "\n";
        }
        
        if (!kernel.post_state.chunks.empty()) {
            float post_checksum = calculateChecksum(kernel.post_state.chunks[0].data.get(), kernel.post_state.chunks[0].size);
            for (size_t i = 1; i < kernel.post_state.chunks.size(); i++) {
                post_checksum += calculateChecksum(kernel.post_state.chunks[i].data.get(), kernel.post_state.chunks[i].size);
            }
            std::cout << "  Post-state: " << std::hex << std::setprecision(8) << post_checksum << std::dec << "\n";
        }
        std::cout << "\n";
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
                    printKernelValues(*kernel);
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