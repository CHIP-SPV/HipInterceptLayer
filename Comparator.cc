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
                  << "  " << program << " <trace> --gen-repro <op#>   Generate reproducer for operation\n";
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
    else if (argc == 4 && std::string(argv[2]) == "--gen-repro") {
        try {
            int op_index = std::stoi(argv[3]);
            CodeGen codegen(argv[1]);
            // output to current directory
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