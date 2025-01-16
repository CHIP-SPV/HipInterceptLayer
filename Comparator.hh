#pragma once

#include "Tracer.hh"
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <cstring>

class Comparator {
public:
    static constexpr const char* RED = "\033[1;31m";
    static constexpr const char* YELLOW = "\033[1;33m";
    static constexpr const char* CYAN = "\033[1;36m";
    static constexpr const char* RESET = "\033[0m";

    static std::string memoryOpTypeToString(MemoryOpType type) {
        switch (type) {
            case MemoryOpType::ALLOC: return "ALLOC";
            case MemoryOpType::COPY: return "COPY";
            case MemoryOpType::COPY_ASYNC: return "COPY_ASYNC";
            case MemoryOpType::SET: return "SET";
            default: return "UNKNOWN";
        }
    }

    static bool compare(const Operation& op1, const Operation& op2) {
        if (op1.type != op2.type) {
            std::cerr << "Operation types do not match" << std::endl;
            return false;
        }

        if (op1.isKernel()) {
            const auto& k1 = dynamic_cast<const KernelExecution&>(op1);
            const auto& k2 = dynamic_cast<const KernelExecution&>(op2);
            return compareKernelExecutions(k1, k2);
        } else {
            const auto& m1 = dynamic_cast<const MemoryOperation&>(op1);
            const auto& m2 = dynamic_cast<const MemoryOperation&>(op2);
            return compareMemoryOperations(m1, m2);
        }
    }

private:
    static bool compareArgStates(const std::vector<ArgState>& args1, const std::vector<ArgState>& args2) {
        if (args1.size() != args2.size()) {
            std::cerr << "Number of arguments does not match: " 
                      << args1.size() << " vs " << args2.size() << std::endl;
            return false;
        }

        for (size_t i = 0; i < args1.size(); ++i) {
            const auto& arg1 = args1[i];
            const auto& arg2 = args2[i];

            if (arg1.data_type_size != arg2.data_type_size || 
                arg1.array_size != arg2.array_size ||
                arg1.data.size() != arg2.data.size()) {
                std::cerr << "Argument " << i << " sizes do not match" << std::endl;
                return false;
            }

            if (std::memcmp(arg1.data.data(), arg2.data.data(), arg1.data.size()) != 0) {
                std::cerr << "Argument " << i << " data does not match" << std::endl;
                return false;
            }
        }
        return true;
    }

    static bool compareKernelExecutions(const KernelExecution& k1, const KernelExecution& k2) {
        if (k1.kernel_name != k2.kernel_name) {
            std::cerr << "Kernel names do not match: " << k1.kernel_name << " vs " << k2.kernel_name << std::endl;
            return false;
        }

        if (!compareArgStates(k1.pre_args, k2.pre_args)) {
            std::cerr << "Pre-execution arguments do not match for kernel " << k1.kernel_name << std::endl;
            printArgStateDifference(k1.pre_args, k2.pre_args);
            return false;
        }

        if (!compareArgStates(k1.post_args, k2.post_args)) {
            std::cerr << "Post-execution arguments do not match for kernel " << k1.kernel_name << std::endl;
            printArgStateDifference(k1.post_args, k2.post_args);
            return false;
        }

        return true;
    }

    static bool compareMemoryOperations(const MemoryOperation& m1, const MemoryOperation& m2) {
        if (m1.type != m2.type) {
            std::cerr << "Memory operation types do not match" << std::endl;
            return false;
        }

        if (!compareArgStates(m1.pre_args, m2.pre_args)) {
            std::cerr << "Pre-operation arguments do not match" << std::endl;
            printArgStateDifference(m1.pre_args, m2.pre_args);
            return false;
        }

        if (!compareArgStates(m1.post_args, m2.post_args)) {
            std::cerr << "Post-operation arguments do not match" << std::endl;
            printArgStateDifference(m1.post_args, m2.post_args);
            return false;
        }

        return true;
    }

    static void printArgStateDifference(const std::vector<ArgState>& args1, 
                                      const std::vector<ArgState>& args2) {
        std::cerr << RED << "Arguments differ:" << RESET << std::endl;
        std::cerr << "First set has " << args1.size() << " arguments" << std::endl;
        std::cerr << "Second set has " << args2.size() << " arguments" << std::endl;

        // If one set is empty, show what's in the other set
        if (args1.empty() && !args2.empty()) {
            std::cerr << "\nSecond set arguments:" << std::endl;
            for (size_t i = 0; i < args2.size(); ++i) {
                const auto& arg = args2[i];
                std::cerr << "  Arg " << i << ": type_size=" << arg.data_type_size 
                         << ", array_size=" << arg.array_size 
                         << ", total_size=" << arg.total_size() << std::endl;
                if (!arg.data.empty()) {
                    std::cerr << "    First 3 values as float: ";
                    size_t num_floats = std::min(size_t(3), arg.data.size() / sizeof(float));
                    const float* float_data = reinterpret_cast<const float*>(arg.data.data());
                    for (size_t j = 0; j < num_floats; ++j) {
                        std::cerr << float_data[j] << " ";
                    }
                    std::cerr << std::endl;
                }
            }
            return;
        }

        size_t min_size = std::min(args1.size(), args2.size());

        for (size_t i = 0; i < min_size; ++i) {
            const auto& arg1 = args1[i];
            const auto& arg2 = args2[i];

            if (arg1.data_type_size != arg2.data_type_size || 
                arg1.array_size != arg2.array_size ||
                arg1.data != arg2.data) {
                std::cerr << YELLOW << "\nDifference in argument " << i << ":" << RESET << std::endl;
                std::cerr << "  First:  type_size=" << arg1.data_type_size 
                   << ", array_size=" << arg1.array_size 
                   << ", total_size=" << arg1.total_size() << std::endl;
                std::cerr << "  Second: type_size=" << arg2.data_type_size 
                   << ", array_size=" << arg2.array_size 
                   << ", total_size=" << arg2.total_size() << std::endl;
                
                // Print first 3 values as floats if data exists
                if (!arg1.data.empty() && !arg2.data.empty()) {
                    std::cerr << "  First 3 values:" << std::endl;
                    size_t num_floats1 = std::min(size_t(3), arg1.data.size() / sizeof(float));
                    size_t num_floats2 = std::min(size_t(3), arg2.data.size() / sizeof(float));
                    
                    std::cerr << "    First:  ";
                    const float* float_data1 = reinterpret_cast<const float*>(arg1.data.data());
                    for (size_t j = 0; j < num_floats1; ++j) {
                        std::cerr << float_data1[j] << " ";
                    }
                    std::cerr << std::endl;
                    
                    std::cerr << "    Second: ";
                    const float* float_data2 = reinterpret_cast<const float*>(arg2.data.data());
                    for (size_t j = 0; j < num_floats2; ++j) {
                        std::cerr << float_data2[j] << " ";
                    }
                    std::cerr << std::endl;
                }
            }
        }

        // Show if there are extra arguments in either set
        if (args1.size() > min_size) {
            std::cerr << "\nExtra arguments in first set:" << std::endl;
            for (size_t i = min_size; i < args1.size(); ++i) {
                const auto& arg = args1[i];
                std::cerr << "  Arg " << i << ": type_size=" << arg.data_type_size 
                         << ", array_size=" << arg.array_size << std::endl;
                if (!arg.data.empty()) {
                    std::cerr << "    First 3 values as float: ";
                    size_t num_floats = std::min(size_t(3), arg.data.size() / sizeof(float));
                    const float* float_data = reinterpret_cast<const float*>(arg.data.data());
                    for (size_t j = 0; j < num_floats; ++j) {
                        std::cerr << float_data[j] << " ";
                    }
                    std::cerr << std::endl;
                }
            }
        }
        if (args2.size() > min_size) {
            std::cerr << "\nExtra arguments in second set:" << std::endl;
            for (size_t i = min_size; i < args2.size(); ++i) {
                const auto& arg = args2[i];
                std::cerr << "  Arg " << i << ": type_size=" << arg.data_type_size 
                         << ", array_size=" << arg.array_size << std::endl;
                if (!arg.data.empty()) {
                    std::cerr << "    First 3 values as float: ";
                    size_t num_floats = std::min(size_t(3), arg.data.size() / sizeof(float));
                    const float* float_data = reinterpret_cast<const float*>(arg.data.data());
                    for (size_t j = 0; j < num_floats; ++j) {
                        std::cerr << float_data[j] << " ";
                    }
                    std::cerr << std::endl;
                }
            }
        }
    }

    static void printHexBytes(const char* data, size_t size) {
        std::cerr << std::hex << std::setfill('0');
        for (size_t i = 0; i < size; ++i) {
            std::cerr << std::setw(2) << (int)(unsigned char)data[i] << " ";
        }
        std::cerr << std::dec << std::setfill(' ');
    }
};