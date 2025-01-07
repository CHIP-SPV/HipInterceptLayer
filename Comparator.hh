#pragma once

#include "Tracer.hh"
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <cstring>

class Comparator {
    std::string file1;
    std::string file2;
public:
    Comparator(const std::string& f1, const std::string& f2) : file1(f1), file2(f2) {}
    
    friend std::ostream& operator<<(std::ostream& os, const Comparator& comp);

    bool compare(const Operation& op1, const Operation& op2) const {
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
    bool compareArgStates(const std::vector<ArgState>& args1, const std::vector<ArgState>& args2) const {
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

    bool compareKernelExecutions(const KernelExecution& k1, const KernelExecution& k2) const {
        if (k1.kernel_name != k2.kernel_name) {
            std::cerr << "Kernel names do not match: " << k1.kernel_name << " vs " << k2.kernel_name << std::endl;
            return false;
        }

        if (!compareArgStates(k1.pre_args, k2.pre_args)) {
            std::cerr << "Pre-execution arguments do not match for kernel " << k1.kernel_name << std::endl;
            printArgStateDifference(std::cerr, k1.pre_args, k2.pre_args);
            return false;
        }

        if (!compareArgStates(k1.post_args, k2.post_args)) {
            std::cerr << "Post-execution arguments do not match for kernel " << k1.kernel_name << std::endl;
            printArgStateDifference(std::cerr, k1.post_args, k2.post_args);
            return false;
        }

        return true;
    }

    bool compareMemoryOperations(const MemoryOperation& m1, const MemoryOperation& m2) const {
        if (m1.type != m2.type) {
            std::cerr << "Memory operation types do not match" << std::endl;
            return false;
        }

        if (!compareArgStates(m1.pre_args, m2.pre_args)) {
            std::cerr << "Pre-operation arguments do not match" << std::endl;
            printArgStateDifference(std::cerr, m1.pre_args, m2.pre_args);
            return false;
        }

        if (!compareArgStates(m1.post_args, m2.post_args)) {
            std::cerr << "Post-operation arguments do not match" << std::endl;
            printArgStateDifference(std::cerr, m1.post_args, m2.post_args);
            return false;
        }

        return true;
    }

    void printArgStateDifference(std::ostream& os, 
                               const std::vector<ArgState>& args1, 
                               const std::vector<ArgState>& args2, 
                               int num_diffs = 3) const {
        os << "Arguments differ:" << std::endl;
        os << "First set has " << args1.size() << " arguments" << std::endl;
        os << "Second set has " << args2.size() << " arguments" << std::endl;

        int diffs_shown = 0;
        size_t min_size = std::min(args1.size(), args2.size());

        for (size_t i = 0; i < min_size && diffs_shown < num_diffs; ++i) {
            const auto& arg1 = args1[i];
            const auto& arg2 = args2[i];

            if (arg1.data_type_size != arg2.data_type_size || 
                arg1.array_size != arg2.array_size ||
                arg1.data != arg2.data) {
                os << "Difference in argument " << i << ":" << std::endl;
                os << "  First:  type_size=" << arg1.data_type_size 
                   << ", array_size=" << arg1.array_size 
                   << ", total_size=" << arg1.total_size() << std::endl;
                os << "  Second: type_size=" << arg2.data_type_size 
                   << ", array_size=" << arg2.array_size 
                   << ", total_size=" << arg2.total_size() << std::endl;
                
                // Print first few bytes of data if sizes match but content differs
                if (arg1.data.size() == arg2.data.size() && !arg1.data.empty()) {
                    os << "  First few bytes:" << std::endl;
                    os << "    First:  ";
                    printHexBytes(os, arg1.data.data(), std::min(size_t(16), arg1.data.size()));
                    os << std::endl;
                    os << "    Second: ";
                    printHexBytes(os, arg2.data.data(), std::min(size_t(16), arg2.data.size()));
                    os << std::endl;
                }
                
                diffs_shown++;
            }
        }
    }

    void printHexBytes(std::ostream& os, const char* data, size_t size) const {
        os << std::hex << std::setfill('0');
        for (size_t i = 0; i < size; ++i) {
            os << std::setw(2) << (int)(unsigned char)data[i] << " ";
        }
        os << std::dec << std::setfill(' ');
    }
};