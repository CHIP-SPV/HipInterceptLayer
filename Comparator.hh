#pragma once

#include "Tracer.hh"
#include <iomanip>
#include <chrono>

class Comparator {
public:
    Comparator(const std::string& path1, const std::string& path2) 
        : tracer1(path1), tracer2(path2) {
        tracer1.setSerializeTrace(false);
        tracer2.setSerializeTrace(false);
    }

    void compare(std::ostream& os) const {
        const auto& trace1 = tracer1.trace_;
        const auto& trace2 = tracer2.trace_;

        os << "\nTotal events to compare: " << trace1.operations.size() << "\n";
        os << "Comparing traces...\n";

        auto start = std::chrono::high_resolution_clock::now();

        // Print progress bar
        const int bar_width = 50;
        int progress = 0;
        
        os << "[";
        for (size_t i = 0; i < trace1.operations.size(); i++) {
            int new_progress = static_cast<int>((i + 1) * bar_width / trace1.operations.size());
            while (progress < new_progress) {
                os << "=";
                progress++;
            }
            os.flush();
        }
        os << "] 100% (" << trace1.operations.size() << "/" << trace1.operations.size() << ")\n\n";

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        os << "Total comparison took " << duration.count() << "ms\n";
        
        bool traces_differ = false;
        for (size_t i = 0; i < trace1.operations.size(); i++) {
            const auto& op1 = trace1.operations[i];
            const auto& op2 = trace2.operations[i];

            if (!compareOperations(*op1, *op2)) {
                if (!traces_differ) {
                    os << "Traces differ:\n\n";
                    traces_differ = true;
                }
                printOperationDifference(os, i, *op1, *op2);
            }
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const Comparator& comp) {
        comp.compare(os);
        return os;
    }

private:
    Tracer tracer1;
    Tracer tracer2;

    bool compareOperations(const Operation& op1, const Operation& op2) const {
        // Compare memory states if they exist
        if (op1.pre_state && op2.pre_state) {
            if (op1.pre_state->size != op2.pre_state->size) return false;
            if (memcmp(op1.pre_state->data.get(), op2.pre_state->data.get(), op1.pre_state->size) != 0) return false;
        }
        
        if (op1.post_state && op2.post_state) {
            if (op1.post_state->size != op2.post_state->size) return false;
            if (memcmp(op1.post_state->data.get(), op2.post_state->data.get(), op1.post_state->size) != 0) return false;
        }

        // Compare specific operation types
        if (auto* kernel1 = dynamic_cast<const KernelExecution*>(&op1)) {
            if (auto* kernel2 = dynamic_cast<const KernelExecution*>(&op2)) {
                return compareKernelExecutions(*kernel1, *kernel2);
            }
        }
        else if (auto* mem1 = dynamic_cast<const MemoryOperation*>(&op1)) {
            if (auto* mem2 = dynamic_cast<const MemoryOperation*>(&op2)) {
                return compareMemoryOperations(*mem1, *mem2);
            }
        }
        
        return false;
    }

    bool compareKernelExecutions(const KernelExecution& k1, const KernelExecution& k2) const {
        return k1.kernel_name == k2.kernel_name &&
               k1.grid_dim.x == k2.grid_dim.x &&
               k1.grid_dim.y == k2.grid_dim.y &&
               k1.grid_dim.z == k2.grid_dim.z &&
               k1.block_dim.x == k2.block_dim.x &&
               k1.block_dim.y == k2.block_dim.y &&
               k1.block_dim.z == k2.block_dim.z &&
               k1.shared_mem == k2.shared_mem;
    }

    bool compareMemoryOperations(const MemoryOperation& m1, const MemoryOperation& m2) const {
        return m1.type == m2.type &&
               m1.size == m2.size &&
               m1.kind == m2.kind;
    }

    void printOperationDifference(std::ostream& os, size_t index, const Operation& op1, const Operation& op2) const {
        if (auto* kernel1 = dynamic_cast<const KernelExecution*>(&op1)) {
            if (auto* kernel2 = dynamic_cast<const KernelExecution*>(&op2)) {
                printKernelDifference(os, index, *kernel1, *kernel2);
                return;
            }
        }
        else if (auto* mem1 = dynamic_cast<const MemoryOperation*>(&op1)) {
            if (auto* mem2 = dynamic_cast<const MemoryOperation*>(&op2)) {
                printMemoryDifference(os, index, *mem1, *mem2);
                return;
            }
        }
    }

    void printKernelDifference(std::ostream& os, size_t index, const KernelExecution& k1, const KernelExecution& k2) const {
        os << "Op#" << index << ": Kernel(" << k1.kernel_name << ")\n";
        os << "  Config: gridDim=(" << k1.grid_dim.x << "," << k1.grid_dim.y << "," << k1.grid_dim.z 
           << "), blockDim=(" << k1.block_dim.x << "," << k1.block_dim.y << "," << k1.block_dim.z 
           << "), shared=" << k1.shared_mem << "\n";

        bool states_differ = false;
        if (k1.pre_state && k2.pre_state && 
            memcmp(k1.pre_state->data.get(), k2.pre_state->data.get(), k1.pre_state->size) != 0) {
            states_differ = true;
        }
        if (k1.post_state && k2.post_state && 
            memcmp(k1.post_state->data.get(), k2.post_state->data.get(), k1.post_state->size) != 0) {
            states_differ = true;
        }

        if (states_differ) {
            os << "  Config differences: Pre-execution memory states differ";
            if (k1.post_state && k2.post_state) {
                os << ", Post-execution memory states differ";
            }
            os << "\n";
        }
    }

    void printMemoryDifference(std::ostream& os, size_t index, const MemoryOperation& m1, const MemoryOperation& m2) const {
        os << "Op#" << index << " (";
        switch (m1.type) {
            case MemoryOpType::COPY:
            case MemoryOpType::COPY_ASYNC:
                os << "hipMemcpyAsync call #" << (index + 1) << "): ";
                os << "(size=" << m1.size << ", kind=";
                switch (m1.kind) {
                    case hipMemcpyHostToDevice: os << "hipMemcpyHostToDevice"; break;
                    case hipMemcpyDeviceToHost: os << "hipMemcpyDeviceToHost"; break;
                    case hipMemcpyDeviceToDevice: os << "hipMemcpyDeviceToDevice"; break;
                    default: os << "unknown"; break;
                }
                os << ", stream=" << m1.stream << ")";
                break;
            case MemoryOpType::ALLOC:
                os << "hipMalloc call #" << (index + 1) << "): ";
                os << "(size=" << m1.size << ", stream=" << m1.stream << ")";
                break;
            default:
                os << "unknown memory operation)";
        }
        os << "\n";
    }
};