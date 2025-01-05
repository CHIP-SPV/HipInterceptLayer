#pragma once

#include "Tracer.hh"
#include <iomanip>
#include <chrono>
#include <vector>
#include <cstring>

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

        // // Print progress bar
        // const int bar_width = 50;
        // int progress = 0;
        
        // os << "[";
        // for (size_t i = 0; i < trace1.operations.size(); i++) {
        //     int new_progress = static_cast<int>((i + 1) * bar_width / trace1.operations.size());
        //     while (progress < new_progress) {
        //         os << "=";
        //         progress++;
        //     }
        //     os.flush();
        // }
        // os << "] 100% (" << trace1.operations.size() << "/" << trace1.operations.size() << ")\n\n";

        // auto end = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // os << "Total comparison took " << duration.count() << "ms\n";
        
        bool traces_differ = false;
        size_t num_operations = std::min(trace1.operations.size(), trace2.operations.size());
        for (size_t i = 0; i < num_operations; i++) {
            auto op1 = tracer1.getOperation(i);
            auto op2 = tracer2.getOperation(i);

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

    Tracer tracer1;
    Tracer tracer2; 
private:
    bool compareMemoryStates(const std::shared_ptr<MemoryState>& state1, 
                               const std::shared_ptr<MemoryState>& state2) const {
        if (!state1 || !state2) return state1 == state2;
        if (state1->total_size != state2->total_size) return false;

        // Create contiguous buffers for comparison
        std::vector<char> buffer1(state1->total_size);
        std::vector<char> buffer2(state2->total_size);

        // Copy chunks into contiguous buffers
        size_t offset1 = 0;
        for (const auto& chunk : state1->chunks) {
            std::memcpy(buffer1.data() + offset1, chunk.data.get(), chunk.size);
            offset1 += chunk.size;
        }

        size_t offset2 = 0;
        for (const auto& chunk : state2->chunks) {
            std::memcpy(buffer2.data() + offset2, chunk.data.get(), chunk.size);
            offset2 += chunk.size;
        }

        // Compare the contiguous buffers
        return std::memcmp(buffer1.data(), buffer2.data(), state1->total_size) == 0;
    }

    bool compareOperations(const Operation& op1, const Operation& op2) const {
        if (op1.type != op2.type) return false;

        // Compare pre-states
        if (op1.pre_state && op2.pre_state) {
            if (!compareMemoryStates(op1.pre_state, op2.pre_state)) return false;
        } else if (op1.pre_state || op2.pre_state) {
            return false;
        }

        // Compare post-states
        if (op1.post_state && op2.post_state) {
            if (!compareMemoryStates(op1.post_state, op2.post_state)) return false;
        } else if (op1.post_state || op2.post_state) {
            return false;
        }

        if (op1.type == OperationType::KERNEL && op2.type == OperationType::KERNEL) {
            const KernelExecution* k1 = dynamic_cast<const KernelExecution*>(&op1);
            const KernelExecution* k2 = dynamic_cast<const KernelExecution*>(&op2);
            return compareKernelExecutions(*k1, *k2);
        } else if (op1.type == OperationType::MEMORY && op2.type == OperationType::MEMORY) {
            const MemoryOperation* m1 = dynamic_cast<const MemoryOperation*>(&op1);
            const MemoryOperation* m2 = dynamic_cast<const MemoryOperation*>(&op2);
            return compareMemoryOperations(*m1, *m2);
        } else {
            std::cout << "Comparing operations of different types" << std::endl;
            return false;
        }

        return true;
    }

    bool compareKernelExecutions(const KernelExecution& k1, const KernelExecution& k2) const {
        std::cout << "Comparing kernels: " << k1.kernel_name << " and " << k2.kernel_name << std::endl;
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
        if (k1.pre_state && k2.pre_state) {
            if (!compareMemoryStates(k1.pre_state, k2.pre_state)) {
                states_differ = true;
            }
        }
        if (k1.post_state && k2.post_state) {
            if (!compareMemoryStates(k1.post_state, k2.post_state)) {
                states_differ = true;
            }
        }

        if (states_differ) {
            os << "  Memory differences: ";
            if (k1.pre_state && k2.pre_state && !compareMemoryStates(k1.pre_state, k2.pre_state)) {
                os << "Pre-execution memory states differ";
            }
            if (k1.post_state && k2.post_state && !compareMemoryStates(k1.post_state, k2.post_state)) {
                if (k1.pre_state && k2.pre_state) os << ", ";
                os << "Post-execution memory states differ";
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