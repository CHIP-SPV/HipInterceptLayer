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

    bool compare(std::ostream& os) {
        const auto& trace1 = tracer1.trace_;
        const auto& trace2 = tracer2.trace_;

        os << "\nTotal events to compare: " << trace1.operations.size() << "\n";
        os << "Comparing traces...\n";

        auto start = std::chrono::high_resolution_clock::now();      
        size_t num_operations = std::min(trace1.operations.size(), trace2.operations.size());
        for (size_t i = 0; i < num_operations; i++) {
            auto op1 = tracer1.getOperation(i);
            auto op2 = tracer2.getOperation(i);

            if (!compareOperations(*op1, *op2)) {
                printOperationDifference(os, i, *op1, *op2);
                return false;
            }
        }
        return true;
    }

    friend std::ostream& operator<<(std::ostream& os, Comparator& comp) {
        comp.compare(os);
        return os;
    }

    Tracer tracer1;
    Tracer tracer2; 
private:
    bool pre_state_differ = false;
    bool post_state_differ = false;

    bool compareMemoryStates(const MemoryState& state1, const MemoryState& state2) const {
        if (state1.total_size != state2.total_size) return false;

        // Create contiguous buffers for comparison
        std::vector<char> buffer1(state1.total_size);
        std::vector<char> buffer2(state2.total_size);

        // Copy chunks into contiguous buffers
        size_t offset1 = 0;
        for (const auto& chunk : state1.chunks) {
            std::memcpy(buffer1.data() + offset1, chunk.data.get(), chunk.size);
            offset1 += chunk.size;
        }

        size_t offset2 = 0;
        for (const auto& chunk : state2.chunks) {
            std::memcpy(buffer2.data() + offset2, chunk.data.get(), chunk.size);
            offset2 += chunk.size;
        }

        // Compare the contiguous buffers
        return std::memcmp(buffer1.data(), buffer2.data(), state1.total_size) == 0;
    }

    bool compareOperations(const Operation& op1, const Operation& op2) {
        if (op1.type != op2.type) return false;

        // Compare pre-states
        if (!compareMemoryStates(op1.pre_state, op2.pre_state)) {
            pre_state_differ = true;
            return false;
        }

        // Compare post-states
        if (!compareMemoryStates(op1.post_state, op2.post_state)) {
            post_state_differ = true;
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
        
        // Compare basic properties
        if (k1.kernel_name != k2.kernel_name ||
            k1.grid_dim.x != k2.grid_dim.x ||
            k1.grid_dim.y != k2.grid_dim.y ||
            k1.grid_dim.z != k2.grid_dim.z ||
            k1.block_dim.x != k2.block_dim.x ||
            k1.block_dim.y != k2.block_dim.y ||
            k1.block_dim.z != k2.block_dim.z ||
            k1.shared_mem != k2.shared_mem) {
            return false;
        }

        // Compare scalar values
        if (k1.scalar_values.size() != k2.scalar_values.size()) {
            std::cout << "Scalar value count mismatch: " << k1.scalar_values.size() 
                     << " vs " << k2.scalar_values.size() << std::endl;
            return false;
        }

        for (size_t i = 0; i < k1.scalar_values.size(); i++) {
            const auto& v1 = k1.scalar_values[i];
            const auto& v2 = k2.scalar_values[i];
            if (v1.size() != v2.size()) {
                std::cout << "Scalar value size mismatch at index " << i << ": " 
                         << v1.size() << " vs " << v2.size() << std::endl;
                return false;
            }
            if (std::memcmp(v1.data(), v2.data(), v1.size()) != 0) {
                std::cout << "Scalar value mismatch at index " << i << std::endl;
                return false;
            }
        }

        return true;
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
 
    void printMemoryStateDifference(std::ostream& os, const MemoryState& state1, const MemoryState& state2, int num_diffs = 3) const {
        // First check if there are any chunks to compare
        if (state1.chunks.empty() || state2.chunks.empty()) {
            os << "One or both states have no data chunks\n";
            return;
        }

        // Get the size of the first chunk from each state
        size_t size1 = state1.chunks[0].size;
        size_t size2 = state2.chunks[0].size;
        
        // Calculate how many elements we can safely compare
        size_t max_elements = std::min(size1, size2) / sizeof(float);
        int max_diffs = std::min(num_diffs, static_cast<int>(max_elements));

        if (pre_state_differ) {
            os << "Pre-state difference: \n";
            for (int i = 0; i < max_diffs; i++) {
                const float* data1 = reinterpret_cast<const float*>(state1.chunks[0].data.get());
                const float* data2 = reinterpret_cast<const float*>(state2.chunks[0].data.get());
                os << "  [" << i << "]: " << data1[i] << " vs " << data2[i] << "\n";
            }
        }
        if (post_state_differ) {
            os << "Post-state difference: \n";
            for (int i = 0; i < max_diffs; i++) {
                const float* data1 = reinterpret_cast<const float*>(state1.chunks[0].data.get());
                const float* data2 = reinterpret_cast<const float*>(state2.chunks[0].data.get());
                os << "  [" << i << "]: " << data1[i] << " vs " << data2[i] << "\n";
            }
        }
    }

    void printKernelDifference(std::ostream& os, size_t index, const KernelExecution& k1, const KernelExecution& k2) const {
        os << "Op#" << index << ": Kernel(" << k1.kernel_name << ")\n";
        os << "  Config: gridDim=(" << k1.grid_dim.x << "," << k1.grid_dim.y << "," << k1.grid_dim.z 
           << "), blockDim=(" << k1.block_dim.x << "," << k1.block_dim.y << "," << k1.block_dim.z 
           << "), shared=" << k1.shared_mem << "\n";

        // Print scalar value differences
        for (size_t i = 0; i < std::max(k1.scalar_values.size(), k2.scalar_values.size()); i++) {
            if (i >= k1.scalar_values.size()) {
                os << "  Scalar arg " << i << ": missing in first execution\n";
                continue;
            }
            if (i >= k2.scalar_values.size()) {
                os << "  Scalar arg " << i << ": missing in second execution\n";
                continue;
            }

            const auto& v1 = k1.scalar_values[i];
            const auto& v2 = k2.scalar_values[i];
            if (v1.size() != v2.size()) {
                os << "  Scalar arg " << i << ": size mismatch (" << v1.size() << " vs " << v2.size() << ")\n";
                continue;
            }

            if (std::memcmp(v1.data(), v2.data(), v1.size()) != 0) {
                // Try to interpret and print the values based on common sizes
                if (v1.size() == sizeof(int)) {
                    os << "  Scalar arg " << i << " (int): " 
                       << *reinterpret_cast<const int*>(v1.data()) << " vs "
                       << *reinterpret_cast<const int*>(v2.data()) << "\n";
                } else if (v1.size() == sizeof(float)) {
                    os << "  Scalar arg " << i << " (float): " 
                       << *reinterpret_cast<const float*>(v1.data()) << " vs "
                       << *reinterpret_cast<const float*>(v2.data()) << "\n";
                } else if (v1.size() == sizeof(double)) {
                    os << "  Scalar arg " << i << " (double): " 
                       << *reinterpret_cast<const double*>(v1.data()) << " vs "
                       << *reinterpret_cast<const double*>(v2.data()) << "\n";
                } else {
                    os << "  Scalar arg " << i << ": binary difference (size=" << v1.size() << ")\n";
                }
            }
        }

        printMemoryStateDifference(os, k1.pre_state, k2.pre_state);
        printMemoryStateDifference(os, k1.post_state, k2.post_state);
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