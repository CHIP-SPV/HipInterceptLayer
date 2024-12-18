#pragma once

#include "Types.hh"
#include "Util.hh"
#include "KernelManager.hh"

#ifndef HIP_INTERCEPT_LAYER_TRACER_HH
#define HIP_INTERCEPT_LAYER_TRACER_HH

    #include "Util.hh"

#include <string>
#include <fstream>
#include <memory>
#include <vector>
#include <map>
#include <sstream>
#include <unordered_map>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <unistd.h>
#include <linux/limits.h>
#include <sys/stat.h>
#include <dlfcn.h>
#include <link.h>
#include <algorithm>
#include <regex>
#include <fstream>

namespace hip_intercept {

// Memory state tracking
struct MemoryState {
    std::unique_ptr<char[]> data;
    size_t size;
    
    explicit MemoryState(size_t s);
    MemoryState(const char* src, size_t s);
    MemoryState(); // Default constructor
    
    // Add copy constructor
    MemoryState(const MemoryState& other) : size(other.size) {
        if (other.data) {
            data = std::make_unique<char[]>(size);
            std::memcpy(data.get(), other.data.get(), size);
        }
    }
    
    // Add copy assignment operator
    MemoryState& operator=(const MemoryState& other) {
        if (this != &other) {
            size = other.size;
            if (other.data) {
                data = std::make_unique<char[]>(size);
                std::memcpy(data.get(), other.data.get(), size);
            } else {
                data.reset();
            }
        }
        return *this;
    }
    
    // Add move constructor
    MemoryState(MemoryState&& other) noexcept 
        : data(std::move(other.data)), size(other.size) {
        other.size = 0;
    }
    
    // Add move assignment operator
    MemoryState& operator=(MemoryState&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);
            size = other.size;
            other.size = 0;
        }
        return *this;
    }
};

// Kernel execution record
struct KernelExecution {
    void* function_address;
    std::string kernel_name;
    dim3 grid_dim;
    dim3 block_dim;
    size_t shared_mem;
    hipStream_t stream;
    uint64_t execution_order;
    
    std::vector<MemoryState> pre_state;
    std::vector<MemoryState> post_state;
    std::vector<std::pair<void*, size_t>> changes;
    std::vector<void*> arg_ptrs;
    std::vector<size_t> arg_sizes;
    std::map<int, std::vector<std::pair<size_t, std::pair<float, float>>>> changes_by_arg;
};

struct MemoryOperation {
    MemoryOpType type;
    void* dst;
    const void* src;
    size_t size;
    int value;
    hipMemcpyKind kind;
    uint64_t execution_order;
    hipStream_t stream;
    
    std::shared_ptr<MemoryState> pre_state;
    std::shared_ptr<MemoryState> post_state;
};

struct Trace {
    std::vector<KernelExecution> kernel_executions;
    std::vector<MemoryOperation> memory_operations;
    uint64_t timestamp;
};


inline std::ostream& operator<<(std::ostream& os, const KernelExecution& exec) {
    os << "KernelExecution: " << exec.kernel_name 
       << " (grid: " << exec.grid_dim.x << "," << exec.grid_dim.y << "," << exec.grid_dim.z
       << ") (block: " << exec.block_dim.x << "," << exec.block_dim.y << "," << exec.block_dim.z
       << ") pre_state: " << exec.pre_state.size()
       << " post_state: " << exec.post_state.size();
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const MemoryOperation& state) {
    os << "MemoryOperation: " << state.type << " dst: " << state.dst
       << " src: " << state.src << " size: " << state.size
       << " value: " << state.value << " kind: " << state.kind
       << " execution_order: " << state.execution_order
       << " stream: " << state.stream;
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const Trace& trace) {
    os << "Trace: " << trace.kernel_executions.size() << " kernel executions" << std::endl;
    os << "Memory operations: " << trace.memory_operations.size() << std::endl;
    for (const auto& exec : trace.kernel_executions) {
        os << exec << std::endl;
    }
    for (const auto& op : trace.memory_operations) {
        os << op << std::endl;
    }
    return os;
}

class Tracer {
public:
    KernelManager& getKernelManager() { return kernel_manager_; }
    const KernelManager& getKernelManager() const { return kernel_manager_; }

    Trace trace_;

    static Tracer& instance();
    
    void recordKernelLaunch(const KernelExecution& exec);
    void recordMemoryOperation(const MemoryOperation& op);
    void flush(); // Write current trace to disk
    
    // Disable copy/move
    Tracer(const Tracer&) = delete;
    Tracer& operator=(const Tracer&) = delete;
    Tracer(Tracer&&) = delete;
    Tracer& operator=(Tracer&&) = delete;

    Tracer(const std::string& path); // Used for loading the trace from file
    ~Tracer() { std::cout << "Tracer destructor called" << std::endl; finalizeTrace(); }
    
    void finalizeTrace();

    friend std::ostream& operator<<(std::ostream& os, const Tracer& tracer) {
        os << "Tracer: " << std::endl;
        os << "Trace: " << tracer.trace_.kernel_executions.size() << " kernel executions" << std::endl;
        os << "Memory operations: " << tracer.trace_.memory_operations.size() << std::endl;
        for (const auto& exec : tracer.trace_.kernel_executions) {
            os << exec << std::endl;
        }
        for (const auto& op : tracer.trace_.memory_operations) {
            os << op << std::endl;
        }
        return os;
    }

private:
    Tracer(); // Used for recording the trace

    void initializeTraceFile();
    void writeEvent(uint32_t type, const void* data, size_t size);
    void writeKernelExecution(const KernelExecution& exec);
    void writeMemoryOperation(const MemoryOperation& op);
    void writeKernelManagerData();

    
    std::string getTraceFilePath() const;
    
    std::ofstream trace_file_;
    std::string trace_path_;
    bool initialized_;
        
    static KernelExecution readKernelExecution(std::ifstream& file);
    static MemoryOperation readMemoryOperation(std::ifstream& file);
    
    uint64_t current_execution_order_;
    KernelManager kernel_manager_;
};

} // namespace hip_intercept
#endif // HIP_INTERCEPT_LAYER_TRACER_HH

