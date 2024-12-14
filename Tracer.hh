#pragma once

#include "Types.hh"
#include "Util.hh"

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
    
    std::map<void*, MemoryState> pre_state;
    std::map<void*, MemoryState> post_state;
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

class Tracer {
public:
    static Tracer& instance();
    
    void recordKernelLaunch(const KernelExecution& exec);
    void recordMemoryOperation(const MemoryOperation& op);
    void flush(); // Write current trace to disk
    
    // Disable copy/move
    Tracer(const Tracer&) = delete;
    Tracer& operator=(const Tracer&) = delete;
    Tracer(Tracer&&) = delete;
    Tracer& operator=(Tracer&&) = delete;

    static Trace loadTrace(const std::string& path);
    
private:
    Tracer(); // Private constructor for singleton
    ~Tracer();
    
    void initializeTraceFile();
    void writeEvent(uint32_t type, const void* data, size_t size);
    void writeKernelExecution(const KernelExecution& exec);
    void writeMemoryOperation(const MemoryOperation& op);
    
    std::string getTraceFilePath() const;
    
    std::ofstream trace_file_;
    std::string trace_path_;
    bool initialized_;
    static constexpr uint32_t TRACE_MAGIC = 0x48495054; // "HIPT"
    static constexpr uint32_t TRACE_VERSION = 1;
        
    static KernelExecution readKernelExecution(std::ifstream& file);
    static MemoryOperation readMemoryOperation(std::ifstream& file);
    
    uint64_t current_execution_order_;
};

} // namespace hip_intercept

#endif // HIP_INTERCEPT_LAYER_TRACER_HH
