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

// Memory state tracking
class MemoryState {
public:
    std::unique_ptr<char[]> data;
    size_t size;
    
    explicit MemoryState(size_t s) : data(new char[s]), size(s) {}
    MemoryState(const char* src, size_t s) : data(new char[s]), size(s) {
        memcpy(data.get(), src, s);
    }
    MemoryState() : size(0) {}
    
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

    void serialize(std::ofstream& file) const {
        file.write(data.get(), size);
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    }

    static MemoryState deserialize(std::ifstream& file) {
        size_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        
        MemoryState state(size);
        if (size > 0) {
            file.read(state.data.get(), size);
        }
        return state;
    }
};

// Forward declarations
class KernelExecution;
class MemoryOperation;

enum class OperationType {
    KERNEL,
    MEMORY
};

class Operation {
public:
    std::shared_ptr<MemoryState> pre_state;
    std::shared_ptr<MemoryState> post_state;
    mutable size_t execution_order;
    
    // Add virtual destructor
    virtual ~Operation() = default;
    
    // Make this a friend function instead of a member function
    friend std::ostream& operator<<(std::ostream& os, const Operation& op);

    Operation(std::shared_ptr<MemoryState> pre_state, std::shared_ptr<MemoryState> post_state)
        : pre_state(pre_state), post_state(post_state) {}

    Operation() : pre_state(nullptr), post_state(nullptr), execution_order(0) {}

    // Make this const
    void setExecutionOrder(size_t order) const {
        execution_order = order;
    }

    size_t getExecutionOrder() const {
        return execution_order;
    }

    static std::unique_ptr<Operation> deserialize(std::ifstream& file);
    virtual void serialize(std::ofstream& file) const = 0;
    
protected:
    virtual void deserializeImpl(std::ifstream& file) = 0;
    // Add pure virtual function for stream output
    virtual void writeToStream(std::ostream& os) const = 0;
};

// Define the stream operator
inline std::ostream& operator<<(std::ostream& os, const Operation& op) {
    op.writeToStream(os);
    return os;
}

class KernelExecution : public Operation {
    public:
    void* function_address;
    std::string kernel_name;
    dim3 grid_dim;
    dim3 block_dim;
    size_t shared_mem;
    hipStream_t stream;
    
    // Add default constructor
    KernelExecution() : Operation(), 
        function_address(nullptr),
        kernel_name(),
        grid_dim(),
        block_dim(),
        shared_mem(0),
        stream(nullptr) {}

    // Existing constructor
    KernelExecution(std::shared_ptr<MemoryState> pre_state, 
                   std::shared_ptr<MemoryState> post_state,
                   void* function_address,
                   std::string kernel_name,
                   dim3 grid_dim,
                   dim3 block_dim,
                   size_t shared_mem,
                   hipStream_t stream)
        : Operation(pre_state, post_state),
          function_address(function_address),
          kernel_name(kernel_name),
          grid_dim(grid_dim),
          block_dim(block_dim),
          shared_mem(shared_mem),
          stream(stream) {}
    
    std::vector<std::pair<void*, size_t>> changes;
    std::vector<void*> arg_ptrs;
    std::vector<size_t> arg_sizes;
    std::map<int, std::vector<std::pair<size_t, std::pair<float, float>>>> changes_by_arg;

    // Replace operator<< with writeToStream
    void writeToStream(std::ostream& os) const override {
        os << "KernelExecution: " << kernel_name 
           << " (grid: " << grid_dim.x << "," << grid_dim.y << "," << grid_dim.z
           << ") (block: " << block_dim.x << "," << block_dim.y << "," << block_dim.z
           << ") pre_state: " << (pre_state ? "present" : "null")
           << " post_state: " << (post_state ? "present" : "null");
    }

    virtual void serialize(std::ofstream& file) const override {
        // Write type identifier
        OperationType type = OperationType::KERNEL;
        file.write(reinterpret_cast<const char*>(&type), sizeof(type));
        
        // Write kernel name length and data
        uint32_t name_length = static_cast<uint32_t>(kernel_name.length());
        file.write(reinterpret_cast<const char*>(&name_length), sizeof(name_length));
        file.write(kernel_name.c_str(), name_length);
        
        // Write kernel data
        file.write(reinterpret_cast<const char*>(&function_address), sizeof(function_address));
        file.write(reinterpret_cast<const char*>(&grid_dim), sizeof(grid_dim));
        file.write(reinterpret_cast<const char*>(&block_dim), sizeof(block_dim));
        file.write(reinterpret_cast<const char*>(&shared_mem), sizeof(shared_mem));
        file.write(reinterpret_cast<const char*>(&stream), sizeof(stream));
        file.write(reinterpret_cast<const char*>(&execution_order), sizeof(execution_order));
        
        // Write argument data
        uint32_t num_args = static_cast<uint32_t>(arg_ptrs.size());
        file.write(reinterpret_cast<const char*>(&num_args), sizeof(num_args));
        for (void* ptr : arg_ptrs) {
            file.write(reinterpret_cast<const char*>(&ptr), sizeof(void*));
        }
        
        // Write memory states if present
        if (pre_state) {
            pre_state->serialize(file);
        }
        if (post_state) {
            post_state->serialize(file);
        }
    }

    protected:
    void deserializeImpl(std::ifstream& file) override {
        // Read kernel name
        uint32_t name_length;
        file.read(reinterpret_cast<char*>(&name_length), sizeof(name_length));
        std::vector<char> name_buffer(name_length + 1);
        file.read(name_buffer.data(), name_length);
        name_buffer[name_length] = '\0';
        kernel_name = std::string(name_buffer.data());
        
        // Read kernel data
        file.read(reinterpret_cast<char*>(&function_address), sizeof(function_address));
        file.read(reinterpret_cast<char*>(&grid_dim), sizeof(grid_dim));
        file.read(reinterpret_cast<char*>(&block_dim), sizeof(block_dim));
        file.read(reinterpret_cast<char*>(&shared_mem), sizeof(shared_mem));
        file.read(reinterpret_cast<char*>(&stream), sizeof(stream));
        file.read(reinterpret_cast<char*>(&execution_order), sizeof(execution_order));
        
        // Read arguments
        uint32_t num_args;
        file.read(reinterpret_cast<char*>(&num_args), sizeof(num_args));
        arg_ptrs.resize(num_args);
        for (uint32_t i = 0; i < num_args; i++) {
            file.read(reinterpret_cast<char*>(&arg_ptrs[i]), sizeof(void*));
        }
        
        // Read memory states
        pre_state = std::make_shared<MemoryState>(MemoryState::deserialize(file));
        post_state = std::make_shared<MemoryState>(MemoryState::deserialize(file));
    }

    // Static factory method becomes a helper
    static std::unique_ptr<KernelExecution> create_from_file(std::ifstream& file) {
        auto exec = std::make_unique<KernelExecution>();
        exec->deserializeImpl(file);
        return exec;
    }
};

class MemoryOperation : public Operation {
    public:
    MemoryOpType type;
    void* dst;
    const void* src;
    size_t size;
    int value;
    hipMemcpyKind kind;
    hipStream_t stream;

    // Add default constructor
    MemoryOperation() : Operation(),
        type(MemoryOpType::COPY),
        dst(nullptr),
        src(nullptr),
        size(0),
        value(0),
        kind(hipMemcpyHostToHost),
        stream(nullptr) {}

    // Existing constructor
    MemoryOperation(std::shared_ptr<MemoryState> pre_state,
                   std::shared_ptr<MemoryState> post_state,
                   MemoryOpType type,
                   void* dst,
                   const void* src,
                   size_t size,
                   int value,
                   hipMemcpyKind kind,
                   hipStream_t stream)
        : Operation(pre_state, post_state),
          type(type),
          dst(dst),
          src(src),
          size(size),
          value(value),
          kind(kind),
          stream(stream) {}

    // Replace operator<< with writeToStream
    void writeToStream(std::ostream& os) const override {
        os << "MemoryOperation: " << static_cast<int>(type) << " dst: " << dst
           << " src: " << src << " size: " << size
           << " value: " << value << " kind: " << kind
           << " execution_order: " << execution_order
           << " stream: " << stream;
    }

    virtual void serialize(std::ofstream& file) const override {
        // Write type identifier
        OperationType type = OperationType::MEMORY;
        file.write(reinterpret_cast<const char*>(&type), sizeof(type));
        
        // Write memory operation data
        file.write(reinterpret_cast<const char*>(&type), sizeof(type));
        file.write(reinterpret_cast<const char*>(&dst), sizeof(dst));
        file.write(reinterpret_cast<const char*>(&src), sizeof(src));
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        file.write(reinterpret_cast<const char*>(&value), sizeof(value));
        file.write(reinterpret_cast<const char*>(&kind), sizeof(kind));
        file.write(reinterpret_cast<const char*>(&stream), sizeof(stream));
        file.write(reinterpret_cast<const char*>(&execution_order), sizeof(execution_order));
        
        // Write memory states
        pre_state->serialize(file);
        post_state->serialize(file);
    }

    protected:
    void deserializeImpl(std::ifstream& file) override {
        // Read memory operation data
        file.read(reinterpret_cast<char*>(&type), sizeof(type));
        file.read(reinterpret_cast<char*>(&dst), sizeof(dst));
        file.read(reinterpret_cast<char*>(&src), sizeof(src));
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        file.read(reinterpret_cast<char*>(&value), sizeof(value));
        file.read(reinterpret_cast<char*>(&kind), sizeof(kind));
        file.read(reinterpret_cast<char*>(&stream), sizeof(stream));
        file.read(reinterpret_cast<char*>(&execution_order), sizeof(execution_order));
        
        // Read memory states
        pre_state = std::make_shared<MemoryState>(MemoryState::deserialize(file));
        post_state = std::make_shared<MemoryState>(MemoryState::deserialize(file));
    }

    static std::unique_ptr<MemoryOperation> create_from_file(std::ifstream& file) {
        auto op = std::make_unique<MemoryOperation>();
        op->deserializeImpl(file);
        return op;
    }
};

class Trace {
    public:
    std::vector<std::unique_ptr<Operation>> operations;  // Change to store unique_ptrs
    uint64_t timestamp;

    void operator<<(std::ostream& os) const {
        os << "Trace: " << operations.size() << " operations" << std::endl;
        for (const auto& op : operations) {
            os << *op << std::endl;
        }
    }

    void addOperation(std::unique_ptr<Operation> op) {  // Take ownership of operation
        op->setExecutionOrder(operations.size());
        operations.push_back(std::move(op));
    }
};


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
        os << "Trace: " << tracer.trace_.operations.size() << " operations" << std::endl;
        for (const auto& op : tracer.trace_.operations) {
            if (auto kernel_exec = dynamic_cast<const KernelExecution*>(op.get())) {
                os << *kernel_exec << std::endl;
            } else if (auto mem_op = dynamic_cast<const MemoryOperation*>(op.get())) {
                os << *mem_op << std::endl;
            }
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

#endif // HIP_INTERCEPT_LAYER_TRACER_HH

