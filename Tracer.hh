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
    static constexpr uint32_t MAGIC_START = 0x4D454D53; // 'MEMS'
    static constexpr uint32_t MAGIC_END = 0x454D454D;   // 'EMEM'
    
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
        // Write start magic
        file.write(reinterpret_cast<const char*>(&MAGIC_START), sizeof(MAGIC_START));
        
        // Write size
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        
        // Write data if present
        if (size > 0 && data) {
            file.write(data.get(), size);
        }
        
        // Write end magic
        file.write(reinterpret_cast<const char*>(&MAGIC_END), sizeof(MAGIC_END));
    }

    static MemoryState deserialize(std::ifstream& file) {
        // Read and verify start magic
        uint32_t magic_start;
        file.read(reinterpret_cast<char*>(&magic_start), sizeof(magic_start));
        if (magic_start != MAGIC_START) {
            std::cerr << "Invalid MemoryState start magic: 0x" 
                      << std::hex << magic_start << std::dec 
                      << " (expected 0x" << std::hex << MAGIC_START << std::dec << ")" 
                      << std::endl;
            throw std::runtime_error("Invalid MemoryState format");
        }
        
        // Read size
        size_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        
        // Create state and read data
        MemoryState state(size);
        if (size > 0) {
            file.read(state.data.get(), size);
        }
        
        // Read and verify end magic
        uint32_t magic_end;
        file.read(reinterpret_cast<char*>(&magic_end), sizeof(magic_end));
        if (magic_end != MAGIC_END) {
            std::cerr << "Invalid MemoryState end magic: 0x" 
                      << std::hex << magic_end << std::dec 
                      << " (expected 0x" << std::hex << MAGIC_END << std::dec << ")" 
                      << std::endl;
            throw std::runtime_error("Invalid MemoryState format");
        }
        
        return state;
    }
};

// Forward declarations
class KernelExecution;
class MemoryOperation;

enum class OperationType : uint32_t {
    KERNEL = 0x4B524E4C,  // 'KRNL' in ASCII
    MEMORY = 0x4D454D4F   // 'MEMO' in ASCII
};

class Operation {
public:
    std::shared_ptr<MemoryState> pre_state;
    std::shared_ptr<MemoryState> post_state;
    mutable size_t execution_order;
    OperationType type;

    bool isKernel() const { return type == OperationType::KERNEL; }
    bool isMemory() const { return type == OperationType::MEMORY; }
    
    // Add virtual destructor
    virtual ~Operation() = default;
    
    // Make this a friend function instead of a member function
    friend std::ostream& operator<<(std::ostream& os, const Operation& op);

    Operation(std::shared_ptr<MemoryState> pre_state, std::shared_ptr<MemoryState> post_state, OperationType type)
        : pre_state(pre_state), post_state(post_state), type(type) {}

    Operation() : pre_state(nullptr), post_state(nullptr), execution_order(0) {}

    // Make this const
    void setExecutionOrder(size_t order) const {
        execution_order = order;
    }

    size_t getExecutionOrder() const {
        return execution_order;
    }

    static std::shared_ptr<Operation> deserialize(std::ifstream& file);

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
    KernelExecution() : Operation(nullptr, nullptr, OperationType::KERNEL), 
        function_address(nullptr),
        kernel_name(),
        grid_dim(),
        block_dim(),
        shared_mem(0),
        stream(nullptr) {
        pre_state = std::make_shared<MemoryState>(1);
        post_state = std::make_shared<MemoryState>(1);
    }

    // Existing constructor
    KernelExecution(std::shared_ptr<MemoryState> pre_state, 
                   std::shared_ptr<MemoryState> post_state,
                   void* function_address,
                   std::string kernel_name,
                   dim3 grid_dim,
                   dim3 block_dim,
                   size_t shared_mem,
                   hipStream_t stream)
        : Operation(pre_state, post_state, OperationType::KERNEL),
          function_address(function_address),
          kernel_name(kernel_name),
          grid_dim(grid_dim),
          block_dim(block_dim),
          shared_mem(shared_mem),
          stream(stream) {
            pre_state = std::make_shared<MemoryState>(1);
            post_state = std::make_shared<MemoryState>(1);
          }
    
    std::vector<void*> arg_ptrs;
    std::vector<size_t> arg_sizes;

    // Replace operator<< with writeToStream
    void writeToStream(std::ostream& os) const override {
        os << "KernelExecution: " << kernel_name 
           << " (grid: " << grid_dim.x << "," << grid_dim.y << "," << grid_dim.z
           << ") (block: " << block_dim.x << "," << block_dim.y << "," << block_dim.z
           << ") pre_state: " << (pre_state ? "present" : "null")
           << " post_state: " << (post_state ? "present" : "null");
    }

    virtual void serialize(std::ofstream& file) const override {
        std::cout << "Serializing KernelExecution" << std::endl;
        // Write type identifier
        OperationType op_type = OperationType::KERNEL;
        file.write(reinterpret_cast<const char*>(&op_type), sizeof(op_type));
        
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
        pre_state.get()->serialize(file);
        post_state.get()->serialize(file);
    }

    static std::shared_ptr<KernelExecution> create_from_file(std::ifstream& file) {
        auto exec = std::make_shared<KernelExecution>();
        exec->deserializeImpl(file);
        return exec;
    }

    protected:
    void deserializeImpl(std::ifstream& file) override {
        std::cout << "Deserializing KernelExecution" << std::endl;
        // Read kernel name
        uint32_t name_length;
        file.read(reinterpret_cast<char*>(&name_length), sizeof(name_length));
        std::vector<char> name_buffer(name_length + 1);
        file.read(name_buffer.data(), name_length);
        name_buffer[name_length] = '\0';
        kernel_name = std::string(name_buffer.data());
        if (kernel_name.empty()) {
            std::cerr << "Kernel name is empty" << std::endl;
            std::abort();
        } else {
            std::cout << "deserialized Kernel name: " << kernel_name << std::endl;
        }
        
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
    MemoryOperation() : Operation(nullptr, nullptr, OperationType::MEMORY),
        type(MemoryOpType::COPY),
        dst(nullptr),
        src(nullptr),
        size(0),
        value(0),
        kind(hipMemcpyHostToHost),
        stream(nullptr) {
            pre_state = std::make_shared<MemoryState>(1);
            post_state = std::make_shared<MemoryState>(1);
        }

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
        : Operation(pre_state, post_state, OperationType::MEMORY),
          type(type),
          dst(dst),
          src(src),
          size(size),
          value(value),
          kind(kind),
          stream(stream) {
            pre_state = std::make_shared<MemoryState>(1);
            post_state = std::make_shared<MemoryState>(1);
        }

    // Replace operator<< with writeToStream
    void writeToStream(std::ostream& os) const override {
        os << "MemoryOperation: " << static_cast<int>(type) << " dst: " << dst
           << " src: " << src << " size: " << size
           << " value: " << value << " kind: " << kind
           << " execution_order: " << execution_order
           << " stream: " << stream;
    }

    virtual void serialize(std::ofstream& file) const override {
        std::cout << "Serializing MemoryOperation" << std::endl;
        // Write type identifier
        OperationType op_type = OperationType::MEMORY;
        file.write(reinterpret_cast<const char*>(&op_type), sizeof(op_type));
        
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
        if (pre_state) {
            pre_state->serialize(file);
        } else {
            size_t size = 0;
            file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        }

        if (post_state) {
            post_state->serialize(file);
        } else {
            size_t size = 0;
            file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        }
    }

    static std::shared_ptr<MemoryOperation> create_from_file(std::ifstream& file) {
        auto op = std::make_shared<MemoryOperation>();
        op->deserializeImpl(file);
        return op;
    }

    protected:
    void deserializeImpl(std::ifstream& file) override {
        std::cout << "Deserializing MemoryOperation" << std::endl;
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
};

class Trace {
    public:
    std::vector<std::shared_ptr<Operation>> operations;  // Change to store shared_ptrs
    uint64_t timestamp;

    void operator<<(std::ostream& os) const {
        os << "Trace: " << operations.size() << " operations" << std::endl;
        for (const auto& op : operations) {
            os << *op << std::endl;
        }
    }

    void addOperation(std::shared_ptr<Operation> op) {  // Take ownership of operation
        op->setExecutionOrder(operations.size());
        operations.push_back(std::move(op));
    }
};


class Tracer {
    bool serialize_trace_ = true;
    std::string file_path;
    std::ofstream trace_file_;
    bool initialized_;
public:
    void setFilePath(const std::string& path) { file_path = path; }
    KernelManager& getKernelManager() { return kernel_manager_; }
    const KernelManager& getKernelManager() const { return kernel_manager_; }

    Trace trace_;
    size_t getNumOperations() const { return trace_.operations.size(); }
    std::shared_ptr<Operation> getOperation(size_t index) const { 
        assert(index < trace_.operations.size());
        return trace_.operations[index]; 
    }
    std::vector<size_t> getOperationsIdxByName(const std::string& name) const {
        std::vector<size_t> indices;
        for (size_t i = 0; i < trace_.operations.size(); ++i) {
            auto kernel_exec = dynamic_cast<const KernelExecution*>(trace_.operations[i].get());
            if (kernel_exec && kernel_exec->kernel_name == name) {
                indices.push_back(i);
            }
        }
        return indices;
    }
    static Tracer& instance();
    
    void recordKernelLaunch(const KernelExecution& exec);
    void recordMemoryOperation(const MemoryOperation& op);
    void flush(); // Write current trace to disk
    
    // Disable copy/move
    Tracer(const Tracer&) = delete;
    Tracer& operator=(const Tracer&) = delete;
    Tracer(Tracer&&) = delete;
    Tracer& operator=(Tracer&&) = delete;
    void setSerializeTrace(bool serialize) { serialize_trace_ = serialize; }

    Tracer(const std::string& path); // Used for loading the trace from file
    Tracer(); // Used for recording the trace
    void initializeTraceFile();

    ~Tracer() { 
        std::cout << "Tracer destructor called" << std::endl; 
        if (serialize_trace_) {
            std::cout << "Serializing trace" << std::endl;
            finalizeTrace();
        }
    }
    
    void finalizeTrace();

    friend std::ostream& operator<<(std::ostream& os, const Tracer& tracer) {
        os << "Tracer: " << std::endl;
        os << "Trace: " << tracer.trace_.operations.size() << " operations" << std::endl;
        for (const auto& op : tracer.trace_.operations) {
            if (auto kernel_exec = dynamic_cast<const KernelExecution*>(op.get())) {
                os << "Op#" << op->execution_order << " " << *kernel_exec << std::endl;
            } else if (auto mem_op = dynamic_cast<const MemoryOperation*>(op.get())) {
                os << "Op#" << op->execution_order << " " << *mem_op << std::endl;
            }
        }
        return os;
    }

    Trace getTrace() && {
        return std::move(trace_);
    }

private:
    void writeEvent(uint32_t type, const void* data, size_t size);
    void writeKernelExecution(const KernelExecution& exec);
    void writeMemoryOperation(const MemoryOperation& op);
    void writeKernelManagerData();

    
    std::string getTraceFilePath() const;
    

        
    static KernelExecution readKernelExecution(std::ifstream& file);
    static MemoryOperation readMemoryOperation(std::ifstream& file);
    
    uint64_t current_execution_order_;
    KernelManager kernel_manager_;
};

#endif // HIP_INTERCEPT_LAYER_TRACER_HH

