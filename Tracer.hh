#pragma once

#include "Types.hh"
#include "Util.hh"
#include "KernelManager.hh"
#define __HIP_PLATFORM_SPIRV__
#include <hip/hip_runtime.h>

// Forward declare the real hipMemcpy function getter
typedef hipError_t (*hipMemcpy_fn)(void *, const void *, size_t, hipMemcpyKind);
hipMemcpy_fn get_real_hipMemcpy();

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
#include <iomanip>
#include "CodeGenKernelHeaders.hh"

// State of an argument, for both scalar and array data
class ArgState {
    public:
    size_t data_type_size;  // Size of each element
    size_t array_size;      // Number of elements (1 for scalars)
    std::vector<char> data; // Raw bytes

    ArgState(size_t type_size = 0, size_t arr_size = 0) 
        : data_type_size(type_size), array_size(arr_size) {
        if (type_size > 0 && arr_size > 0) {
            data.resize(type_size * arr_size);
        }
    }

    size_t total_size() const { return data_type_size * array_size; }

    void captureGpuMemory(void* ptr, size_t capture_size) {
        if (!ptr || capture_size == 0) return;
        
        // Ensure GPU operations are complete
        hipError_t err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            std::cerr << "Failed to synchronize device before memory capture: " 
                      << hipGetErrorString(err) << std::endl;
            return;
        }
        
        // Resize the data vector to accommodate the captured memory
        data_type_size = sizeof(char);
        array_size = capture_size;
        data.resize(capture_size);
        
        // Get the real hipMemcpy function
        hipMemcpy_fn real_memcpy = get_real_hipMemcpy();
        if (!real_memcpy) {
            std::cerr << "Failed to get real hipMemcpy function" << std::endl;
            return;
        }
        
        // Copy from GPU to the data buffer
        err = real_memcpy(data.data(), ptr, capture_size, hipMemcpyDeviceToHost);
        if (err != hipSuccess) {
            std::cerr << "Failed to capture GPU memory at " << ptr 
                      << " of size " << capture_size 
                      << ": " << hipGetErrorString(err) << std::endl;
            return;
        }

        // Calculate and print hash
        float checksum = calculateChecksum(data.data(), capture_size);
        std::cout << "Captured GPU memory at " << ptr << " size: " << capture_size 
                  << " checksum: " << std::hex << std::setprecision(8) << checksum << std::dec << std::endl;
    }

    void captureHostMemory(void* ptr, size_t capture_size) {
        if (!ptr || capture_size == 0) return;
        
        // Resize the data vector to accommodate the captured memory
        data_type_size = sizeof(char);
        array_size = capture_size;
        data.resize(capture_size);
        
        // Copy the host memory directly
        std::memcpy(data.data(), ptr, capture_size);

        // Calculate and print hash
        float checksum = calculateChecksum(data.data(), capture_size);
        std::cout << "Captured Host memory at " << ptr << " size: " << capture_size 
                  << " checksum: " << std::hex << std::setprecision(8) << checksum << std::dec << std::endl;
    }

    void serialize(std::ofstream& file) const {
        file.write(reinterpret_cast<const char*>(&data_type_size), sizeof(data_type_size));
        file.write(reinterpret_cast<const char*>(&array_size), sizeof(array_size));
        if (!data.empty()) {
            file.write(data.data(), data.size());
        }
    }

    static ArgState deserialize(std::ifstream& file) {
        ArgState arg;
        file.read(reinterpret_cast<char*>(&arg.data_type_size), sizeof(arg.data_type_size));
        file.read(reinterpret_cast<char*>(&arg.array_size), sizeof(arg.array_size));
        if (arg.data_type_size > 0 && arg.array_size > 0) {
            arg.data.resize(arg.data_type_size * arg.array_size);
            file.read(arg.data.data(), arg.data.size());
        }
        return arg;
    }
};

enum class OperationType : uint32_t {
    KERNEL = 0x4B524E4C,  // 'KRNL' in ASCII
    MEMORY = 0x4D454D4F   // 'MEMO' in ASCII
};

class Operation {
public:
    std::vector<ArgState> pre_args;
    std::vector<ArgState> post_args;
    OperationType type;

    bool isKernel() const { return type == OperationType::KERNEL; }
    bool isMemory() const { return type == OperationType::MEMORY; }
    
    virtual ~Operation() = default;
    
    friend std::ostream& operator<<(std::ostream& os, const Operation& op);

    Operation(std::vector<ArgState> pre, std::vector<ArgState> post, OperationType type)
        : pre_args(std::move(pre)), post_args(std::move(post)), type(type) {}

    Operation() : pre_args(), post_args() {}

    static std::shared_ptr<Operation> deserialize(std::ifstream& file);

    virtual void serialize(std::ofstream& file) const = 0;

protected:
    virtual void deserializeImpl(std::ifstream& file) = 0;
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
    std::vector<void*> arg_ptrs;
    std::vector<size_t> arg_sizes;  // Sizes of pointer arguments
    
    // Add default constructor
    KernelExecution() : Operation(std::vector<ArgState>(), std::vector<ArgState>(), OperationType::KERNEL), 
        function_address(nullptr),
        kernel_name(),
        grid_dim(),
        block_dim(),
        shared_mem(0),
        stream(nullptr) {
    }

    // Existing constructor
    KernelExecution(std::vector<ArgState> pre, 
                   std::vector<ArgState> post,
                   void* function_address,
                   std::string kernel_name,
                   dim3 grid_dim,
                   dim3 block_dim,
                   size_t shared_mem,
                   hipStream_t stream)
        : Operation(std::move(pre), std::move(post), OperationType::KERNEL),
          function_address(function_address),
          kernel_name(kernel_name),
          grid_dim(grid_dim),
          block_dim(block_dim),
          shared_mem(shared_mem),
          stream(stream) {
    }
    
    // Replace operator<< with writeToStream
    void writeToStream(std::ostream& os) const override {
        os << "KernelExecution: " << kernel_name 
           << " (grid: " << grid_dim.x << "," << grid_dim.y << "," << grid_dim.z
           << ") (block: " << block_dim.x << "," << block_dim.y << "," << block_dim.z
           << ") pre_state: " << (pre_args.size() > 0 ? "present" : "null")
           << " post_state: " << (post_args.size() > 0 ? "present" : "null");
    }

    virtual void serialize(std::ofstream& file) const override {
        std::cout << "Serializing KernelExecution" << std::endl;
        // Write type identifier
        OperationType op_type = OperationType::KERNEL;
        file.write(reinterpret_cast<const char*>(&op_type), sizeof(op_type));
        
        // Write kernel name length and data
        uint32_t name_length = static_cast<uint32_t>(kernel_name.length());
        std::cout << "Serializing kernel name: " << kernel_name << " length: " << name_length << std::endl;
        file.write(reinterpret_cast<const char*>(&name_length), sizeof(name_length));
        file.write(kernel_name.c_str(), name_length);
        
        // Write kernel data
        file.write(reinterpret_cast<const char*>(&function_address), sizeof(function_address));
        file.write(reinterpret_cast<const char*>(&grid_dim), sizeof(grid_dim));
        file.write(reinterpret_cast<const char*>(&block_dim), sizeof(block_dim));
        file.write(reinterpret_cast<const char*>(&shared_mem), sizeof(shared_mem));
        file.write(reinterpret_cast<const char*>(&stream), sizeof(stream));
        
        // Write argument data
        uint32_t num_args = static_cast<uint32_t>(arg_ptrs.size());
        file.write(reinterpret_cast<const char*>(&num_args), sizeof(num_args));
        for (void* ptr : arg_ptrs) {
            file.write(reinterpret_cast<const char*>(&ptr), sizeof(void*));
        }

        // Write argument sizes
        uint32_t num_sizes = static_cast<uint32_t>(arg_sizes.size());
        file.write(reinterpret_cast<const char*>(&num_sizes), sizeof(num_sizes));
        for (size_t size : arg_sizes) {
            file.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
        }
        
        // Write memory states
        bool has_pre_state = pre_args.size() > 0;
        file.write(reinterpret_cast<const char*>(&has_pre_state), sizeof(has_pre_state));
        if (has_pre_state) {
            for (const auto& arg : pre_args) {
                arg.serialize(file);
            }
        }

        bool has_post_state = post_args.size() > 0;
        file.write(reinterpret_cast<const char*>(&has_post_state), sizeof(has_post_state));
        if (has_post_state) {
            for (const auto& arg : post_args) {
                arg.serialize(file);
            }
        }
    }

    static std::shared_ptr<KernelExecution> create_from_file(std::ifstream& file) {
        auto exec = std::make_shared<KernelExecution>();
        exec->deserializeImpl(file);
        return exec;
    }

    protected:
    void deserializeImpl(std::ifstream& file) override {
        std::cout << "Deserializing KernelExecution" << std::endl;
        
        // Read and verify operation type
        OperationType op_type;
        file.read(reinterpret_cast<char*>(&op_type), sizeof(op_type));
        if (op_type != OperationType::KERNEL) {
            std::cerr << "Invalid operation type during KernelExecution deserialization" << std::endl;
            throw std::runtime_error("Invalid operation type");
        }
        
        // Read kernel name
        uint32_t name_length;
        file.read(reinterpret_cast<char*>(&name_length), sizeof(name_length));
        std::cout << "Reading kernel name of length: " << name_length << std::endl;
        
        if (name_length > 0 && name_length < 1024) {  // Add reasonable size limit
            std::vector<char> name_buffer(name_length + 1, '\0');
            file.read(name_buffer.data(), name_length);
            kernel_name = std::string(name_buffer.data(), name_length);
            std::cout << "Deserialized kernel name: " << kernel_name << std::endl;
        } else {
            std::cerr << "Warning: Invalid kernel name length: " << name_length << std::endl;
            throw std::runtime_error("Invalid kernel name length");
        }
        
        // Read kernel data
        file.read(reinterpret_cast<char*>(&function_address), sizeof(function_address));
        file.read(reinterpret_cast<char*>(&grid_dim), sizeof(grid_dim));
        file.read(reinterpret_cast<char*>(&block_dim), sizeof(block_dim));
        file.read(reinterpret_cast<char*>(&shared_mem), sizeof(shared_mem));
        file.read(reinterpret_cast<char*>(&stream), sizeof(stream));
        
        // Read arguments
        uint32_t num_args;
        file.read(reinterpret_cast<char*>(&num_args), sizeof(num_args));
        if (num_args > 1024) {  // Add reasonable size limit
            throw std::runtime_error("Invalid number of arguments");
        }
        arg_ptrs.resize(num_args);
        for (uint32_t i = 0; i < num_args; i++) {
            file.read(reinterpret_cast<char*>(&arg_ptrs[i]), sizeof(void*));
        }

        // Read argument sizes
        uint32_t num_sizes;
        file.read(reinterpret_cast<char*>(&num_sizes), sizeof(num_sizes));
        if (num_sizes > 1024) {  // Add reasonable size limit
            throw std::runtime_error("Invalid number of argument sizes");
        }
        arg_sizes.resize(num_sizes);
        for (uint32_t i = 0; i < num_sizes; i++) {
            file.read(reinterpret_cast<char*>(&arg_sizes[i]), sizeof(size_t));
        }

        // Read memory states
        bool has_pre_state;
        file.read(reinterpret_cast<char*>(&has_pre_state), sizeof(has_pre_state));
        if (has_pre_state) {
            pre_args.clear();
            for (uint32_t i = 0; i < num_args; i++) {
                pre_args.push_back(ArgState::deserialize(file));
            }
        }

        bool has_post_state;
        file.read(reinterpret_cast<char*>(&has_post_state), sizeof(has_post_state));
        if (has_post_state) {
            post_args.clear();
            for (uint32_t i = 0; i < num_args; i++) {
                post_args.push_back(ArgState::deserialize(file));
            }
        }
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
    MemoryOperation() : Operation(std::vector<ArgState>(), std::vector<ArgState>(), OperationType::MEMORY),
        type(MemoryOpType::COPY),
        dst(nullptr),
        src(nullptr),
        size(0),
        value(0),
        kind(hipMemcpyDefault),
        stream(nullptr) {
        }

    // Existing constructor
    MemoryOperation(std::vector<ArgState> pre,
                   std::vector<ArgState> post,
                   MemoryOpType type,
                   void* dst,
                   const void* src,
                   size_t size,
                   int value,
                   hipMemcpyKind kind,
                   hipStream_t stream)
        : Operation(std::move(pre), std::move(post), OperationType::MEMORY),
          type(type),
          dst(dst),
          src(src),
          size(size),
          value(value),
          kind(kind),
          stream(stream) {
        }

    std::string kindToString(hipMemcpyKind kind) const {
        switch (kind) {
            case hipMemcpyHostToHost: return "hipMemcpyHostToHost";
            case hipMemcpyHostToDevice: return "hipMemcpyHostToDevice";
            case hipMemcpyDeviceToHost: return "hipMemcpyDeviceToHost";
            case hipMemcpyDeviceToDevice: return "hipMemcpyDeviceToDevice";
            case hipMemcpyDefault: return "hipMemcpyDefault";
            default: return "Unknown";
        }
    }
        
    // Replace operator<< with writeToStream
    void writeToStream(std::ostream& os) const override {
        os << "MemoryOperation: " << static_cast<int>(type) << " dst: " << dst
           << " src: " << src << " size: " << size
           << " value: " << value << " kind: " << kindToString(kind)
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
        
        // Write memory states
        bool has_pre_state = pre_args.size() > 0;
        file.write(reinterpret_cast<const char*>(&has_pre_state), sizeof(has_pre_state));
        if (has_pre_state) {
            for (const auto& arg : pre_args) {
                arg.serialize(file);
            }
        }

        bool has_post_state = post_args.size() > 0;
        file.write(reinterpret_cast<const char*>(&has_post_state), sizeof(has_post_state));
        if (has_post_state) {
            for (const auto& arg : post_args) {
                arg.serialize(file);
            }
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
        // Read and verify operation type
        OperationType op_type;
        file.read(reinterpret_cast<char*>(&op_type), sizeof(op_type));
        if (op_type != OperationType::MEMORY) {
            std::cerr << "Invalid operation type during MemoryOperation deserialization" << std::endl;
            throw std::runtime_error("Invalid operation type");
        }
        
        // Read memory operation data
        file.read(reinterpret_cast<char*>(&type), sizeof(type));
        file.read(reinterpret_cast<char*>(&dst), sizeof(dst));
        file.read(reinterpret_cast<char*>(&src), sizeof(src));
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        file.read(reinterpret_cast<char*>(&value), sizeof(value));
        file.read(reinterpret_cast<char*>(&kind), sizeof(kind));
        file.read(reinterpret_cast<char*>(&stream), sizeof(stream));
        
        // Read memory states
        bool has_pre_state;
        file.read(reinterpret_cast<char*>(&has_pre_state), sizeof(has_pre_state));
        if (has_pre_state) {
            pre_args.clear();
            pre_args.push_back(ArgState::deserialize(file));
        }

        bool has_post_state;
        file.read(reinterpret_cast<char*>(&has_post_state), sizeof(has_post_state));
        if (has_post_state) {
            post_args.clear();
            post_args.push_back(ArgState::deserialize(file));
        }
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
            finalizeTrace(file_path);
        }
    }
    
    void finalizeTrace(std::string final_trace_path = "");

    friend std::ostream& operator<<(std::ostream& os, const Tracer& tracer) {
        os << "Tracer: " << std::endl;
        os << "Trace: " << tracer.trace_.operations.size() << " operations" << std::endl;
        for (const auto& op : tracer.trace_.operations) {
            auto idx = &op - &tracer.trace_.operations[0];
            if (auto kernel_exec = dynamic_cast<const KernelExecution*>(op.get())) {
                os << "Op#" << idx << " " << *kernel_exec << std::endl;
            } else if (auto mem_op = dynamic_cast<const MemoryOperation*>(op.get())) {
                os << "Op#" << idx << " " << *mem_op << std::endl;
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

