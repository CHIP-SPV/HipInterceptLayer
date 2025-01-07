#ifndef HIP_INTERCEPT_LAYER_INTERCEPTOR_HH
#define HIP_INTERCEPT_LAYER_INTERCEPTOR_HH

#include "Tracer.hh"
#include "KernelManager.hh"
#include <string>
#include <vector>
#include <unordered_map>

// Forward declarations
struct dim3;
struct hipDeviceProp_t;

// GPU allocation tracking
class AllocationInfo {
public:
    size_t size;
    std::unique_ptr<char[]> shadow_copy;
    
    AllocationInfo(size_t s) : size(s), shadow_copy(new char[s]) {}
};

class Interceptor {
    std::unordered_map<void*, AllocationInfo> gpu_allocations;
    std::string exe_path;

public:
    static Interceptor& instance() {
        static Interceptor instance;
        return instance;
    }

    ~Interceptor() {
        gpu_allocations.clear();
    }

    const std::string& getExePath() const {
        return exe_path;
    }

    void setExePath(const std::string& path) {
        exe_path = path;
    }

    AllocationInfo& addAllocation(void* ptr, size_t size) {
        auto [it, inserted] = gpu_allocations.emplace(ptr, AllocationInfo(size));
        return it->second;
    }

    void removeAllocation(void* ptr) {
        auto it = gpu_allocations.find(ptr);
        if (it != gpu_allocations.end()) {
            gpu_allocations.erase(it);
        }
    }

    std::pair<void*, AllocationInfo*> findContainingAllocation(void* ptr) {
        auto it = gpu_allocations.find(ptr);
        if (it != gpu_allocations.end()) {
            return std::make_pair(it->first, &it->second);
        }
        std::cerr << "Allocation not found for pointer: " << ptr << std::endl;
        std::abort();
    }
};

// External C interface declarations
extern "C" {
    hipError_t hipMalloc(void **ptr, size_t size);
    hipError_t hipLaunchKernel(const void *function_address, dim3 numBlocks,
                              dim3 dimBlocks, void **args, size_t sharedMemBytes,
                              hipStream_t stream);
    hipError_t hipDeviceSynchronize(void);
    hipError_t hipFree(void* ptr);
    hipError_t hipMemcpy(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind);
    hipError_t hipMemset(void *dst, int value, size_t sizeBytes);
    hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX,
                                    unsigned int gridDimY, unsigned int gridDimZ,
                                    unsigned int blockDimX, unsigned int blockDimY,
                                    unsigned int blockDimZ, unsigned int sharedMemBytes,
                                    hipStream_t stream, void** kernelParams,
                                    void** extra);
    hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, 
                             hipMemcpyKind kind, hipStream_t stream);
}

// Helper functions for state capture
void capturePreState(KernelExecution& exec, const Kernel& kernel, void** args) {
    const auto& arguments = kernel.getArguments();
    
    std::cout << "\nPRE-EXECUTION ARGUMENT VALUES:" << std::endl;
    
    // First pass: store argument info
    for (size_t i = 0; i < arguments.size(); i++) {
        const auto& arg = arguments[i];
        void* param_value = args[i];
        exec.arg_ptrs.push_back(param_value);  // Store all argument pointers
        
        if (arg.isPointer()) {
            void* device_ptr = *(void**)param_value;
            auto [base_ptr, info] = Interceptor::instance().findContainingAllocation(device_ptr);
            if (base_ptr && info) {
                exec.arg_sizes.push_back(info->size);
                
                // Print pre-execution info and capture state
                std::cout << "  Arg " << i << " (" << arg.getType() << "): ";
                ArgState arg_state;
                arg_state.captureGpuMemory(device_ptr, info->size);
                exec.pre_args.push_back(std::move(arg_state));
            }
        } else {
            // For non-pointer types, capture the value
            std::cout << "  Arg " << i << " (" << arg.getType() << "): ";
            
            // Get the size based on the argument type
            size_t value_size = arg.getSize();
            if (value_size == 0) {
                std::cerr << "Error: Unknown type size for argument " << i << " type: " << arg.getType() << std::endl;
                std::abort();
            }
            
            // Store the scalar value
            std::vector<char> value_data(value_size);
            std::memcpy(value_data.data(), param_value, value_size);
            
            // Create ArgState for the scalar value
            ArgState arg_state(value_size, 1);
            std::memcpy(arg_state.data.data(), value_data.data(), value_size);
            exec.pre_args.push_back(std::move(arg_state));
            
            // Print the value using the Argument's printValue method
            arg.printValue(std::cout, param_value);
            std::cout << std::endl;
        }
    }
}

void capturePostState(KernelExecution& exec, const Kernel& kernel, void** args) {
    const auto& arguments = kernel.getArguments();
    
    std::cout << "\nPOST-EXECUTION ARGUMENT VALUES:" << std::endl;
    for (size_t i = 0; i < arguments.size(); i++) {
        const auto& arg = arguments[i];
        void* param_value = args[i];
        
        if (arg.isPointer()) {
            void* device_ptr = *(void**)param_value;
            auto [base_ptr, info] = Interceptor::instance().findContainingAllocation(device_ptr);
            if (base_ptr && info) {
                std::cout << "  Arg " << i << " (" << arg.getType() << "): ";
                ArgState arg_state;
                arg_state.captureGpuMemory(device_ptr, info->size);
                exec.post_args.push_back(std::move(arg_state));
            }
        } else {
            // For scalar values, capture the final state
            size_t value_size = arg.getSize();
            ArgState arg_state(value_size, 1);
            std::memcpy(arg_state.data.data(), param_value, value_size);
            exec.post_args.push_back(std::move(arg_state));
            std::cout << "  Arg " << i << " (" << arg.getType() << "): ";
            arg.printValue(std::cout, param_value);
            std::cout << "\n";
        }
    }
}

#endif // HIP_INTERCEPT_LAYER_INTERCEPTOR_HH
