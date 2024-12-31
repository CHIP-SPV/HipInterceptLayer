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

#endif // HIP_INTERCEPT_LAYER_INTERCEPTOR_HH
