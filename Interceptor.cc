#include "Interceptor.hh"
#include "Tracer.hh"
#include <cxxabi.h>
#include <dlfcn.h>
#include <link.h>
#include <iostream>
#include <regex>
#include <sstream>
#include <filesystem>
#include <fstream>
#include <unistd.h>

// Function pointer types
typedef hipError_t (*hipMalloc_fn)(void**, size_t);
typedef hipError_t (*hipMemcpy_fn)(void*, const void*, size_t, hipMemcpyKind);
typedef hipError_t (*hipLaunchKernel_fn)(const void*, dim3, dim3, void**, size_t, hipStream_t);
typedef hipError_t (*hipDeviceSynchronize_fn)(void);
typedef hipError_t (*hipFree_fn)(void*);
typedef hipError_t (*hipMemset_fn)(void*, int, size_t);
typedef hipError_t (*hipModuleLaunchKernel_fn)(hipFunction_t, unsigned int,
                                              unsigned int, unsigned int,
                                              unsigned int, unsigned int,
                                              unsigned int, unsigned int,
                                              hipStream_t, void**, void**);
typedef hipError_t (*hipModuleGetFunction_fn)(hipFunction_t*, hipModule_t, const char*);

// Add RTC-related typedefs here
typedef hiprtcResult (*hiprtcCompileProgram_fn)(hiprtcProgram, int, const char**);
typedef hiprtcResult (*hiprtcCreateProgram_fn)(hiprtcProgram*, const char*, const char*, int, const char**, const char**);

// Add with other function pointer typedefs
typedef hipError_t (*hipMemcpyAsync_fn)(void*, const void*, size_t, hipMemcpyKind, hipStream_t);

// Get the real function pointers
void* getOriginalFunction(const char* name) {
    std::cout << "Looking for symbol: " << name << std::endl;
    
    // Try to find the symbol in any loaded library
    void* sym = dlsym(RTLD_NEXT, name);
    if (!sym) {
        std::cerr << "ERROR: Could not find implementation of " << name 
                  << ": " << dlerror() << std::endl;
        
        // Print currently loaded libraries for debugging
        void* handle = dlopen(NULL, RTLD_NOW);
        if (handle) {
            link_map* map;
            dlinfo(handle, RTLD_DI_LINKMAP, &map);
            std::cerr << "Loaded libraries:" << std::endl;
            while (map) {
                std::cerr << "  " << map->l_name << std::endl;
                map = map->l_next;
            }
        }
        
        std::cerr << "Make sure the real HIP runtime library is loaded." << std::endl;
        exit(1);
    }
    
    std::cout << "Found symbol " << name << " at " << sym << std::endl;
    return sym;
}

// Lazy function pointer getters
hipMalloc_fn get_real_hipMalloc() {
    static auto fn = (hipMalloc_fn)getOriginalFunction("hipMalloc");
    return fn;
}

hipMemcpy_fn get_real_hipMemcpy() {
    static auto fn = (hipMemcpy_fn)getOriginalFunction("hipMemcpy");
    return fn;
}

hipLaunchKernel_fn get_real_hipLaunchKernel() {
    static auto fn = (hipLaunchKernel_fn)getOriginalFunction("hipLaunchKernel");
    return fn;
}

hipDeviceSynchronize_fn get_real_hipDeviceSynchronize() {
    static auto fn = (hipDeviceSynchronize_fn)getOriginalFunction("hipDeviceSynchronize");
    return fn;
}

hipFree_fn get_real_hipFree() {
    static auto fn = (hipFree_fn)getOriginalFunction("hipFree");
    return fn;
}

hipMemset_fn get_real_hipMemset() {
    static auto fn = (hipMemset_fn)getOriginalFunction("hipMemset");
    return fn;
}

hipModuleLaunchKernel_fn get_real_hipModuleLaunchKernel() {
    static auto fn = (hipModuleLaunchKernel_fn)getOriginalFunction("hipModuleLaunchKernel");
    return fn;
}

hipModuleGetFunction_fn get_real_hipModuleGetFunction() {
    static auto fn = (hipModuleGetFunction_fn)getOriginalFunction("hipModuleGetFunction");
    return fn;
}

// Function pointer getters
hiprtcCreateProgram_fn get_real_hiprtcCreateProgram() {
    static auto fn = (hiprtcCreateProgram_fn)getOriginalFunction("hiprtcCreateProgram");
    return fn;
}

hiprtcCompileProgram_fn get_real_hiprtcCompileProgram() {
    static auto fn = (hiprtcCompileProgram_fn)getOriginalFunction("hiprtcCompileProgram");
    return fn;
}

// Add with other function getters
hipMemcpyAsync_fn get_real_hipMemcpyAsync() {
    static auto fn = (hipMemcpyAsync_fn)getOriginalFunction("hipMemcpyAsync");
    return fn;
}

// Helper to find which allocation a pointer belongs to
static void createShadowCopy(void* base_ptr, AllocationInfo& info) {
    // Copy current GPU memory to shadow copy
    hipError_t err = get_real_hipMemcpy()(
        info.shadow_copy.get(), 
        base_ptr,
        info.size,
        hipMemcpyDeviceToHost);
    
    if (err != hipSuccess) {
        std::cerr << "Failed to create shadow copy for allocation at " 
                  << base_ptr << " of size " << info.size << std::endl;
    } else {
        // Print first 3 values for debugging
        float* values = reinterpret_cast<float*>(info.shadow_copy.get());
        std::cout << "Shadow copy for " << base_ptr << " first 3 values: "
                  << values[0] << ", " << values[1] << ", " << values[2] 
                  << std::endl;
    }
}

static int getArgumentIndex(void* ptr, const std::vector<void*>& arg_ptrs) {
    for (size_t i = 0; i < arg_ptrs.size(); i++) {
        if (arg_ptrs[i] == ptr) return i;
    }
    return -1;
}

static hipError_t hipMemcpy_impl(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind) {
    std::cout << "\n=== INTERCEPTED hipMemcpy ===\n";
    std::cout << "hipMemcpy(dst=" << dst << ", src=" << src 
              << ", size=" << sizeBytes << ", kind=" << memcpyKindToString(kind) << ")\n";
    
    MemoryOperation op;
    op.type = MemoryOpType::COPY;
    op.dst = dst;
    op.src = src;
    op.size = sizeBytes;
    op.kind = kind;
    static uint64_t op_count = 0;
    op.execution_order = op_count++;
    
    // Initialize pre_state and post_state
    op.pre_state = std::make_shared<MemoryState>(sizeBytes);
    op.post_state = std::make_shared<MemoryState>(sizeBytes);
    
    // Capture pre-copy state if destination is GPU memory
    if (kind != hipMemcpyHostToHost) {
        auto [base_ptr, info] = Interceptor::instance().findContainingAllocation(dst);
        if (base_ptr && info) {
            createShadowCopy(base_ptr, *info);
            memcpy(op.pre_state->data.get(), info->shadow_copy.get(), sizeBytes);
        }
    }
    
    // Perform the copy
    hipError_t result = get_real_hipMemcpy()(dst, src, sizeBytes, kind);
    
    // Capture post-copy state
    if (kind != hipMemcpyHostToHost) {
        auto [base_ptr, info] = Interceptor::instance().findContainingAllocation(dst);
        if (base_ptr && info) {
            createShadowCopy(base_ptr, *info);
            memcpy(op.post_state->data.get(), info->shadow_copy.get(), sizeBytes);
        }
    }
    
    // Record operation using Tracer
    Tracer::instance().recordMemoryOperation(op);
    return result;
}

static hipError_t hipMemset_impl(void *dst, int value, size_t sizeBytes) {
    std::cout << "hipMemset(dst=" << dst << ", value=" << value 
              << ", size=" << sizeBytes << ")\n";
    
    MemoryOperation op;
    op.type = MemoryOpType::SET;
    op.dst = dst;
    op.size = sizeBytes;
    op.value = value;
    static uint64_t op_count = 0;
    op.execution_order = op_count++;
    
    // Initialize states
    op.pre_state = std::make_shared<MemoryState>(sizeBytes);
    op.post_state = std::make_shared<MemoryState>(sizeBytes);
    
    // Capture pre-set state
    auto [base_ptr, info] = Interceptor::instance().findContainingAllocation(dst);
    if (base_ptr && info) {
        createShadowCopy(base_ptr, *info);
        memcpy(op.pre_state->data.get(), info->shadow_copy.get(), sizeBytes);
    }
    
    // Perform the memset
    hipError_t result = get_real_hipMemset()(dst, value, sizeBytes);
    
    // Capture post-set state
    if (base_ptr && info) {
        createShadowCopy(base_ptr, *info);
        memcpy(op.post_state->data.get(), info->shadow_copy.get(), sizeBytes);
    }
    
    // Record operation using Tracer
    Tracer::instance().recordMemoryOperation(op);
    return result;
}

static hipError_t hipLaunchKernel_impl(const void *function_address, dim3 numBlocks,
                                     dim3 dimBlocks, void **args, size_t sharedMemBytes,
                                     hipStream_t stream) {
    std::cout << "\nDEBUG: hipLaunchKernel_impl for function " << function_address << std::endl;
              
    // Get kernel name and print args using Tracer
    auto kernel = Tracer::instance().getKernelManager().getKernelByPointer(function_address);
    std::cout << kernel.getName() << " " << kernel.getSignature() << std::endl;
    
    
    // Create execution record
    KernelExecution exec;
    exec.function_address = (void*)function_address;
    exec.kernel_name = kernel.getName();
    exec.grid_dim = numBlocks;
    exec.block_dim = dimBlocks;
    exec.shared_mem = sharedMemBytes;
    exec.stream = stream;
    static uint64_t kernel_count = 0;
    exec.execution_order = kernel_count++;

    std::cout << "\nDEBUG: Processing kernel arguments:"
              << "\n  Total args: " << kernel.getArguments().size() << std::endl;

    const auto& arguments = kernel.getArguments();
    for (size_t i = 0; i < arguments.size(); i++) {
        const auto& arg = arguments[i];
        void* param_value = args[i];
        
        std::cout << "DEBUG: Processing arg " << i 
                  << "\n  Type: " << arg.getType()
                  << "\n  Is pointer: " << arg.isPointer() << std::endl;
        
        if (arg.isPointer() && !arg.isVector()) {
            void* device_ptr = *(void**)param_value;
            exec.arg_ptrs.push_back(device_ptr);
            
            std::cout << "DEBUG: Found pointer argument:"
                      << "\n  Device ptr: " << device_ptr << std::endl;
            
            auto [base_ptr, info] = Interceptor::instance().findContainingAllocation(device_ptr);
            if (base_ptr && info) {
                std::cout << "DEBUG: Creating shadow copy for arg " << i
                          << "\n  Base ptr: " << base_ptr
                          << "\n  Size: " << info->size << std::endl;
                          
                createShadowCopy(base_ptr, *info);
                exec.pre_state = std::make_shared<MemoryState>(info->shadow_copy.get(), info->size);
            } else {
                std::cerr << "WARNING: Could not find allocation for arg " << i
                          << " ptr: " << device_ptr << std::endl;
            }
        } else {
            // For scalar arguments, create a memory state with the value
            exec.arg_ptrs.push_back(param_value);
            size_t arg_size = arg.getSize();
            exec.pre_state = std::make_shared<MemoryState>(reinterpret_cast<const char*>(param_value), arg_size);
            exec.arg_sizes.push_back(arg_size);
        }
    }

    // Launch kernel
    hipError_t result = get_real_hipLaunchKernel()(function_address, numBlocks, 
                                                  dimBlocks, args, sharedMemBytes, stream);
    (void)get_real_hipDeviceSynchronize()();
    
    // Capture post-execution state
    std::cout << "DEBUG: Capturing post-execution state..." << std::endl;
    
    for (size_t i = 0; i < exec.arg_ptrs.size(); i++) {
        void* device_ptr = exec.arg_ptrs[i];
        auto [base_ptr, info] = Interceptor::instance().findContainingAllocation(device_ptr);
        if (base_ptr && info) {
            std::cout << "DEBUG: Creating post-exec shadow copy for arg " << i
                      << "\n  Base ptr: " << base_ptr
                      << "\n  Size: " << info->size << std::endl;
                      
            createShadowCopy(base_ptr, *info);
            exec.post_state = std::make_shared<MemoryState>(info->shadow_copy.get(), info->size);
        } else {
            std::cerr << "WARNING: Could not find allocation for post-state arg " << i
                      << " ptr: " << device_ptr << std::endl;
        }
    }
    
    // Record kernel execution using Tracer
    Tracer::instance().recordKernelLaunch(exec);
    
    return result;
}

// Simplified function to find kernel signature
std::string getFunctionSignatureFromSource(const std::string& source, const std::string& kernel_name) {
    if (kernel_name.empty() || source.empty()) {
        std::cout << "Empty kernel name or source provided" << std::endl;
        return "";
    }

    std::cout << "Searching for kernel '" << kernel_name << "' in source code" << std::endl;
    
    // Read entire source into a single string, preserving newlines
    std::istringstream stream(source);
    std::string full_source;
    std::string line;
    while (std::getline(stream, line)) {
        full_source += line + "\n";
    }

    try {
        // Find all __global__ function declarations
        std::regex global_regex(R"(__global__[^(]*\([^)]*\))");
        std::sregex_iterator it(full_source.begin(), full_source.end(), global_regex);
        std::sregex_iterator end;

        // Look through each match for our kernel name
        while (it != end) {
            std::string signature = it->str();
            std::cout << "Found global function: " << signature << std::endl;
            
            // If this signature contains our kernel name
            if (signature.find(kernel_name) != std::string::npos) {
                std::cout << "Found matching kernel signature: " << signature << std::endl;
                return signature;
            }
            ++it;
        }
    } catch (const std::regex_error& e) {
        std::cout << "Regex error: " << e.what() << std::endl;
        return "";
    }
    
    std::cout << "Failed to find signature for kernel: " << kernel_name << std::endl;
    return "";
}



// Add this helper function in the anonymous namespace
    void parseKernelsFromSource(const std::string& source) {

        
        try {
            // Add kernels from the source code
            Tracer::instance().getKernelManager().addFromModuleSource(source);
            
            // Log the parsing attempt
            std::cout << "Parsed kernels from source code" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to parse kernels from source: " << e.what() << std::endl;
        }
    }

std::string preprocess_source(const std::string& source,
                            int numHeaders, const char** headers, const char** headerNames) {
    if (source.empty()) {
        std::cerr << "Empty source provided" << std::endl;
        return source;
    }

    // First write all sources to temporary files
    std::string temp_dir = "/tmp/hip_intercept_XXXXXX";
    char* temp_dir_buf = strdup(temp_dir.c_str());
    if (!mkdtemp(temp_dir_buf)) {
        std::cerr << "Failed to create temp directory" << std::endl;
        free(temp_dir_buf);
        return source;
    }
    temp_dir = temp_dir_buf;
    free(temp_dir_buf);

    // Write headers
    std::vector<std::string> header_paths;
    if (numHeaders > 0 && headers && headerNames) {
        for (int i = 0; i < numHeaders; i++) {
            if (headers[i] && headerNames[i]) {  // Check both pointers
                std::string header_path = temp_dir + "/" + 
                    (headerNames[i] ? headerNames[i] : "header_" + std::to_string(i));
                std::ofstream header_file(header_path);
                if (!header_file) {
                    std::cerr << "Failed to create header file: " << header_path << std::endl;
                    continue;
                }
                header_file << headers[i];
                header_paths.push_back(header_path);
            }
        }
    }

    // Write main source
    std::string source_path = temp_dir + "/source.hip";
    std::ofstream source_file(source_path);
    if (!source_file) {
        std::cerr << "Failed to create source file" << std::endl;
        std::filesystem::remove_all(temp_dir);  // Clean up before returning
        return source;
    }
    source_file << source;
    source_file.close();

    // Build g++ command
    std::string output_path = temp_dir + "/preprocessed.hip";
    std::stringstream cmd;
    cmd << "g++ -E -x c++ "; // -E for preprocessing only, -x c++ to force C++ mode
    
    // Add include paths for headers
    for (const auto& header_path : header_paths) {
        cmd << "-I" << std::filesystem::path(header_path).parent_path() << " ";
    }
    
    // Input and output files
    cmd << source_path << " -o " << output_path;

    std::cout << "Preprocessing command: " << cmd.str() << std::endl;

    // Execute preprocessor
    int result = system(cmd.str().c_str());
    if (result != 0) {
        std::cerr << "Preprocessing failed with code " << result << std::endl;
        std::filesystem::remove_all(temp_dir);  // Clean up before returning
        return source;
    }

    // Read preprocessed output
    std::ifstream preprocessed_file(output_path);
    if (!preprocessed_file) {
        std::cerr << "Failed to read preprocessed file" << std::endl;
        std::filesystem::remove_all(temp_dir);  // Clean up before returning
        return source;
    }

    std::stringstream buffer;
    buffer << preprocessed_file.rdbuf();
    std::string preprocessed = buffer.str();

    // Clean up temporary files
    std::filesystem::remove_all(temp_dir);

    return preprocessed;
}

// Update hiprtcCreateProgram implementation
hiprtcResult hiprtcCreateProgram(hiprtcProgram* prog,
                                const char* src,
                                const char* name,
                                int numHeaders,
                                const char** headers,
                                const char** headerNames) {
    std::cout << "\n=== INTERCEPTED hiprtcCreateProgram ===\n";
    
    hiprtcResult result = get_real_hiprtcCreateProgram()(prog, src, name, 
                                                        numHeaders, headers, headerNames);
    
    if (result == HIPRTC_SUCCESS && prog && src) {
        // Store source code using KernelManager
        Tracer::instance().getKernelManager().addRTCProgram(*prog, src);
        std::cout << "Stored RTC program source for handle " << *prog << std::endl;

        auto preprocessed_src = preprocess_source(src, numHeaders, headers, headerNames);
        // Parse kernels from the source code immediately
        parseKernelsFromSource(preprocessed_src);
    }
    
    return result;
}


extern "C" {

hipError_t hipMalloc(void **ptr, size_t size) {
    std::cout << "hipMalloc(ptr=" << (void*)ptr << ", size=" << size << ")\n";
    
    // Create memory operation record
    MemoryOperation op;
    op.type = MemoryOpType::ALLOC;
    op.dst = nullptr;  // Will be filled after allocation
    op.src = nullptr;
    op.size = size;
    op.kind = hipMemcpyHostToHost;
    static uint64_t op_count = 0;
    op.execution_order = op_count++;
    
    hipError_t result = get_real_hipMalloc()(ptr, size);
    
    if (result == hipSuccess && ptr && *ptr) {
        op.dst = *ptr;
        
        // First create the allocation info
        auto& info = Interceptor::instance().addAllocation(*ptr, size);
        
        // Create and capture initial state
        op.pre_state = std::make_shared<MemoryState>(size);
        op.post_state = std::make_shared<MemoryState>(size);
        
        // Capture initial state of allocated memory
        createShadowCopy(*ptr, info);
        memcpy(op.pre_state->data.get(), info.shadow_copy.get(), size);
        memcpy(op.post_state->data.get(), info.shadow_copy.get(), size);
        
        std::cout << "Tracking GPU allocation at " << *ptr 
                  << " of size " << size << std::endl;
        Tracer::instance().recordMemoryOperation(op);
    }
    
    return result;
}

hipError_t hipLaunchKernel(const void *function_address, dim3 numBlocks,
                          dim3 dimBlocks, void **args, size_t sharedMemBytes,
                          hipStream_t stream) {
    std::cout << "\n=== INTERCEPTED hipLaunchKernel ===\n";
    std::cout << "hipLaunchKernel(function_address=" << function_address 
              << ", numBlocks={" << numBlocks.x << "," << numBlocks.y << "," << numBlocks.z << "}"
              << ", dimBlocks={" << dimBlocks.x << "," << dimBlocks.y << "," << dimBlocks.z << "}"
              << ", sharedMem=" << sharedMemBytes
              << ", stream=" << stream << ")\n";
    return hipLaunchKernel_impl(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream);
}

hipError_t hipDeviceSynchronize(void) {
    return get_real_hipDeviceSynchronize()();
}

hipError_t hipFree(void* ptr) {
    if (ptr)
        Interceptor::instance().removeAllocation(ptr);
    return get_real_hipFree()(ptr);
}

__attribute__((visibility("default")))
hipError_t hipMemcpy(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind) {
    return hipMemcpy_impl(dst, src, sizeBytes, kind);
}

__attribute__((visibility("default")))
hipError_t hipMemset(void *dst, int value, size_t sizeBytes) {
    return hipMemset_impl(dst, value, sizeBytes);
}

// Update hipModuleLaunchKernel to use KernelManager
hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX,
                                 unsigned int gridDimY, unsigned int gridDimZ,
                                 unsigned int blockDimX, unsigned int blockDimY,
                                 unsigned int blockDimZ, unsigned int sharedMemBytes,
                                 hipStream_t stream, void** kernelParams,
                                 void** extra) {
    std::cout << "\n=== INTERCEPTED hipModuleLaunchKernel ===\n";
    std::cout << "hipModuleLaunchKernel(\n"
              << "    function=" << f
              << "\n    gridDim={" << gridDimX << "," << gridDimY << "," << gridDimZ << "}"
              << "\n    blockDim={" << blockDimX << "," << blockDimY << "," << blockDimZ << "}"
              << "\n    sharedMem=" << sharedMemBytes
              << "\n    stream=" << stream << "\n";
 
    // Get kernel info from KernelManager
    std::string kernel_name = Tracer::instance().getKernelManager().getRTCKernelName(f);
    if (kernel_name.empty()) {
        std::cout << "No kernel name found for function handle " << f << std::endl;
        std::abort();
    }
 
    std::cout << "Looking up kernel: '" << kernel_name << "'" << std::endl;
    Kernel kernel = Tracer::instance().getKernelManager().getKernelByName(kernel_name);

    // Create execution record
    KernelExecution exec;
    exec.function_address = f;
    exec.kernel_name = kernel.getName();
    exec.grid_dim = {gridDimX, gridDimY, gridDimZ};
    exec.block_dim = {blockDimX, blockDimY, blockDimZ};
    exec.shared_mem = sharedMemBytes;
    exec.stream = stream;
    static uint64_t kernel_count = 0;
    exec.execution_order = kernel_count++;

    std::cout << "\nDEBUG: Processing kernel arguments:"
              << "\n  Total args: " << kernel.getArguments().size() << std::endl;

    // Modified parameter handling
    const auto& arguments = kernel.getArguments();
    for (size_t i = 0; i < arguments.size(); i++) {
        const auto& arg = arguments[i];
        void* param_value = kernelParams[i];
        
        std::cout << "DEBUG: Processing arg " << i 
                  << "\n  Type: " << arg.getType()
                  << "\n  Is pointer: " << arg.isPointer() << std::endl;
        
        // For pointer types, we need to dereference kernelParams[i] to get the actual pointer
        if (arg.getType().find("*") != std::string::npos && !arg.isVector()) {
            void* device_ptr = *(void**)param_value;
            exec.arg_ptrs.push_back(device_ptr);
            
            std::cout << "DEBUG: Found pointer argument:"
                      << "\n  Device ptr: " << device_ptr << std::endl;
            
            // Try to capture pre-execution state
            auto [base_ptr, info] = Interceptor::instance().findContainingAllocation(device_ptr);
            if (base_ptr && info) {
                std::cout << "DEBUG: Creating shadow copy for arg " << i
                          << "\n  Base ptr: " << base_ptr
                          << "\n  Size: " << info->size << std::endl;
                          
                createShadowCopy(base_ptr, *info);
                exec.pre_state = std::make_shared<MemoryState>(info->shadow_copy.get(), info->size);
            } else {
                std::cerr << "WARNING: Could not find allocation for arg " << i
                          << " ptr: " << device_ptr << std::endl;
            }
        } else {
            // For scalar arguments, create a memory state with the value
            exec.arg_ptrs.push_back(param_value);
            size_t arg_size = arg.getSize();
            exec.pre_state = std::make_shared<MemoryState>(reinterpret_cast<const char*>(param_value), arg_size);
            exec.arg_sizes.push_back(arg_size);
        }
    }

    // Launch kernel
    hipError_t result = get_real_hipModuleLaunchKernel()(f, gridDimX, gridDimY, gridDimZ,
                                                        blockDimX, blockDimY, blockDimZ,
                                                        sharedMemBytes, stream,
                                                        kernelParams, extra);
    (void)get_real_hipDeviceSynchronize()();
    
    // Capture post-execution state
    std::cout << "DEBUG: Capturing post-execution state..." << std::endl;
    
    for (size_t i = 0; i < arguments.size(); i++) {
        const auto& arg = arguments[i];
        if (arg.getType().find("*") != std::string::npos && !arg.isVector()) {
            void* device_ptr = *(void**)kernelParams[i];
            auto [base_ptr, info] = Interceptor::instance().findContainingAllocation(device_ptr);
            if (base_ptr && info) {
                std::cout << "DEBUG: Creating post-exec shadow copy for arg " << i
                          << "\n  Base ptr: " << base_ptr
                          << "\n  Size: " << info->size << std::endl;
                          
                createShadowCopy(base_ptr, *info);
                exec.post_state = std::make_shared<MemoryState>(info->shadow_copy.get(), info->size);
            } else {
                std::cerr << "WARNING: Could not find allocation for post-state arg " << i
                          << " ptr: " << device_ptr << std::endl;
            }
        }
    }

    Tracer::instance().recordKernelLaunch(exec);
    return result;
}

// Update hipModuleGetFunction to use KernelManager
hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* kname) {
    std::cout << "\n=== INTERCEPTED hipModuleGetFunction ===\n";
    std::cout << "hipModuleGetFunction(function=" << function 
              << ", module=" << module 
              << ", kname=" << kname << ")\n";
              
    hipError_t result = get_real_hipModuleGetFunction()(function, module, kname);
    
    if (result == hipSuccess && function && *function) {
        // Store the kernel name using KernelManager
        Tracer::instance().getKernelManager().addRTCKernel(*function, kname);
        std::cout << "Stored RTC kernel name '" << kname 
                  << "' for function handle " << *function << std::endl;
    }
    
    return result;
}

hiprtcResult hiprtcCompileProgram(hiprtcProgram prog,
                                 int numOptions,
                                 const char** options) {
    std::cout << "\n=== INTERCEPTED hiprtcCompileProgram ===\n";
    return get_real_hiprtcCompileProgram()(prog, numOptions, options);
}

// Add implementation
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, 
                         hipMemcpyKind kind, hipStream_t stream) {
    static uint64_t op_count = 0;
    std::cout << "\n=== INTERCEPTED hipMemcpyAsync #" << op_count << " ===\n";
    std::cout << "hipMemcpyAsync(dst=" << dst << ", src=" << src 
              << ", size=" << sizeBytes << ", kind=" << memcpyKindToString(kind)
              << ", stream=" << stream << ")\n";
    
    MemoryOperation op;
    op.type = MemoryOpType::COPY_ASYNC;
    op.dst = dst;
    op.src = src;
    op.size = sizeBytes;
    op.kind = kind;
    op.stream = stream;
    op.execution_order = op_count++;

    // For device-to-host or device-to-device copies, capture pre-state
    if (kind == hipMemcpyDeviceToHost || kind == hipMemcpyDeviceToDevice) {
        auto [base_ptr, info] = Interceptor::instance().findContainingAllocation(const_cast<void*>(src));
        if (base_ptr && info) {
            createShadowCopy(base_ptr, *info);
            op.pre_state = std::make_shared<MemoryState>(
                info->shadow_copy.get(), info->size);
        }
    }

    // Execute the actual memory copy
    hipError_t result = get_real_hipMemcpyAsync()(dst, src, sizeBytes, kind, stream);

    // Synchronize to ensure the copy is complete before capturing state
    (void)get_real_hipDeviceSynchronize()();

    // For host-to-device or device-to-device copies, capture post-state
    if (kind == hipMemcpyHostToDevice || kind == hipMemcpyDeviceToDevice) {
        auto [base_ptr, info] = Interceptor::instance().findContainingAllocation(dst);
        if (base_ptr && info) {
            createShadowCopy(base_ptr, *info);
            op.post_state = std::make_shared<MemoryState>(
                info->shadow_copy.get(), info->size);
        }
    }

    // Record the operation
    Tracer::instance().recordMemoryOperation(op);
    
    return result;
}

static bool g_should_intercept = false;

// Called when the library is loaded
__attribute__((constructor))
void hip_intercept_init() {
    // Get the executable name
    char exe_path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path)-1);
    if (len != -1) {
        exe_path[len] = '\0';
        std::string exe_name = std::filesystem::path(exe_path).filename();
        auto abs_path = std::filesystem::absolute(exe_path);
        //std::cout << "hip_intercept_init() called for process: " << abs_path << std::endl;

        
        // Skip interception for known compiler/toolchain executables
        const std::vector<std::string> skip_list = {
            "cc1plus",
            "g++",
            "gcc",
            "ld",
            "as",
            "collect2",
            "x86_64-linux-gnu-g++",
            "x86_64-linux-gnu-gcc",
            "gdb",
            "grep",
            "nm",
            "objdump",
            "readelf",
            "strip"
            };
            
        g_should_intercept = std::none_of(skip_list.begin(), skip_list.end(),
            [&exe_name](const std::string& skip) {
                return exe_name.find(skip) != std::string::npos;
            });
        
        if (g_should_intercept) {
            Interceptor::instance().setExePath(abs_path);
        }
    }
}

} // extern "C"
