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

// Simplified function to find kernel signature
std::string getFunctionSignatureFromSource(const std::string &source,
                                           const std::string &kernel_name) {
  if (kernel_name.empty() || source.empty()) {
    std::cout << "Empty kernel name or source provided" << std::endl;
    return "";
  }

  std::cout << "Searching for kernel '" << kernel_name << "' in source code"
            << std::endl;

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
    std::sregex_iterator it(full_source.begin(), full_source.end(),
                            global_regex);
    std::sregex_iterator end;

    // Look through each match for our kernel name
    while (it != end) {
      std::string signature = it->str();
      std::cout << "Found global function: " << signature << std::endl;

      // If this signature contains our kernel name
      if (signature.find(kernel_name) != std::string::npos) {
        std::cout << "Found matching kernel signature: " << signature
                  << std::endl;
        return signature;
      }
      ++it;
    }
  } catch (const std::regex_error &e) {
    std::cout << "Regex error: " << e.what() << std::endl;
    return "";
  }

  std::cout << "Failed to find signature for kernel: " << kernel_name
            << std::endl;
  return "";
}



// Add this helper function in the anonymous namespace
void parseKernelsFromSource(const std::string &source) {

  try {
    // Add kernels from the source code
    Tracer::instance().getKernelManager().addFromModuleSource(source);

    // Log the parsing attempt
    std::cout << "Parsed kernels from source code" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Failed to parse kernels from source: " << e.what()
              << std::endl;
  }
}

std::string typeSub(const std::string &source) {
  std::string result = source;
  std::map<std::string, std::string> replacements;
  
  // Regular expressions for matching different type definitions
  std::regex typedef_regex(R"(typedef\s+([^;]+)\s+(\w+);)");
  std::regex using_regex(R"(using\s+(\w+)\s*=\s*([^;]+);)");
  std::regex define_regex(R"(#define\s+(\w+)\s+([^\n]+))");
  
  // First pass: collect all type definitions and remove the lines
  std::string::const_iterator search_start(source.cbegin());
  std::smatch match;
  
  // Find all typedefs and remove them
  while (std::regex_search(result, match, typedef_regex)) {
    replacements[match[2]] = match[1];
    // Remove the typedef line
    result.erase(match.position(), match.length());
  }
  
  // Find all using declarations and remove them
  while (std::regex_search(result, match, using_regex)) {
    replacements[match[1]] = match[2];
    // Remove the using line
    result.erase(match.position(), match.length());
  }
  
  // Find all #define macros and remove them
  while (std::regex_search(result, match, define_regex)) {
    replacements[match[1]] = match[2];
    // Remove the #define line
    result.erase(match.position(), match.length());
  }
  
  // Resolve chained typedefs to their base types
  bool changed;
  do {
    changed = false;
    for (auto &[alias, type] : replacements) {
      // For each word in the type, check if it's an alias and replace it
      for (const auto &[search_alias, replacement] : replacements) {
        std::regex word_regex("\\b" + search_alias + "\\b");
        std::string new_type = std::regex_replace(type, word_regex, replacement);
        if (new_type != type) {
          type = new_type;
          changed = true;
        }
      }
    }
  } while (changed);
  
  // Second pass: perform replacements
  for (const auto &[alias, original] : replacements) {
    std::regex word_regex("\\b" + alias + "\\b");
    result = std::regex_replace(result, word_regex, original);
  }
  
  return result;
}

std::string preprocess_source(const std::string &source, int numHeaders,
                              const char **headers, const char **headerNames) {
  if (source.empty()) {
    std::cerr << "Empty source provided" << std::endl;
    return source;
  }

  // First write all sources to temporary files
  std::string temp_dir = "/tmp/hip_intercept_XXXXXX";
  char *temp_dir_buf = strdup(temp_dir.c_str());
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
      if (headers[i] && headerNames[i]) { // Check both pointers
        std::string header_path =
            temp_dir + "/" +
            (headerNames[i] ? headerNames[i] : "header_" + std::to_string(i));
        std::ofstream header_file(header_path);
        if (!header_file) {
          std::cerr << "Failed to create header file: " << header_path
                    << std::endl;
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
    return source;
  }
  source_file << source;
  source_file.close();

  // Build g++ command
  std::string output_path = temp_dir + "/preprocessed.hip";
  std::stringstream cmd;
  cmd << "g++ -E -x c++ "; // -E for preprocessing only, -x c++ to force C++
                           // mode

  // Add include paths for headers
  for (const auto &header_path : header_paths) {
    cmd << "-I" << std::filesystem::path(header_path).parent_path() << " ";
  }

  // Input and output files
  cmd << source_path << " -o " << output_path;

  std::cout << "Preprocessing command: " << cmd.str() << std::endl;

  // Execute preprocessor
  int result = system(cmd.str().c_str());
  if (result != 0) {
    std::cerr << "Preprocessing failed with code " << result << std::endl;
    return source;
  }

  // read the preprocessed file from output_path and apply type substitution
  std::ifstream input_file(output_path);
  if (!input_file) {
    std::cerr << "Failed to read preprocessed file for type substitution" << std::endl;
    return source;
  }
  std::string typeSubbed;
  {
    std::stringstream buffer;
    buffer << input_file.rdbuf();
    typeSubbed = typeSub(buffer.str());
  }
  
  std::ofstream output_file(output_path);
  if (!output_file) {
    std::cerr << "Failed to write type-substituted file" << std::endl;
    return source;
  }
  output_file << typeSubbed;

  // Read preprocessed output
  std::ifstream preprocessed_file(output_path);
  if (!preprocessed_file) {
    std::cerr << "Failed to read preprocessed file" << std::endl;
    return source;
  }

  std::stringstream buffer;
  buffer << preprocessed_file.rdbuf();
  std::string preprocessed = buffer.str();

  std::cout << "Temporary files are stored in: " << temp_dir << std::endl;
  return preprocessed;
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
                ArgState arg_state(arg.getTypeSize(), info->size);
                arg_state.captureGpuMemory(device_ptr, info->size, arg.getTypeSize());
                exec.pre_args.push_back(std::move(arg_state));
            }
        } else {
            // For non-pointer types, capture the value
            std::cout << "  Arg " << i << " (" << arg.getType() << "): ";
            
            // Create ArgState for the scalar value
            ArgState arg_state(arg.getTypeSize(), 1);
            arg_state.captureHostMemory(param_value, arg.getTypeSize(), arg.getTypeSize());
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
                ArgState arg_state(arg.getTypeSize(), info->size);
                arg_state.captureGpuMemory(device_ptr, info->size, arg.getTypeSize());
                exec.post_args.push_back(std::move(arg_state));
            }
        } else {
            // For scalar values, capture the final state
            ArgState arg_state(arg.getTypeSize(), arg.getTypeSize());
            arg_state.captureHostMemory(param_value, arg.getTypeSize(), arg.getTypeSize());
            exec.post_args.push_back(std::move(arg_state));
            std::cout << "  Arg " << i << " (" << arg.getType() << "): ";
            arg.printValue(std::cout, param_value);
            std::cout << "\n";
        }
    }
}

#endif // HIP_INTERCEPT_LAYER_INTERCEPTOR_HH
