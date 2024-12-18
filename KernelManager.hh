#ifndef HIP_INTERCEPT_LAYER_KERNEL_MANAGER_HH
#define HIP_INTERCEPT_LAYER_KERNEL_MANAGER_HH

#include <string>
#include <vector>
#include <iostream>
#include <regex>
#include <stack>
#include <fstream>
#include <unordered_map>
#include <queue>
#include <link.h>
#include <elf.h>
#include <sstream>
#include <iomanip>
#define __HIP_PLATFORM_SPIRV__
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

const uint32_t KRNL_MAGIC = 0x4B524E4C;
const uint32_t KRNL_VERSION = 1;

inline std::string demangle(const std::string& mangled_name) {
    int status;
    char* demangled = abi::__cxa_demangle(mangled_name.c_str(), nullptr, nullptr, &status);
    
    std::string result;
    if (status == 0 && demangled) {
        result = demangled;
        free(demangled);
    } else {
        result = mangled_name;  // Return original if demangling fails
    }
    
    std::cout << "Demangled " << mangled_name << " to " << result << std::endl;
    return result;
}

class Argument {
public:
    std::string name;
    std::string type;
    size_t size;
    bool is_pointer;
    bool is_vector;
    void operator<<(std::ostream& os) const {
        os << "Argument: " << type << " " << name << " (size: " << size << " is_pointer: " << is_pointer << " is_vector: " << is_vector << ")\n";
    }
    Argument(std::string name, std::string type) {
        if (name.empty() || type.empty()) {
            //std::cerr << "Warning: Creating argument with empty name or type" << std::endl;
        }
        this->name = name;
        this->type = type;
        this->size = isVector() ? 16 : sizeof(void*);
        this->is_pointer = type.find("*") != std::string::npos;
        this->is_vector = isVector();
        std::cout << "    Argument: " << this->type << " " << this->name 
                  << " (size: " << this->size << ")" << std::endl;
    }

    bool isPointer() const {
        return is_pointer;
    }

    size_t getSize() const {
        return size;
    }

    std::string getName() const {
        return name;
    }

    bool isVector() const {
        static const std::vector<std::string> vector_types = {
            "float4", "float3", "float2",
            "int4", "int3", "int2",
            "uint4", "uint3", "uint2",
            "double4", "double3", "double2",
            "long4", "long3", "long2",
            "ulong4", "ulong3", "ulong2",
            "char4", "char3", "char2",
            "uchar4", "uchar3", "uchar2",
            "HIP_vector_type"
        };
        
        for (const auto& vtype : vector_types) {
            if (type.find(vtype) != std::string::npos) {
                return true;
            }
        }
        return false;
    }

    std::string getType() const {
        return type;
    }

    std::string getBaseType() const {
        // Remove __restrict__ and similar qualifiers
        std::string base_type = type;
        std::regex qualifiers(R"(\s*(?:__restrict__|__restrict|__global__|__device__|__host__|__constant__|__shared__|__managed__|__pinned__|__constant__|__shared__|__managed__|__pinned__))");
        base_type = std::regex_replace(base_type, qualifiers, "");
        // Remove any trailing spaces
        base_type = std::regex_replace(base_type, std::regex("\\s+$"), "");
        // Remove any leading spaces
        base_type = std::regex_replace(base_type, std::regex("^\\s+"), "");

        // Also strip *
        base_type = std::regex_replace(base_type, std::regex("\\*"), "");
        return base_type;
    }

    void serialize(std::ofstream& file) const {
        //std::cout << "Serializing argument: " << name << " (" << type << ")" << std::endl;
        // Write the argument name
        uint32_t name_length = name.length();
        file.write(reinterpret_cast<const char*>(&name_length), sizeof(uint32_t));
        file.write(name.c_str(), name_length);
        
        // Write the type
        uint32_t type_length = type.length();
        file.write(reinterpret_cast<const char*>(&type_length), sizeof(uint32_t));
        file.write(type.c_str(), type_length);
        
        // Write other properties
        file.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(&is_pointer), sizeof(bool));
        file.write(reinterpret_cast<const char*>(&is_vector), sizeof(bool));
    }

    void deserialize(std::ifstream& file) {
        // Read the argument name
        uint32_t name_length;
        file.read(reinterpret_cast<char*>(&name_length), sizeof(uint32_t));
        name.resize(name_length);
        file.read(&name[0], name_length);
        
        // Read the type
        uint32_t type_length;
        file.read(reinterpret_cast<char*>(&type_length), sizeof(uint32_t));
        type.resize(type_length);
        file.read(&type[0], type_length);
        
        // Read other properties
        file.read(reinterpret_cast<char*>(&size), sizeof(size_t));
        file.read(reinterpret_cast<char*>(&is_pointer), sizeof(bool));
        file.read(reinterpret_cast<char*>(&is_vector), sizeof(bool));
    }
};

class Kernel {
    std::string kernelSource;
    std::string moduleSource;
    std::string name;
    std::string signature;
    std::vector<Argument> arguments;
    void* host_address;
    void* device_address;

    void parseKernelSource() {
        if (moduleSource.empty() || name.empty()) {
            return;
        }

        // Find the kernel signature in the module source
        // First try to find the exact signature
        size_t kernelStart = moduleSource.find(signature);
        
        // If not found, try finding it with __global__ void prefix
        if (kernelStart == std::string::npos) {
            std::string fullSignature = "__global__ void " + name;
            kernelStart = moduleSource.find(fullSignature);
            if (kernelStart == std::string::npos) {
                std::cout << "Could not find kernel signature in module source" << std::endl;
                return;
            }
        }

        std::cout << "Found kernel at position: " << kernelStart << std::endl;
        std::cout << "Module source snippet: " << moduleSource.substr(kernelStart, 100) << "..." << std::endl;

        // Find the opening brace after the signature
        size_t braceStart = moduleSource.find('{', kernelStart);
        if (braceStart == std::string::npos) {
            std::cout << "Could not find kernel opening brace" << std::endl;
            return;
        }

        // Use a stack-based approach to find matching braces
        std::stack<size_t> braceStack;
        size_t braceEnd = std::string::npos;
        bool inString = false;
        char stringDelimiter = 0;

        // Push the first opening brace onto the stack
        braceStack.push(braceStart);

        for (size_t pos = braceStart + 1; pos < moduleSource.length(); ++pos) {
            char c = moduleSource[pos];

            // Handle string literals to avoid counting braces inside strings
            if ((c == '"' || c == '\'') && (pos == 0 || moduleSource[pos-1] != '\\')) {
                if (!inString) {
                    inString = true;
                    stringDelimiter = c;
                } else if (c == stringDelimiter) {
                    inString = false;
                }
                continue;
            }

            // Skip characters while in a string
            if (inString) continue;

            // Handle braces
            if (c == '{') {
                braceStack.push(pos);
            } else if (c == '}') {
                if (braceStack.empty()) {
                    std::cout << "Mismatched braces found" << std::endl;
                    return;
                }
                braceStack.pop();
                if (braceStack.empty()) {
                    braceEnd = pos;
                    break;
                }
            }
        }

        if (braceEnd == std::string::npos) {
            std::cout << "Could not find matching closing brace" << std::endl;
            return;
        }

        // Extract the complete kernel source including signature and body
        kernelSource = moduleSource.substr(kernelStart, braceEnd - kernelStart + 1);
        //std::cout << "Kernel source:\n" << kernelSource << std::endl;
    }
public:

    void operator<<(std::ostream& os) const {
        os << "Kernel: " << name << " (" << signature << ")" << std::endl;
        os << "    Host address: " << host_address << std::endl;
        os << "    Device address: " << device_address << std::endl;
        for (const auto& arg : arguments) {
            arg << os;
        }
    }
    void* getHostAddress() const {
        return host_address;
    }

    void setHostAddress(void* addr) {
        std::cout << "Setting host address to: " << std::hex << addr << std::dec << std::endl;
        host_address = addr;
    }

    void* getDeviceAddress() const {
        return device_address;
    }

    void setDeviceAddress(void* addr) {
        std::cout << "Setting device address to: " << std::hex << addr << std::dec << std::endl;
        device_address = addr;
    }

    std::vector<Argument>getArguments() const {
        return arguments;
    }

    std::string getName() const {
        return name;
    }

    std::string getSignature() const {
        return signature;
    }

    std::string getSource() const {
        return kernelSource;
    }

    std::string getModuleSource() const {
        return moduleSource;
    }

    void setModuleSource(const std::string& module_source) {
        this->moduleSource = module_source;
        parseKernelSource();
    }
    Kernel() {}

    std::pair<std::string, std::vector<Argument>> getKernelInfo(std::string signature) const {
        std::vector<Argument> args;
        std::string kernel_name;
        // Extract kernel name from device stub
        size_t stub_pos = signature.find("__device_stub__");
        if (stub_pos != std::string::npos) {
            // Remove __device_stub__ prefix to get actual kernel name
            signature = signature.substr(stub_pos + 15); // 15 is length of "__device_stub__"
        }

        // Extract kernel name from source signature
        size_t void_pos = signature.find("void");
        if (void_pos != std::string::npos) {
            signature = signature.substr(void_pos + 4 + 1); // void +1 for space
        }
        
        // Extract kernel name (everything up to the opening parenthesis)
        size_t name_end = signature.find('(');
        if (name_end != std::string::npos) {
            kernel_name = signature.substr(0, name_end);
            // Trim whitespace
            kernel_name.erase(0, kernel_name.find_first_not_of(" "));
            kernel_name.erase(kernel_name.find_last_not_of(" ") + 1);
            std::cout << "Kernel name: " << kernel_name << std::endl;
        }

        // Extract arguments string (everything between parentheses)
        size_t args_start = signature.find('(');
        size_t args_end = signature.find_last_of(')');
        if (args_start != std::string::npos && args_end != std::string::npos) {
            std::string args_str = signature.substr(args_start + 1, args_end - args_start - 1);
            
            // Split arguments by comma
            size_t pos = 0;
            size_t next_pos;
            while ((next_pos = args_str.find(',', pos)) != std::string::npos) {
                std::string arg = args_str.substr(pos, next_pos - pos);
                
                // Trim whitespace
                arg.erase(0, arg.find_first_not_of(" "));
                arg.erase(arg.find_last_not_of(" ") + 1);
                
                // Handle types with * (pointers)
                size_t asterisk_pos = arg.find('*');
                if (asterisk_pos != std::string::npos) {
                    std::string type = arg.substr(0, asterisk_pos + 1);
                    std::string name = arg.substr(asterisk_pos + 1);
                    
                    // Trim whitespace
                    type.erase(0, type.find_first_not_of(" "));
                    type.erase(type.find_last_not_of(" ") + 1);
                    name.erase(0, name.find_first_not_of(" "));
                    name.erase(name.find_last_not_of(" ") + 1);
                    
                    if (name.empty()) {
                        name = "arg" + std::to_string(args.size() + 1);
                    }
                    args.emplace_back(name, type);
                } else {
                    // Handle non-pointer types
                    size_t last_space = arg.find_last_of(" ");
                    std::string type, name;
                    
                    if (last_space != std::string::npos) {
                        type = arg.substr(0, last_space);
                        name = arg.substr(last_space + 1);
                    } else {
                        type = arg;
                        name = "arg" + std::to_string(args.size() + 1);
                    }
                    
                    args.emplace_back(name, type);
                }
                pos = next_pos + 1;
            }
            
            // Handle last argument
            std::string arg = args_str.substr(pos);
            arg.erase(0, arg.find_first_not_of(" "));
            arg.erase(arg.find_last_not_of(" ") + 1);
            
            // Handle types with * (pointers)
            size_t asterisk_pos = arg.find('*');
            if (asterisk_pos != std::string::npos) {
                std::string type = arg.substr(0, asterisk_pos + 1);
                std::string name = arg.substr(asterisk_pos + 1);
                
                // Trim whitespace
                type.erase(0, type.find_first_not_of(" "));
                type.erase(type.find_last_not_of(" ") + 1);
                name.erase(0, name.find_first_not_of(" "));
                name.erase(name.find_last_not_of(" ") + 1);
                
                if (name.empty()) {
                    name = "arg" + std::to_string(args.size() + 1);
                }
                args.emplace_back(name, type);
            } else {
                // Handle non-pointer types
                size_t last_space = arg.find_last_of(" ");
                std::string type, name;
                
                if (last_space != std::string::npos) {
                    type = arg.substr(0, last_space);
                    name = arg.substr(last_space + 1);
                } else {
                    type = arg;
                    name = "arg" + std::to_string(args.size() + 1);
                }
                
                args.emplace_back(name, type);
            }
        }

        
        return {kernel_name, args};
    }

    Kernel(std::string signature) {
        std::cout << "\nParsing kernel signature: " << signature << std::endl;
        
        // Trim whitespace from signature and normalize newlines
        this->signature = signature;
        
        // Replace newlines and multiple spaces with single space
        std::regex whitespace_regex(R"(\s+)");
        this->signature = std::regex_replace(this->signature, whitespace_regex, " ");
        
        // Trim leading/trailing whitespace
        this->signature.erase(0, this->signature.find_first_not_of(" "));
        this->signature.erase(this->signature.find_last_not_of(" ") + 1);

        std::tie(name, arguments) = getKernelInfo(signature);
    }

    void serialize(std::ofstream& file) const {
        // Write string lengths
        uint32_t kernel_source_len = kernelSource.length();
        uint32_t module_source_len = moduleSource.length();
        uint32_t name_len = name.length();
        uint32_t signature_len = signature.length();
        
        //std::cout << "Writing kernel '" << name << "' with lengths:"
                  //<< "\n  kernel_source: " << kernel_source_len
                  //<< "\n  module_source: " << module_source_len
                  //<< "\n  name: " << name_len
                  //<< "\n  signature: " << signature_len << std::endl;
        
        file.write(reinterpret_cast<const char*>(&kernel_source_len), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&module_source_len), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&name_len), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&signature_len), sizeof(uint32_t));
        
        // Write string data
        file.write(kernelSource.c_str(), kernel_source_len);
        file.write(moduleSource.c_str(), module_source_len);
        file.write(name.c_str(), name_len);
        file.write(signature.c_str(), signature_len);
        
        // Write arguments
        uint32_t num_args = arguments.size();
        file.write(reinterpret_cast<const char*>(&num_args), sizeof(uint32_t));
        
        //std::cout << "Writing " << num_args << " arguments" << std::endl;
        for (const auto& arg : arguments) {
            arg.serialize(file);
        }
    }

    static Kernel deserialize(std::ifstream& file) {
        Kernel kernel;
        
        // Read string lengths
        uint32_t kernel_source_len, module_source_len, name_len, signature_len;
        file.read(reinterpret_cast<char*>(&kernel_source_len), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&module_source_len), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&name_len), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&signature_len), sizeof(uint32_t));
        
        //std::cout << "Reading kernel with lengths:"
                  //<< "\n  kernel_source: " << kernel_source_len
                  //<< "\n  module_source: " << module_source_len
                  //<< "\n  name: " << name_len
                  //<< "\n  signature: " << signature_len << std::endl;
                  
        // Sanity check lengths
        const uint32_t MAX_STRING_LENGTH = 1000000;  // 1MB limit
        if (kernel_source_len > MAX_STRING_LENGTH || 
            module_source_len > MAX_STRING_LENGTH ||
            name_len > MAX_STRING_LENGTH || 
            signature_len > MAX_STRING_LENGTH) {
            std::cerr << "Error: Invalid string length detected in kernel deserialization" << std::endl;
            throw std::runtime_error("Invalid string length in kernel deserialization");
        }
        
        // Read strings
        kernel.kernelSource.resize(kernel_source_len);
        kernel.moduleSource.resize(module_source_len);
        kernel.name.resize(name_len);
        kernel.signature.resize(signature_len);
        
        file.read(&kernel.kernelSource[0], kernel_source_len);
        file.read(&kernel.moduleSource[0], module_source_len);
        file.read(&kernel.name[0], name_len);
        file.read(&kernel.signature[0], signature_len);
        
        // Read arguments
        uint32_t num_args;
        file.read(reinterpret_cast<char*>(&num_args), sizeof(uint32_t));
        
        if (num_args > 100) {  // Arbitrary reasonable limit
            std::cerr << "Error: Invalid argument count: " << num_args << std::endl;
            throw std::runtime_error("Invalid argument count in kernel deserialization");
        }
        
        //std::cout << "Reading " << num_args << " arguments" << std::endl;
        kernel.arguments.reserve(num_args);
        
        for (uint32_t i = 0; i < num_args; i++) {
            // Create a temporary argument with placeholder values
            Argument arg("", "");
            // Deserialize into the temporary argument
            arg.deserialize(file);
            // Add the deserialized argument to the kernel
            kernel.arguments.push_back(std::move(arg));
        }
        
        return kernel;
    }


};

class KernelManager {
    std::vector<Kernel> kernels;
    std::unordered_map<hipFunction_t, std::string> rtc_kernel_names;
    std::unordered_map<hiprtcProgram, std::string> rtc_program_sources;
    std::vector<std::string> object_files;

    // Get kernel object file
 std::string getKernelObjectFile(const void* function_address) const {
    //std::cout << "\nSearching for kernel object file containing address " 
    //          << function_address << std::endl;
              
    std::queue<std::string> files_to_check;
    std::set<std::string> checked_files;
    
    // Start with /proc/self/exe
    files_to_check.push("/proc/self/exe");
    //std::cout << "Starting search with /proc/self/exe" << std::endl;
    
    // Helper function to get dependencies using ldd
    auto getDependencies = [](const std::string& path) {
        std::vector<std::string> deps;
        std::string cmd = "ldd " + path;
        //std::cout << "Running: " << cmd << std::endl;
        
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
        if (!pipe) {
            std::cerr << "Failed to run ldd: " << strerror(errno) << std::endl;
            return deps;
        }
        
        char buffer[512];
        while (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
            std::string line(buffer);
            // Look for => in ldd output
            size_t arrow_pos = line.find("=>");
            if (arrow_pos != std::string::npos) {
                // Extract path after =>
                size_t path_start = line.find('/', arrow_pos);
                size_t path_end = line.find(" (", arrow_pos);
                if (path_start != std::string::npos && path_end != std::string::npos) {
                    std::string dep = line.substr(path_start, path_end - path_start);
                    deps.push_back(dep);
                    std::cout << "Found dependency: " << dep << std::endl;
                }
            }
        }
        return deps;
    };
    
    // Helper function to check if address is in file
    auto isAddressInFile = [](const std::string& path, const void* addr) {
        //std::cout << "Checking if address " << addr << " is in " << path << std::endl;
        
        struct CallbackData {
            const void* target_addr;
            bool found;
            std::string found_path;
        };
        
        CallbackData data = {addr, false, ""};
        
        // Callback for dl_iterate_phdr
        auto callback = [](struct dl_phdr_info* info, size_t size, void* data) {
            auto params = static_cast<CallbackData*>(data);
            const void* target_addr = params->target_addr;
            
            std::string lib_path = info->dlpi_name[0] ? info->dlpi_name : "/proc/self/exe";
            //std::cout << "Checking segments in " << lib_path
            //          << " at base address " << (void*)info->dlpi_addr << std::endl;
            
            for (int j = 0; j < info->dlpi_phnum; j++) {
                const ElfW(Phdr)* phdr = &info->dlpi_phdr[j];
                if (phdr->p_type == PT_LOAD) {
                    void* start = (void*)(info->dlpi_addr + phdr->p_vaddr);
                    void* end = (void*)((char*)start + phdr->p_memsz);
                    //std::cout << "  Segment " << j << ": " << start << " - " << end << std::endl;
                    
                    if (target_addr >= start && target_addr < end) {
                        //std::cout << "  Found address in this segment!" << std::endl;
                        params->found = true;
                        params->found_path = lib_path;
                        return 1;  // Stop iteration
                    }
                }
            }
            return 0;  // Continue iteration
        };
        
        dl_iterate_phdr(callback, &data);
        
        if (!data.found) {
            //std::cout << "Address not found in " << path << std::endl;
            return std::make_pair(false, std::string());
        }
        
        return std::make_pair(true, data.found_path);
    };
    
    while (!files_to_check.empty()) {
        std::string current_file = files_to_check.front();
        files_to_check.pop();
        
        if (checked_files.count(current_file)) {
            //std::cout << "Already checked " << current_file << ", skipping" << std::endl;
            continue;
        }
        
        //std::cout << "\nChecking file: " << current_file << std::endl;
        checked_files.insert(current_file);
        
        // Check if the function_address is in this file
        auto [found, actual_path] = isAddressInFile(current_file, function_address);
        if (found) {
            char resolved_path[PATH_MAX];
            if (realpath(actual_path.c_str(), resolved_path) != nullptr) {
                std::string abs_path(resolved_path);
                std::cout << "Found kernel in " << abs_path << std::endl;
                return abs_path;
            }
            // If realpath fails, return the original path
            std::cout << "Found kernel in " << actual_path << std::endl;
            return actual_path;
        }
        // Add dependencies to queue
        //std::cout << "Getting dependencies for " << current_file << std::endl;
        for (const auto& dep : getDependencies(current_file)) {
            if (!checked_files.count(dep)) {
                //std::cout << "Adding to queue: " << dep << std::endl;
                files_to_check.push(dep);
            } else {
                //std::cout << "Already checked dependency: " << dep << std::endl;
            }
        }
    }
    std::cerr << "Searched the following files for kernel address " << function_address << std::endl;
    for (const auto& file : checked_files) {
            std::cerr << "  " << file << std::endl;
                }
        std::abort();
    }

public:
    void operator<<(std::ostream& os) const {
        os << "KernelManager: " << kernels.size() << " kernels" << std::endl;
        for (const auto& kernel : kernels) {
            kernel << os;
        }
    }

    KernelManager() {}
    ~KernelManager() {}

    void writeKernelManagerHeader(std::ofstream& file) {
        //  std::cout << "Writing kernel manager header at position " << file.tellp();
        //std::cout << " - Magic: 0x" << std::hex << KRNL_MAGIC << ", Version: " << std::dec << KRNL_VERSION << std::endl;
        uint32_t magic = KRNL_MAGIC;  // KRNL
        uint32_t version = KRNL_VERSION;
        file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    }

    void readKernelManagerHeader(std::ifstream& file) {
        uint32_t magic;
        uint32_t version;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        //std::cout << "Reading kernel manager header at position " << file.tellg();
        //std::cout << " - Magic: 0x" << std::hex << magic << ", Version: " << std::dec << version << std::endl;
        if (magic != KRNL_MAGIC) {
            std::cerr << "Invalid kernel manager format in trace (magic: 0x" 
                      << std::hex << magic << ", expected: 0x" << KRNL_MAGIC << ")" << std::endl;
            throw std::runtime_error("Invalid kernel manager format in trace");
        }
        if (version != KRNL_VERSION) {
            std::cerr << "Unsupported kernel manager version in trace" << std::endl;
            throw std::runtime_error("Unsupported kernel manager version in trace");
        }
    }

    void addFromBinary(std::string object_file) {
        // Check if object file already processed
        if (std::find(object_files.begin(), object_files.end(), object_file) != object_files.end()) {
            std::cerr << "KernelManager: Object file " << object_file << " already processed" << std::endl;
            std::abort();
        }
        const_cast<KernelManager*>(this)->object_files.push_back(object_file);
        
        // Get the base address where this object file is loaded
        uintptr_t base_addr = getBaseAddr(object_file);
        
        std::cout << "Object file loaded at base address: 0x" << std::hex << base_addr << std::dec << std::endl;
        
        // Use nm to get symbol information from the object file
        std::string cmd = "nm -C " + object_file + " | grep __device_stub__";
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            std::cerr << "Failed to run nm command: " << strerror(errno) << std::endl;
            return;
        }

        char buffer[1024];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            std::string line(buffer);
            
            // Parse the nm output format: "address T symbol_name"
            std::istringstream iss(line);
            std::string addr_str;
            std::string symbol_type;
            std::string symbol_name;
            
            // Read the address, type, and symbol name
            if (!(iss >> addr_str >> symbol_type)) {
                std::cerr << "Failed to parse nm output line: " << line << std::endl;
                continue;
            }
            
            // Get the rest of the line as the symbol name
            std::getline(iss >> std::ws, symbol_name);
            
            // Convert hex string to void* address and add base address for device stub
            void* device_address = nullptr;
            try {
                uintptr_t addr = std::stoull(addr_str, nullptr, 16);
                addr += base_addr;  // Add base address to get actual runtime address
                device_address = reinterpret_cast<void*>(addr);
            } catch (const std::exception& e) {
                std::cerr << "Failed to convert device address: " << addr_str << " - " << e.what() << std::endl;
                continue;
            }
            
            // Create kernel and set device address
            Kernel kernel(symbol_name);
            kernel.setDeviceAddress(device_address);
                
                // Look up the host function address
                std::string host_cmd = "nm -C " + object_file + " | grep -w \"" + kernel.getName() + "\"";
                std::cout << "Running host command: " << host_cmd << std::endl;
                FILE* host_pipe = popen(host_cmd.c_str(), "r");
                if (host_pipe) {
                    char host_buffer[1024];
                    if (fgets(host_buffer, sizeof(host_buffer), host_pipe)) {
                        std::string host_line(host_buffer);
                        std::istringstream host_iss(host_line);
                        std::string host_addr_str;
                        std::string host_symbol_type;
                        std::cout << "host_line: " << host_line << std::endl;
                        
                        if (host_iss >> host_addr_str >> host_symbol_type) {
                            try {
                                uintptr_t host_addr = std::stoull(host_addr_str, nullptr, 16);
                                host_addr += base_addr;
                                void* host_address = reinterpret_cast<void*>(host_addr);
                                kernel.setHostAddress(host_address);
                                std::cout << "Found kernel '" << kernel.getName() << "'"
                                          << "\n  device address: " << std::hex << device_address
                                          << "\n  host address: " << host_address << std::dec << std::endl;
                            } catch (const std::exception& e) {
                                std::cerr << "Failed to convert host address: " << host_addr_str 
                                          << " - " << e.what() << std::endl;
                            }
                        }
                    }
                    pclose(host_pipe);
                }
            
            kernels.push_back(kernel);
        }
        pclose(pipe);
    }

    uintptr_t getBaseAddr(const std::string& target_file) {
        // Use the same logic as in addFromBinary to get the base address
        struct CallbackData {
            const std::string& target_file;
            uintptr_t base_addr;
        };
        
        CallbackData data = {target_file, 0};
        
        dl_iterate_phdr([](dl_phdr_info* info, size_t size, void* data_ptr) {
            auto data = reinterpret_cast<CallbackData*>(data_ptr);
            if (info->dlpi_name && strlen(info->dlpi_name) > 0) {
                char real_path[PATH_MAX];
                char target_real_path[PATH_MAX];    
                
                if (realpath(info->dlpi_name, real_path) && 
                    realpath(data->target_file.c_str(), target_real_path)) {
                    
                    if (strcmp(real_path, target_real_path) == 0) {
                        std::cout << "Found matching object file: " << real_path 
                                  << " at base address: 0x" << std::hex << info->dlpi_addr << std::dec << std::endl;
                        data->base_addr = info->dlpi_addr;
                        return 1;  // Stop iteration
                    }
                }
            }
            return 0;
        }, &data);
        
        if (data.base_addr == 0) {
            std::cout << "Warning: Could not find base address for " << target_file << std::endl;
            // Try /proc/self/exe if target file wasn't found
            if (realpath("/proc/self/exe", nullptr) == target_file) {
                dl_iterate_phdr([](dl_phdr_info* info, size_t size, void* data_ptr) {
                    auto base_addr_ptr = reinterpret_cast<uintptr_t*>(data_ptr);
                    if (!info->dlpi_name || strlen(info->dlpi_name) == 0) {  // Main executable has empty name
                        *base_addr_ptr = info->dlpi_addr;
                        return 1;
                    }
                    return 0;
                }, &data.base_addr);
            }
        }
        
        return data.base_addr;
    }

    Kernel getKernelByPointer(const void* host_fund_addr) {
        // First, search kernels by function_address
        auto it = std::find_if(kernels.begin(), kernels.end(),
            [&](const Kernel& k) { return k.getHostAddress() == host_fund_addr; });
        if (it != kernels.end()) {
            return *it;
        }
        // If not found, create kernels from binary device stubs
        auto object_file = getKernelObjectFile(host_fund_addr);
        addFromBinary(object_file); // this will abort if object_file is already processed
        return getKernelByPointer(host_fund_addr);
    }

    void addFromModuleSource(const std::string& module_source) {
        if (module_source.empty()) {
            std::cout << "Empty source provided to KernelManager" << std::endl;
            return;
        }

        std::cout << "Processing module source of length " << module_source.length() << std::endl;

        // Regex to match __global__ kernel declarations
        std::regex kernel_regex(R"(__global__\s+\w+\s+(\w+)\s*\(([^)]*)\))");
        
        std::sregex_iterator it(module_source.begin(), module_source.end(), kernel_regex);
        std::sregex_iterator end;

        size_t kernels_found = 0;
        while (it != end) {
            std::smatch match = *it;
            if (match.size() >= 3) {
                Kernel kernel(match[0].str());
                kernel.setModuleSource(module_source);
                
                // Add to kernels vector if not already present
                auto existing = std::find_if(kernels.begin(), kernels.end(),
                    [&](const Kernel& k) { return k.getSignature() == kernel.getSignature(); });
                    
                if (existing == kernels.end()) {
                    kernels.push_back(kernel);
                    std::cout << "Added kernel: " << kernel.getSignature() << std::endl;
                    kernels_found++;
                }
            }
            ++it;
        }
        
        std::cout << "Found " << kernels_found << " new kernels in module source" << std::endl;
    }

    Kernel getKernelByName(const std::string& name) const {
        auto kernel_it = std::find_if(kernels.begin(), kernels.end(),
            [&](const Kernel& k) { return k.getName() == name; });
        
        if (kernel_it != kernels.end())
            return *kernel_it;

        // Try to demangle the name first
        std::string demangled = demangle(name);
        
        // Extract just the kernel name from the demangled signature
        size_t pos = demangled.find('(');
        std::string kernel_name = pos != std::string::npos ? 
            demangled.substr(0, pos) : demangled;
        
        kernel_it = std::find_if(kernels.begin(), kernels.end(),
            [&](const Kernel& k) { return k.getName() == kernel_name; });
        
        if (kernel_it == kernels.end()) {
            std::cerr << "No kernel found with name: " << kernel_name << std::endl;
            std::cerr << "Available kernels: " << std::endl;
            for (const auto& kernel : kernels) {
                std::cerr << "  " << kernel.getName() << std::endl;
            }
        }
        
        return *kernel_it;
    }

    // Add direct serialization methods for use in trace file
    void serialize(std::ofstream& file) const {
        uint32_t num_kernels = kernels.size();
        //std::cout << "Serializing " << num_kernels << " kernels" << std::endl;
        
        // Write kernel count
        file.write(reinterpret_cast<const char*>(&num_kernels), sizeof(uint32_t));
        
        // Write each kernel
        for (const auto& kernel : kernels) {
            //std::cout << "Serializing kernel: " << kernel.getName() << std::endl;
            kernel.serialize(file);
            file.flush();  // Ensure data is written
        }
    }

    void deserialize(std::ifstream& file) {
        uint32_t num_kernels;
        file.read(reinterpret_cast<char*>(&num_kernels), sizeof(uint32_t));
        
        //std::cout << "File position before reading kernel count: " << file.tellg() - std::streamoff(sizeof(uint32_t)) << std::endl;
        //std::cout << "Kernel count value read: " << num_kernels << std::endl;
        
        // Sanity check on kernel count
        if (num_kernels > 1000) {  // Arbitrary reasonable limit
            std::cerr << "Error: Invalid kernel count: " << num_kernels << std::endl;
            throw std::runtime_error("Invalid kernel count in deserialization");
        }
        
        //std::cout << "Starting to deserialize " << num_kernels << " kernels" << std::endl;
        
        kernels.clear();  // Clear any existing kernels
        kernels.reserve(num_kernels);  // Reserve space for efficiency

        for (uint32_t i = 0; i < num_kernels; i++) {
            //std::cout << "\nDeserializing kernel " << i + 1 << " of " << num_kernels 
                      //<< " at position " << file.tellg() << std::endl;
            try {
                kernels.push_back(Kernel::deserialize(file));
                //std::cout << "Successfully deserialized kernel " << i + 1 << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error deserializing kernel " << i + 1 << ": " << e.what() << std::endl;
                throw;
            }
        }
        
        //std::cout << "Successfully deserialized all " << kernels.size() << " kernels" << std::endl;
    }

    size_t getNumKernels() const { return kernels.size(); }

    void addRTCKernel(hipFunction_t func, const std::string& name) {
        rtc_kernel_names[func] = name;
    }

    std::string getRTCKernelName(hipFunction_t func) const {
        auto it = rtc_kernel_names.find(func);
        if (it != rtc_kernel_names.end()) {
            return it->second;
        }
        return "";
    }

    void addRTCProgram(hiprtcProgram prog, const std::string& source) {
        rtc_program_sources[prog] = source;
    }

    std::string getRTCProgramSource(hiprtcProgram prog) const {
        auto it = rtc_program_sources.find(prog);
        if (it != rtc_program_sources.end()) {
            return it->second;
        }
        return "";
    }
};

#endif // HIP_INTERCEPT_LAYER_KERNEL_MANAGER_HH