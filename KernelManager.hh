#ifndef HIP_INTERCEPT_LAYER_KERNEL_MANAGER_HH
#define HIP_INTERCEPT_LAYER_KERNEL_MANAGER_HH

#include <string>
#include <vector>
#include <iostream>
#include <regex>
#include <stack>
#include <fstream>

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
    Argument(std::string name, std::string type) {
        if (name.empty() || type.empty()) {
            std::cerr << "Warning: Creating argument with empty name or type" << std::endl;
        }
        this->name = name;
        this->type = type;
        this->size = isVector() ? 16 : sizeof(void*);
        this->is_pointer = type.find("*") != std::string::npos;
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

    void serialize(std::ofstream& file) const {
        uint32_t name_len = name.length();
        uint32_t type_len = type.length();
        
        //std::cout << "Serializing argument - Name: '" << name << "' (len=" << name_len 
        //          << "), Type: '" << type << "' (len=" << type_len << ")" << std::endl;
        
        file.write(reinterpret_cast<const char*>(&name_len), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&type_len), sizeof(uint32_t));
        
        file.write(name.c_str(), name_len);
        file.write(type.c_str(), type_len);
        
        file.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(&is_pointer), sizeof(bool));
    }

    static Argument deserialize(std::ifstream& file) {
        uint32_t name_len, type_len;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&type_len), sizeof(uint32_t));
        
        std::cout << "Reading strings with lengths: name_len=" << name_len 
                  << ", type_len=" << type_len << std::endl;
        
        if (name_len > 1000 || type_len > 1000) {
            std::cerr << "Warning: Suspiciously large string lengths detected" << std::endl;
            throw std::runtime_error("Invalid string lengths in deserialization");
        }
        
        // Create vectors to hold the string data
        std::vector<char> name_buffer(name_len + 1, '\0');  // +1 for null termination
        std::vector<char> type_buffer(type_len + 1, '\0');
        
        // Read the string data into the buffers
        file.read(name_buffer.data(), name_len);
        file.read(type_buffer.data(), type_len);
        
        // Debug output of raw data
        std::cout << "Raw name buffer: ";
        for (size_t i = 0; i < name_len && i < 20; ++i) {
            std::cout << std::hex << (int)(unsigned char)name_buffer[i] << " ";
        }
        std::cout << std::dec << std::endl;
        
        std::cout << "Raw type buffer: ";
        for (size_t i = 0; i < type_len && i < 20; ++i) {
            std::cout << std::hex << (int)(unsigned char)type_buffer[i] << " ";
        }
        std::cout << std::dec << std::endl;
        
        // Convert to strings
        std::string name(name_buffer.data(), name_len);
        std::string type(type_buffer.data(), type_len);
        
        std::cout << "String lengths after conversion: name=" << name.length() 
                  << ", type=" << type.length() << std::endl;
        
        // Create argument with the read data
        Argument arg(name, type);
        
        // Read the remaining fields
        file.read(reinterpret_cast<char*>(&arg.size), sizeof(size_t));
        file.read(reinterpret_cast<char*>(&arg.is_pointer), sizeof(bool));
        
        return arg;
    }
};

class Kernel {
    std::string kernelSource;
    std::string moduleSource;
    std::string name;
    std::string signature;
    std::vector<Argument> arguments;




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

        size_t nameEnd = signature.find('(');
        if (nameEnd != std::string::npos) {
            // Extract everything after __global__ void
            std::string afterGlobal = signature.substr(signature.find("void") + 4);
            // Extract just the kernel name
            this->name = afterGlobal.substr(0, afterGlobal.find('(')); 
            // Trim any whitespace
            this->name.erase(0, this->name.find_first_not_of(" "));
            this->name.erase(this->name.find_last_not_of(" ") + 1);
        }
        std::cout << "  Kernel name: " << this->name << std::endl;
        std::cout << "  Kernel signature: " << this->signature << std::endl;

        // Parse arguments
        std::regex arg_regex(R"((\w+(?:\s*\*\s*(?:__restrict__)?)?)\s+(\w+)(?:\s*,|\s*\)))");
        std::string args = signature.substr(nameEnd + 1);
        std::sregex_iterator it(args.begin(), args.end(), arg_regex);
        std::sregex_iterator end;
        
        std::cout << "  Parsing arguments..." << std::endl;
        while (it != end) {
            std::smatch match = *it;
            arguments.push_back(Argument(match[2].str(), match[1].str()));
            ++it;
        }

        std::cout << "  Found " << arguments.size() << " arguments" << std::endl;
    }

    void serialize(std::ofstream& file) const {
        // Write string lengths
        uint32_t kernel_source_len = kernelSource.length();
        uint32_t module_source_len = moduleSource.length();
        uint32_t name_len = name.length();
        uint32_t signature_len = signature.length();
        
        std::cout << "Writing kernel '" << name << "' with lengths:"
                  << "\n  kernel_source: " << kernel_source_len
                  << "\n  module_source: " << module_source_len
                  << "\n  name: " << name_len
                  << "\n  signature: " << signature_len << std::endl;
        
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
        
        std::cout << "Writing " << num_args << " arguments" << std::endl;
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
        
        std::cout << "Reading kernel with lengths:"
                  << "\n  kernel_source: " << kernel_source_len
                  << "\n  module_source: " << module_source_len
                  << "\n  name: " << name_len
                  << "\n  signature: " << signature_len << std::endl;
                  
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
        
        std::cout << "Reading " << num_args << " arguments" << std::endl;
        kernel.arguments.reserve(num_args);
        
        for (uint32_t i = 0; i < num_args; i++) {
            kernel.arguments.push_back(Argument::deserialize(file));
        }
        
        return kernel;
    }
};

class KernelManager {
    std::vector<Kernel> kernels;
public:
    KernelManager() {}
    ~KernelManager() {}

    void writeKernelManagerHeader(std::ofstream& file) {
        std::cout << "Writing kernel manager header at position " << file.tellp();
        std::cout << " - Magic: 0x" << std::hex << KRNL_MAGIC << ", Version: " << std::dec << KRNL_VERSION << std::endl;
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
        std::cout << "Reading kernel manager header at position " << file.tellg();
        std::cout << " - Magic: 0x" << std::hex << magic << ", Version: " << std::dec << version << std::endl;
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
        auto it = std::find_if(kernels.begin(), kernels.end(),
            [&](const Kernel& k) { return k.getName() == name; });
            
        if (it != kernels.end()) {
            return *it;
        }
        
        return getKernelByNameMangled(name);
    }

    Kernel getKernelByNameMangled(const std::string& name) const {
        // Try to demangle the name first
        std::string demangled = demangle(name);
        
        // Extract just the kernel name from the demangled signature
        size_t pos = demangled.find('(');
        std::string kernel_name = pos != std::string::npos ? 
            demangled.substr(0, pos) : demangled;
            
        auto it = std::find_if(kernels.begin(), kernels.end(),
            [&](const Kernel& k) { return k.getName() == kernel_name; });
            
        if (it == kernels.end()) {
            std::cerr << "No kernel found with name: " << kernel_name << std::endl;
            std::abort();
        }
        
        return *it;
    }

    // Add direct serialization methods for use in trace file
    void serialize(std::ofstream& file) const {
        uint32_t num_kernels = kernels.size();
        std::cout << "Serializing " << num_kernels << " kernels" << std::endl;
        
        // Write kernel count
        file.write(reinterpret_cast<const char*>(&num_kernels), sizeof(uint32_t));
        
        // Write each kernel
        for (const auto& kernel : kernels) {
            std::cout << "Serializing kernel: " << kernel.getName() << std::endl;
            kernel.serialize(file);
            file.flush();  // Ensure data is written
        }
    }

    void deserialize(std::ifstream& file) {
        uint32_t num_kernels;
        file.read(reinterpret_cast<char*>(&num_kernels), sizeof(uint32_t));
        
        std::cout << "File position before reading kernel count: " << file.tellg() - std::streamoff(sizeof(uint32_t)) << std::endl;
        std::cout << "Kernel count value read: " << num_kernels << std::endl;
        
        // Sanity check on kernel count
        if (num_kernels > 1000) {  // Arbitrary reasonable limit
            std::cerr << "Error: Invalid kernel count: " << num_kernels << std::endl;
            throw std::runtime_error("Invalid kernel count in deserialization");
        }
        
        std::cout << "Starting to deserialize " << num_kernels << " kernels" << std::endl;
        
        kernels.clear();  // Clear any existing kernels
        kernels.reserve(num_kernels);  // Reserve space for efficiency

        for (uint32_t i = 0; i < num_kernels; i++) {
            std::cout << "\nDeserializing kernel " << i + 1 << " of " << num_kernels 
                      << " at position " << file.tellg() << std::endl;
            try {
                kernels.push_back(Kernel::deserialize(file));
                std::cout << "Successfully deserialized kernel " << i + 1 << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error deserializing kernel " << i + 1 << ": " << e.what() << std::endl;
                throw;
            }
        }
        
        std::cout << "Successfully deserialized all " << kernels.size() << " kernels" << std::endl;
    }

    size_t getNumKernels() const { return kernels.size(); }
};

#endif // HIP_INTERCEPT_LAYER_KERNEL_MANAGER_HH