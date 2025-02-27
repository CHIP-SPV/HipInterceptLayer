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
#include "common.h"

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

static inline bool isKeyword(const std::string& str) {
    return std::find(known_keywords.begin(), known_keywords.end(), str) != known_keywords.end();
}

static inline bool isLastTokenKnownKeyword(const std::string& str) {
    // Find the last space that's not inside template brackets
    int template_depth = 0;
    size_t last_space = std::string::npos;
    
    for (size_t i = 0; i < str.length(); i++) {
        if (str[i] == '<') template_depth++;
        else if (str[i] == '>') template_depth--;
        else if (str[i] == ' ' && template_depth == 0) last_space = i;
    }

    std::string last_token;
    if (last_space != std::string::npos) {
        last_token = str.substr(last_space + 1);
    } else {
        last_token = str;
    }

    // Remove any trailing pointer asterisks
    last_token.erase(std::remove(last_token.begin(), last_token.end(), '*'), last_token.end());

    auto is_keyword = isKeyword(last_token);
    auto is_vector = last_token.find("HIP_vector_type") != std::string::npos;
    return is_keyword || is_vector;
}

static inline std::string trimWhiteSpaces(const std::string& str) {
    std::string result = str;
    
    // Remove leading whitespace
    result.erase(0, result.find_first_not_of(" \t\n\r\f\v"));
    
    // Remove trailing whitespace
    result.erase(result.find_last_not_of(" \t\n\r\f\v") + 1);
    
    // Replace multiple spaces with single space
    result = std::regex_replace(result, std::regex("\\s+"), " ");
    
    return result;
}

static inline std::vector<std::string> splitArgs(const std::string& source) {
    std::vector<std::string> result;
    std::string current;
    int angle_bracket_count = 0;
    
    // Iterate through each character
    for (char c : source) {
        if (c == '<') {
            angle_bracket_count++;
            current += c;
        } else if (c == '>') {
            angle_bracket_count--;
            current += c;
        } else if (c == ',' && angle_bracket_count == 0) {
            // Only split on comma if we're not inside angle brackets
            if (!current.empty()) {
                result.push_back(current);
                current.clear();
            }
        } else {
            current += c;
        }
    }
    
    // Don't forget the last argument
    if (!current.empty()) {
        result.push_back(current);
    }
    
    // Trim each argument
    for (auto& arg : result) {
        arg = trimWhiteSpaces(arg);
    }
    
    std::cout << "splitArgs completed. Found " << result.size() << " arguments." << std::endl;
    return result;
}

static inline std::pair<std::string, std::string> processArgWithRename(const std::string& arg, int idx) {
    std::string type, name;
    std::string newName("arg" + std::to_string(idx));
    std::string arg_copy = arg;

    if (isLastTokenKnownKeyword(arg)) {
        arg_copy = arg_copy + " " + newName;
    }

    type = arg_copy.substr(0, arg_copy.find_last_of(' '));
    name = arg_copy.substr(arg_copy.find_last_of(' ') + 1);

    // if name starts with *, move it to the end of the type
    if (name[0] == '*') {
        type = type + "*";
        name = name.substr(1);
    }

    return std::make_pair(type, name);
}

class Argument {
public:
    std::string name;
    std::string type;
    size_t size;
    bool is_pointer;
    bool is_vector;

private:
    const std::unordered_map<std::string, size_t> type_sizes{
            {"bool", sizeof(bool)},
            {"char", sizeof(char)},
            {"short", sizeof(short)},
            {"int", sizeof(int)},
            {"long", sizeof(long)},
            {"float", sizeof(float)},
            {"double", sizeof(double)},
            {"size_t", sizeof(size_t)},
            {"int8_t", sizeof(int8_t)},
            {"uint8_t", sizeof(uint8_t)},
            {"int16_t", sizeof(int16_t)},
            {"uint16_t", sizeof(uint16_t)},
            {"int32_t", sizeof(int32_t)},
            {"uint32_t", sizeof(uint32_t)},
            {"int64_t", sizeof(int64_t)},
            {"uint64_t", sizeof(uint64_t)},
            {"long long", sizeof(long long)},
            {"char2", 2 * sizeof(char)},
            {"uchar2", 2 * sizeof(unsigned char)},
            {"float4", 4 * sizeof(float)}
    };

    size_t parseTypeToInt(const std::string& type) const {
        size_t longest_match = 0;
        size_t size = 1;
        
        for (const auto& type_size : type_sizes) {
            if (type.find(type_size.first) != std::string::npos) {
                if (type_size.first.length() > longest_match) {
                    longest_match = type_size.first.length();
                    size = type_size.second;
                }
            }
        }

        if (longest_match == 0) {
            if (is_pointer) 
                return sizeof(void*); 

            std::cerr << "Unknown type : " << type << std::endl;
            std::abort();
        }
        
        return size;
    }

    static std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t\n\r");
        size_t last = str.find_last_not_of(" \t\n\r");
        if (first == std::string::npos || last == std::string::npos)
            return "";
        return str.substr(first, last - first + 1);
    }

public:
    void operator<<(std::ostream& os) const {
        os << "Argument: " << type << " " << name << " (size: " << size << " is_pointer: " << is_pointer << " is_vector: " << is_vector << ")\n";
    }
    Argument(std::string name, std::string type) {
        if (name.empty() || type.empty()) {
            //std::cerr << "Warning: Creating argument with empty name or type" << std::endl;
        }
        this->name = name;
        this->type = type;
        this->is_pointer = type.find("*") != std::string::npos;
        this->is_vector = getVectorSize() > 0;
        this->size = getTypeSize();
        std::cout << "    Argument: " << this->type << " " << this->name 
                  << " (size: " << this->size << ")" << std::endl;
    }

    size_t getTypeSize() const {
        // Get the base type by removing pointer if present
        std::string base_type = type;
        bool is_ptr = base_type.find('*') != std::string::npos;
        if (is_ptr) {
            base_type = base_type.substr(0, base_type.find('*'));
        }

        // Handle vector types
        size_t vec_size = getVectorSize();
        if (vec_size > 0) {
            // For vector types, multiply base type size by vector size
            if (base_type.find("float") != std::string::npos || base_type.find("int") != std::string::npos || 
                base_type.find("uint") != std::string::npos) {
                return vec_size * 4;  // 4 bytes per element
            }
            if (base_type.find("double") != std::string::npos) {
                return vec_size * 8;  // 8 bytes per element
            }
        }

        // Trim and normalize the base type
        base_type = trim(base_type);
        
        // Look up the type size in our map
        return parseTypeToInt(base_type);
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

    size_t getVectorSize() const {
        // Handle HIP vector types
        if (type.find("HIP_vector_type") != std::string::npos) {
            size_t template_start = type.find('<');
            size_t template_end = type.find_last_of('>');
            if (template_start != std::string::npos && template_end != std::string::npos) {
                std::string template_params = type.substr(template_start + 1, template_end - template_start - 1);
                // Extract the vector size (second template parameter)
                size_t comma_pos = template_params.find(',');
                if (comma_pos != std::string::npos) {
                    std::string size_str = trim(template_params.substr(comma_pos + 1));
                    try {
                        return std::stoul(size_str);
                    } catch (...) {
                        return 0;
                    }
                }
            }
            return 0;
        }

        // Handle standard vector types (float2, float3, float4, etc.)
        std::string base = type;
        size_t ptr_pos = base.find('*');
        if (ptr_pos != std::string::npos) {
            base = base.substr(0, ptr_pos);
        }

        static const std::vector<std::string> vector_types = {
            "float4", "float3", "float2",
            "int4", "int3", "int2",
            "uint4", "uint3", "uint2",
            "double4", "double3", "double2",
            "long4", "long3", "long2",
            "ulong4", "ulong3", "ulong2",
            "char4", "char3", "char2",
            "uchar4", "uchar3", "uchar2"
        };

        for (const auto& vtype : vector_types) {
            if (base.find(vtype) != std::string::npos) {
                char suffix = vtype.back();
                return suffix - '0';
            }
        }

        return 0;  // Not a vector type
    }

    void printValue(std::ostream& os, const void* param_value) const {
        if (!param_value) {
            os << "null";
            return;
        }

        std::string base = getBaseType();
        size_t vec_size = getVectorSize();
        
        if (vec_size > 0) {
            // Handle vector types
            if (base.find("float") != std::string::npos) {
                const float* values = (const float*)param_value;
                os << "(";
                for (size_t i = 0; i < vec_size; i++) {
                    if (i > 0) os << ", ";
                    os << values[i];
                }
                os << ")";
            } else if (base.find("int") != std::string::npos) {
                const int* values = (const int*)param_value;
                os << "(";
                for (size_t i = 0; i < vec_size; i++) {
                    if (i > 0) os << ", ";
                    os << values[i];
                }
                os << ")";
            } else {
                os << "vector of unknown type " << base;
            }
        } else {
            // Handle scalar types
            if (base == "int") {
                os << *(const int*)param_value;
            } else if (base == "unsigned int") {
                os << *(const unsigned int*)param_value;
            } else if (base == "float") {
                os << *(const float*)param_value;
            } else if (base == "double") {
                os << *(const double*)param_value;
            } else if (base == "bool") {
                os << (*(const bool*)param_value ? "true" : "false");
            } else if (base == "size_t") {
                os << *(const size_t*)param_value;
            } else {
                os << "value of unknown type " << base << " at " << param_value;
            }
        }
    }

    std::string getType() const {
        return type;
    }

    std::string getBaseType() const {
        // Special handling for HIP vector types
        if (type.find("HIP_vector_type") != std::string::npos) {
            size_t template_start = type.find('<');
            size_t template_end = type.find_last_of('>');
            if (template_start != std::string::npos && template_end != std::string::npos) {
                std::string template_params = type.substr(template_start + 1, template_end - template_start - 1);
                // Extract the base type (first template parameter)
                size_t comma_pos = template_params.find(',');
                if (comma_pos != std::string::npos) {
                    return trim(template_params.substr(0, comma_pos));
                }
            }
            return type; // Return full type if parsing fails
        }

        // Remove pointer and const qualifiers
        std::string base = type;
        
        // Remove pointer
        size_t ptr_pos = base.find('*');
        if (ptr_pos != std::string::npos) {
            base = base.substr(0, ptr_pos);
        }
        
        // Remove const and restrict qualifiers
        static const std::vector<std::string> qualifiers = {"const", "restrict", "__restrict__"};
        for (const auto& qualifier : qualifiers) {
            size_t pos = base.find(qualifier);
            if (pos != std::string::npos) {
                base.erase(pos, qualifier.length());
            }
        }
        
        // Trim whitespace
        base.erase(0, base.find_first_not_of(" "));
        base.erase(base.find_last_not_of(" ") + 1);
        
        // Keep vector types intact (don't strip the number)
        static const std::vector<std::string> vector_types = {
            "float4", "float3", "float2",
            "int4", "int3", "int2",
            "uint4", "uint3", "uint2",
            "double4", "double3", "double2"
        };
        
        for (const auto& vtype : vector_types) {
            if (base.find(vtype) != std::string::npos) {
                return vtype;
            }
        }
        
        // For non-vector types, return the base type
        return base;
    }

    void serialize(std::ofstream& file) const {
        // std::cout << "[DEBUG] Serializing argument: " << name << " (" << type << ")" << std::endl;
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

    static Argument deserialize(std::ifstream& file) {
        // Read the argument name
        uint32_t name_length;
        file.read(reinterpret_cast<char*>(&name_length), sizeof(uint32_t));
        std::string name(name_length, '\0');
        file.read(&name[0], name_length);
        
        // Read the type
        uint32_t type_length;
        file.read(reinterpret_cast<char*>(&type_length), sizeof(uint32_t));
        std::string type(type_length, '\0');
        file.read(&type[0], type_length);
        
        // Create argument with actual name and type
        Argument arg(name, type);
        
        // Read other properties
        file.read(reinterpret_cast<char*>(&arg.size), sizeof(size_t));
        file.read(reinterpret_cast<char*>(&arg.is_pointer), sizeof(bool));
        file.read(reinterpret_cast<char*>(&arg.is_vector), sizeof(bool));
        
        return arg;
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

    // Add trim as a private static method in Kernel class
    static std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t\n\r");
        size_t last = str.find_last_not_of(" \t\n\r");
        if (first == std::string::npos || last == std::string::npos)
            return "";
        return str.substr(first, last - first + 1);
    }

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
        assert(args_start != std::string::npos && args_end != std::string::npos);
        std::string args_str = signature.substr(args_start + 1, args_end - args_start - 1);
        auto argsStrVec = splitArgs(args_str);
        std::vector<Argument> processedArgs;
        for (size_t i = 0; i < argsStrVec.size(); ++i) {
            auto new_arg = processArgWithRename(argsStrVec[i], i);
            processedArgs.push_back(Argument(new_arg.second, new_arg.first));
        }

        return {kernel_name, processedArgs};
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
        // std::cout << "[DEBUG] Serializing kernel: " << name << std::endl;
        // Write string lengths
        uint32_t kernel_source_len = kernelSource.length();
        uint32_t module_source_len = moduleSource.length();
        uint32_t name_len = name.length();
        uint32_t signature_len = signature.length();
        
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
        
        for (const auto& arg : arguments) {
            arg.serialize(file);
        }
        // std::cout << "[DEBUG] Finished serializing kernel: " << name << std::endl;
    }

    static Kernel deserialize(std::ifstream& file) {
        // std::cout << "[DEBUG] Deserializing kernel" << std::endl;
        Kernel kernel;
        
        // Read string lengths
        uint32_t kernel_source_len, module_source_len, name_len, signature_len;
        file.read(reinterpret_cast<char*>(&kernel_source_len), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&module_source_len), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&name_len), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&signature_len), sizeof(uint32_t));
        
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
        
        kernel.arguments.reserve(num_args);
        
        for (uint32_t i = 0; i < num_args; i++) {
            // Deserialize into the temporary argument
            Argument arg = Argument::deserialize(file);
            // Add the deserialized argument to the kernel
            kernel.arguments.push_back(std::move(arg));
        }
        
        // std::cout << "[DEBUG] Finished deserializing kernel: " << kernel.name << std::endl;
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
        // std::cout << "[DEBUG] Serializing KernelManager with " << kernels.size() << " kernels" << std::endl;
        uint32_t num_kernels = kernels.size();
        file.write(reinterpret_cast<const char*>(&num_kernels), sizeof(uint32_t));
        
        for (const auto& kernel : kernels) {
            kernel.serialize(file);
        }
        // std::cout << "[DEBUG] Finished serializing KernelManager" << std::endl;
    }

    void deserialize(std::ifstream& file) {
        // std::cout << "[DEBUG] Deserializing KernelManager" << std::endl;
        uint32_t num_kernels;
        file.read(reinterpret_cast<char*>(&num_kernels), sizeof(uint32_t));
        
        if (num_kernels > 1000) {
            std::cerr << "Error: Invalid kernel count: " << num_kernels << std::endl;
            throw std::runtime_error("Invalid kernel count in deserialization");
        }
        
        kernels.clear();
        kernels.reserve(num_kernels);

        for (uint32_t i = 0; i < num_kernels; i++) {
            kernels.push_back(Kernel::deserialize(file));
        }
        // std::cout << "[DEBUG] Finished deserializing KernelManager with " << kernels.size() << " kernels" << std::endl;
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