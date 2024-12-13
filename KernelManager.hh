#ifndef HIP_INTERCEPT_LAYER_KERNEL_MANAGER_HH
#define HIP_INTERCEPT_LAYER_KERNEL_MANAGER_HH

#include <string>
#include <vector>
#include <iostream>
#include <regex>
#include <stack>


class Argument {
public:
    std::string name;
    std::string type;
    std::string value;

    Argument(std::string name, std::string type) {
        this->name = name;
        this->type = type;
        std::cout << "    Argument: " << this->type << " " << this->name << std::endl;
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
        std::cout << "Kernel source:\n" << kernelSource << std::endl;
    }
public:

    const std::string getName() const {
        return name;
    }

    const std::string getSignature() const {
        return signature;
    }

    const std::string getSource() const {
        return kernelSource;
    }

    const std::string getModuleSource() const {
        return moduleSource;
    }

    void setModuleSource(const std::string& module_source) {
        this->moduleSource = module_source;
        parseKernelSource();
    }
    Kernel() {}

    Kernel(std::string signature) {
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
        std::cout << "Kernel name: " << this->name << std::endl;
        std::cout << "Kernel signature: " << this->signature << std::endl;

        // Updated regex to handle more complex parameter formats
        std::regex arg_regex(R"((\w+(?:\s*\*\s*(?:__restrict__)?)?)\s+(\w+)(?:\s*,|\s*\)))");
        std::string args = signature.substr(nameEnd + 1);
        std::sregex_iterator it(args.begin(), args.end(), arg_regex);
        std::sregex_iterator end;
        while (it != end) {
            std::smatch match = *it;
            arguments.push_back(Argument(match[2].str(), match[1].str()));
            ++it;
        }
    }
};

class KernelManager {
    std::vector<Kernel> kernels;
public:
    KernelManager();
    ~KernelManager();
    /// @brief Add kernels from a module source file using regex
    /// @param module_source 
    void addFromModuleSource(const std::string& module_source);
    Kernel getKernelByName(const std::string& name);
    Kernel getKernelByNameMangled(const std::string& name);
};

#endif // HIP_INTERCEPT_LAYER_KERNEL_MANAGER_HH