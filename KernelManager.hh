#ifndef HIP_INTERCEPT_LAYER_KERNEL_MANAGER_HH
#define HIP_INTERCEPT_LAYER_KERNEL_MANAGER_HH

#include <string>
#include <vector>
#include <iostream>
#include <regex>


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
public:
    std::string name;
    std::string signature;
    std::string source;
    std::string binary;
    std::vector<Argument> arguments;

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