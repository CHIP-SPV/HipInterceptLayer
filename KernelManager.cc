#include "Interceptor.hh"
#include <regex>
#include <iostream>

KernelManager::KernelManager() {}
KernelManager::~KernelManager() {}

void KernelManager::addFromModuleSource(const std::string& module_source) {
    if (module_source.empty()) {
        std::cout << "Empty source provided to KernelManager" << std::endl;
        return;
    }

    // Regex to match __global__ kernel declarations
    std::regex kernel_regex(R"(__global__\s+\w+\s+(\w+)\s*\(([^)]*)\))");
    
    std::sregex_iterator it(module_source.begin(), module_source.end(), kernel_regex);
    std::sregex_iterator end;

    while (it != end) {
        std::smatch match = *it;
        if (match.size() >= 3) {
            Kernel kernel;
            kernel.name = match[1].str();
            kernel.signature = match[0].str();
            kernel.source = module_source;
            
            // Add to kernels vector if not already present
            auto existing = std::find_if(kernels.begin(), kernels.end(),
                [&](const Kernel& k) { return k.name == kernel.name; });
                
            if (existing == kernels.end()) {
                kernels.push_back(kernel);
                std::cout << "Added kernel: " << kernel.name 
                         << " with signature: " << kernel.signature << std::endl;
            }
        }
        ++it;
    }
}

Kernel KernelManager::getKernelByName(const std::string& name) {
    auto it = std::find_if(kernels.begin(), kernels.end(),
        [&](const Kernel& k) { return k.name == name; });
        
    if (it != kernels.end()) {
        return *it;
    }
    
    // Return empty kernel if not found
    return Kernel();
}

Kernel KernelManager::getKernelByNameMangled(const std::string& name) {
    // Try to demangle the name first
    std::string demangled = demangle(name);
    
    // Extract just the kernel name from the demangled signature
    size_t pos = demangled.find('(');
    std::string kernel_name = pos != std::string::npos ? 
        demangled.substr(0, pos) : demangled;
        
    return getKernelByName(kernel_name);
} 