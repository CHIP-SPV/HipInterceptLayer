class Kernel {
public:
    std::string name;
    std::string signature;
    std::string source;
    std::string binary;
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