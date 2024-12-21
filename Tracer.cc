#include "Tracer.hh"

// Tracer implementation
Tracer& Tracer::instance() {
    static Tracer instance_;
    return instance_;
}

Tracer::Tracer() :
    initialized_(false),
    current_execution_order_(0) {
    initializeTraceFile();
}

void Tracer::initializeTraceFile() {
    if (initialized_) return;
    
    file_path = getTraceFilePath();
    if (file_path.empty()) return; // Skip tracing for this process
    
    trace_file_.open(file_path, std::ios::binary);
    if (!trace_file_) {
        std::cerr << "Failed to open trace file: " << file_path << std::endl;
        return;
    }
    
    std::cout << "\n=== HIP Trace File ===\n"
              << "Writing trace to: " << file_path << "\n"
              << "===================\n\n";
    // Don't need to write tracer header here since this won't be the first thing in the file
    // Kernel Manager will write its header at the beginning of the file
    // This happens in finalizeTrace()
        
    initialized_ = true;
}

void Tracer::finalizeTrace() {
    if (!initialized_) return;
    
    std::cout << "Finalizing trace with " << kernel_manager_.getNumKernels() << " kernels" << std::endl;
    
    // Create a new file for the final trace
    std::string final_trace_path = file_path + ".final";
    std::ofstream final_trace(final_trace_path, std::ios::binary);
    if (!final_trace) {
        std::cerr << "Failed to create final trace file: " << final_trace_path << std::endl;
        return;
    }
    
    // Write kernel manager data
    kernel_manager_.serialize(final_trace);

    // Write number of operations
    size_t num_operations = trace_.operations.size();
    final_trace.write(reinterpret_cast<const char*>(&num_operations), sizeof(num_operations));
    std::cout << "Wrote " << num_operations << " operations" << std::endl;

    // Write all operations
    for (const auto& op : trace_.operations) {
        op->serialize(final_trace);
    }
    
    // Close files
    final_trace.close();
    trace_file_.close();
    
    // Replace original with final
    std::cout << "Replacing " << file_path << " with " << final_trace_path << std::endl;
    std::filesystem::rename(final_trace_path, file_path);
    
    std::cout << "\n\nTrace finalized successfully" << std::endl;
    // kernel_manager_ << std::cout;
    initialized_ = false;
}

void Tracer::recordKernelLaunch(const KernelExecution& exec) {
    if (!initialized_) return;
    
    auto exec_ptr = std::make_shared<KernelExecution>(exec);
    trace_.addOperation(std::move(exec_ptr));
    // exec.serialize(trace_file_);
    flush();
}

void Tracer::recordMemoryOperation(const MemoryOperation& op) {
    if (!initialized_) return;
    
    auto op_ptr = std::make_shared<MemoryOperation>(op);
    trace_.addOperation(std::move(op_ptr));
    // op.serialize(trace_file_);
    flush();
}

std::string Tracer::getTraceFilePath() const {
    static int trace_id = 0;
    
    // Check for environment variable override first
    const char* HIP_TRACE_LOCATION = getenv("HIP_TRACE_LOCATION");
    std::string base_dir;
    
    if (HIP_TRACE_LOCATION) {
        base_dir = HIP_TRACE_LOCATION;
    } else {
        // Use default location under HOME
        const char* home = getenv("HOME");
        if (!home) home = "/tmp";
        base_dir = std::string(home) + "/HipInterceptLayerTraces";
    }
    
    // Get binary name
    char self_path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", self_path, sizeof(self_path) - 1);
    if (len == -1) {
        return base_dir + "/unknown-" + std::to_string(trace_id++) + ".trace";
    }
    self_path[len] = '\0';
    
    std::string binary_name = std::filesystem::path(self_path).filename();
    
    // Skip system utilities
    static const std::vector<std::string> ignore_list = {
        "grep", "dash", "nm", "ld", "as", "objdump", "readelf", "addr2line"
    };
    
    for (const auto& ignored : ignore_list) {
        if (binary_name.find(ignored) != std::string::npos) {
            return ""; // Skip tracing
        }
    }
    
    // Create tracer directory
    std::cout << "Creating tracer directory: " << base_dir << std::endl;
    mkdir(base_dir.c_str(), 0755);
    
    // Find next available trace ID
    std::string base_path = base_dir + "/" + binary_name + "-";
    while (access((base_path + std::to_string(trace_id) + ".trace").c_str(), F_OK) != -1) {
        trace_id++;
    }
    
    auto trace_path = base_path + std::to_string(trace_id++) + ".trace";
    std::cout << "Trace file path: " << trace_path << std::endl;
    return trace_path;
}

void Tracer::flush() {
    if (!initialized_) return;
    trace_file_.flush();
}

Tracer::Tracer(const std::string& path) : file_path{path} {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open trace file: " + path);
    }

    // Read kernel manager data first
    kernel_manager_.deserialize(file);
    
    // Read number of operations
    size_t num_operations;
    file.read(reinterpret_cast<char*>(&num_operations), sizeof(num_operations));
    std::cout << "Read " << num_operations << " operations" << std::endl;

    // Read each operation
    for (size_t i = 0; i < num_operations; i++) {
        auto op = Operation::deserialize(file);
        trace_.addOperation(std::move(op));
    }

    std::cout << "Trace loaded from file: " << path << " with " << trace_.operations.size() << " operations" << std::endl;
    initialized_ = true;
}

std::shared_ptr<Operation> Operation::deserialize(std::ifstream& file) {
    auto start_pos = file.tellg();
    
    // Read operation type
    OperationType type;
    file.read(reinterpret_cast<char*>(&type), sizeof(type));
    
    std::cout << "Read operation type 0x" << std::hex << static_cast<uint32_t>(type) 
              << std::dec << " at position " << start_pos << std::endl;

    // Validate magic numbers
    if (type != OperationType::KERNEL && type != OperationType::MEMORY) {
        std::cerr << "Invalid operation type magic number: 0x" 
                  << std::hex << static_cast<uint32_t>(type) << std::dec << "\n"
                  << "Expected either:\n"
                  << "  KERNEL (0x" << std::hex << static_cast<uint32_t>(OperationType::KERNEL) << ")\n"
                  << "  MEMORY (0x" << std::hex << static_cast<uint32_t>(OperationType::MEMORY) << ")"
                  << std::dec << std::endl;
        std::abort();
    }

    switch(type) {
        case OperationType::KERNEL: 
            return KernelExecution::create_from_file(file);
        case OperationType::MEMORY: 
            return MemoryOperation::create_from_file(file);
        default: 
            // This should never happen due to the validation above
            std::abort();
    }
}