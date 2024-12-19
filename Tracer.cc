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
    
    trace_path_ = getTraceFilePath();
    if (trace_path_.empty()) return; // Skip tracing for this process
    
    trace_file_.open(trace_path_, std::ios::binary);
    if (!trace_file_) {
        std::cerr << "Failed to open trace file: " << trace_path_ << std::endl;
        return;
    }
    
    std::cout << "\n=== HIP Trace File ===\n"
              << "Writing trace to: " << trace_path_ << "\n"
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
    std::string final_trace_path = trace_path_ + ".final";
    std::ofstream final_trace(final_trace_path, std::ios::binary);
    if (!final_trace) {
        std::cerr << "Failed to create final trace file: " << final_trace_path << std::endl;
        return;
    }
    
    // Write kernel manager data
    kernel_manager_.serialize(final_trace);
    
    // Write all operations
    for (const auto& op : trace_.operations) {
        op->serialize(final_trace);
    }
    
    // Close files
    final_trace.close();
    trace_file_.close();
    
    // Replace original with final
    std::filesystem::rename(final_trace_path, trace_path_);
    
    std::cout << "\n\nTrace finalized successfully" << std::endl;
    kernel_manager_ << std::cout;
    initialized_ = false;
}

void Tracer::recordKernelLaunch(const KernelExecution& exec) {
    if (!initialized_) return;
    
    // Create a unique_ptr with a copy of exec
    auto exec_ptr = std::make_unique<KernelExecution>(exec);
    trace_.addOperation(std::move(exec_ptr));
    exec.serialize(trace_file_);
    flush();
}

void Tracer::recordMemoryOperation(const MemoryOperation& op) {
    if (!initialized_) return;
    
    // Create a unique_ptr with a copy of op
    auto op_ptr = std::make_unique<MemoryOperation>(op);
    trace_.addOperation(std::move(op_ptr));
    op.serialize(trace_file_);
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

Tracer::Tracer(const std::string& path) {
    trace_path_ = path;
    std::cout << "Loading trace from: " << path << std::endl;
    
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open trace file: " + path);
    }
    
    // Deserialize kernel manager
    kernel_manager_.deserialize(file);
    
    // Read operations until end of file
    while (file.good() && !file.eof()) {
        try {
            auto op = Operation::deserialize(file);
            if (op) {
                trace_.operations.push_back(std::move(op));
            }
        } catch (const std::exception& e) {
            if (!file.eof()) {
                std::cerr << "Error reading operation: " << e.what() << std::endl;
                throw;
            }
            break;
        }
    }
    
    std::cout << "Trace loaded successfully with " 
              << trace_.operations.size() << " operations" << std::endl;
}
