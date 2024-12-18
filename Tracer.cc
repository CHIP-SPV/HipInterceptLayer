#include "Tracer.hh"

namespace hip_intercept {

// MemoryState implementation
MemoryState::MemoryState(size_t s) : data(new char[s]), size(s) {}

MemoryState::MemoryState(const char* src, size_t s) : data(new char[s]), size(s) {
    memcpy(data.get(), src, s);
}

MemoryState::MemoryState() : size(0) {}

// Tracer implementation
Tracer& Tracer::instance() {
    static Tracer instance_;
    return instance_;
}

Tracer::Tracer() :
    initialized_(false)
    , current_execution_order_(0) {
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
    
    kernel_manager_.writeKernelManagerHeader(final_trace);
    
    // Serialize kernel manager
    kernel_manager_.serialize(final_trace);
    
    // Copy all events from original trace
    std::ifstream original_trace(trace_path_, std::ios::binary);
    if (!original_trace) {
        std::cerr << "Failed to open original trace for reading: " << trace_path_ << std::endl;
        final_trace.close();
        std::filesystem::remove(final_trace_path);
        return;
    }
    
    // Copy remaining data (events)
    std::vector<char> buffer(4096);  // Use a buffer for efficient copying
    while (original_trace) {
        original_trace.read(buffer.data(), buffer.size());
        std::streamsize bytes_read = original_trace.gcount();
        if (bytes_read > 0) {
            final_trace.write(buffer.data(), bytes_read);
        }
    }
    
    // Close files
    final_trace.close();
    original_trace.close();
    trace_file_.close();
    
    // Replace original with final
    std::filesystem::rename(final_trace_path, trace_path_);
    
    std::cout << "\n\nTnrace finalized successfully" << std::endl;
    kernel_manager_ << std::cout;
    initialized_ = false;
}

void Tracer::recordKernelLaunch(const KernelExecution& exec) {
    if (!initialized_) return;
    
    // Add debug assertions for pre/post states
    size_t total_pre_states = 0;
    size_t total_post_states = 0;
    
    // Count valid pre-states
    for (const auto& state : exec.pre_state) {
        if (state.data && state.size > 0) {
            total_pre_states++;
        }
    }
    
    // Count valid post-states
    for (const auto& state : exec.post_state) {
        if (state.data && state.size > 0) {
            total_post_states++;
        }
    }
    
    if (total_pre_states == 0 || total_post_states == 0) {
        std::cerr << "\nWARNING: Kernel '" << exec.kernel_name 
                  << "' has no memory states captured!\n"
                  << "Pre states: " << total_pre_states 
                  << ", Post states: " << total_post_states << "\n"
                  << "Number of arguments: " << exec.arg_ptrs.size() << "\n\n";
    }
    
    // Assign execution order
    KernelExecution ordered_exec = exec;
    ordered_exec.execution_order = current_execution_order_++;
    
    writeKernelExecution(ordered_exec);
    flush(); // Ensure the event is written immediately
}

void Tracer::recordMemoryOperation(const MemoryOperation& op) {
    if (!initialized_) return;
    
    // Assign execution order
    MemoryOperation ordered_op = op;
    ordered_op.execution_order = current_execution_order_++;
    
    writeMemoryOperation(ordered_op);
    flush(); // Ensure the event is written immediately
}

void Tracer::writeEvent(uint32_t type, const void* data, size_t size) {
    struct {
        uint32_t type;
        uint64_t timestamp;
        uint32_t size;
    } event_header = {
        type,
        static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()),
        static_cast<uint32_t>(size)
    };
    
    trace_file_.write(reinterpret_cast<char*>(&event_header), sizeof(event_header));
    trace_file_.write(reinterpret_cast<const char*>(data), size);
}

void Tracer::writeKernelExecution(const KernelExecution& exec) {
    // Add debug output for memory states
    std::cout << "\nWriting kernel execution for: " << exec.kernel_name << "\n";
    std::cout << "Pre-state entries: " << exec.pre_state.size() << "\n";
    std::cout << "Post-state entries: " << exec.post_state.size() << "\n";
    
    // Serialize kernel execution data
    struct {
        void* function_address;
        uint32_t name_length;
        dim3 grid_dim;
        dim3 block_dim;
        size_t shared_mem;
        hipStream_t stream;
        uint64_t execution_order;
        uint32_t num_args;
        uint32_t num_pre_states;
        uint32_t num_post_states;
        uint32_t num_changes;
    } kernel_data = {
        exec.function_address,
        static_cast<uint32_t>(exec.kernel_name.length()),
        exec.grid_dim,
        exec.block_dim,
        exec.shared_mem,
        exec.stream,
        exec.execution_order,
        static_cast<uint32_t>(exec.arg_ptrs.size()),
        static_cast<uint32_t>(exec.pre_state.size()),
        static_cast<uint32_t>(exec.post_state.size()),
        0  // Will count total changes below
    };
    
    // Count total changes
    for (const auto& [arg_idx, changes] : exec.changes_by_arg) {
        kernel_data.num_changes += changes.size();
    }
    
    writeEvent(1, &kernel_data, sizeof(kernel_data));
    
    // Write kernel name
    trace_file_.write(exec.kernel_name.c_str(), exec.kernel_name.length());
    
    // Write argument pointers
    for (void* ptr : exec.arg_ptrs) {
        trace_file_.write(reinterpret_cast<const char*>(&ptr), sizeof(void*));
    }

    // Write pre-states
    for (const auto& state : exec.pre_state) {
        // Write state size
        trace_file_.write(reinterpret_cast<const char*>(&state.size), sizeof(size_t));
        // Write state data
        if (state.data && state.size > 0) {
            trace_file_.write(state.data.get(), state.size);
        }
    }

    // Write post-states
    for (const auto& state : exec.post_state) {
        // Write state size
        trace_file_.write(reinterpret_cast<const char*>(&state.size), sizeof(size_t));
        // Write state data
        if (state.data && state.size > 0) {
            trace_file_.write(state.data.get(), state.size);
        }
    }
    
    // Write memory changes
    for (const auto& [arg_idx, changes] : exec.changes_by_arg) {
        for (const auto& [element_index, values] : changes) {
            struct {
                int arg_idx;
                size_t element_index;
                float pre_val;
                float post_val;
            } change_data = {
                arg_idx,
                element_index,
                values.first,
                values.second
            };
            trace_file_.write(reinterpret_cast<const char*>(&change_data), sizeof(change_data));
        }
    }
}

void Tracer::writeMemoryOperation(const MemoryOperation& op) {
    struct {
        MemoryOpType type;
        void* dst;
        const void* src;
        size_t size;
        int value;
        hipMemcpyKind kind;
        uint64_t execution_order;
        size_t pre_state_size;
        size_t post_state_size;
    } mem_op_data = {
        op.type,
        op.dst,
        op.src,
        op.size,
        op.value,
        op.kind,
        op.execution_order,
        op.pre_state ? op.pre_state->size : 0,
        op.post_state ? op.post_state->size : 0
    };
    
    writeEvent(2, &mem_op_data, sizeof(mem_op_data));
    
    // Write pre-state if exists
    if (op.pre_state && op.pre_state->data) {
        trace_file_.write(op.pre_state->data.get(), op.pre_state->size);
    }
    
    // Write post-state if exists
    if (op.post_state && op.post_state->data) {
        trace_file_.write(op.post_state->data.get(), op.post_state->size);
    }
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
            std::cerr << "Failed to open trace file: " << path << std::endl;
            throw std::runtime_error("Failed to open trace file: " + path);
        }
        
        kernel_manager_.readKernelManagerHeader(file);
        
        // Deserialize kernel manager
        try {
            kernel_manager_.deserialize(file);
            std::cout << "Kernel manager deserialized successfully with " 
                      << kernel_manager_.getNumKernels() << " kernels" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to deserialize kernel manager: " << e.what() << std::endl;
            throw;
        }
        
        // Read events until end of file
        while (file.good() && !file.eof()) {
            struct {
                uint32_t type;
                uint64_t timestamp;
                uint32_t size;
            } event_header;
            
            if (!file.read(reinterpret_cast<char*>(&event_header), sizeof(event_header))) {
                if (file.eof()) break;  // Normal end of file
                throw std::runtime_error("Failed to read event header");
            }
            
            switch (event_header.type) {
                case 1: // Kernel execution
                    instance().trace_.kernel_executions.push_back(readKernelExecution(file));
                    break;
                case 2: // Memory operation
                    instance().trace_.memory_operations.push_back(readMemoryOperation(file));
                    break;
                default:
                    std::cerr << "Unknown event type in trace file: " << event_header.type << std::endl;
                    throw std::runtime_error("Unknown event type in trace file");
            }
        }

        std::cout << "Trace loaded successfully with " 
                  << instance().trace_.kernel_executions.size() << " kernel executions and "
                  << instance().trace_.memory_operations.size() << " memory operations" << std::endl;


    std::cout << "Tracer initialized with kernels: " << kernel_manager_.getNumKernels() << std::endl;
}

KernelExecution Tracer::readKernelExecution(std::ifstream& file) {
    KernelExecution exec;
    
    struct {
        void* function_address;
        uint32_t name_length;
        dim3 grid_dim;
        dim3 block_dim;
        size_t shared_mem;
        hipStream_t stream;
        uint64_t execution_order;
        uint32_t num_args;
        uint32_t num_pre_states;
        uint32_t num_post_states;
        uint32_t num_changes;
    } kernel_data;
    
    file.read(reinterpret_cast<char*>(&kernel_data), sizeof(kernel_data));
    
    // Read kernel name
    std::vector<char> name_buffer(kernel_data.name_length + 1);
    file.read(name_buffer.data(), kernel_data.name_length);
    name_buffer[kernel_data.name_length] = '\0';
    
    exec.function_address = kernel_data.function_address;
    exec.kernel_name = name_buffer.data();
    exec.grid_dim = kernel_data.grid_dim;
    exec.block_dim = kernel_data.block_dim;
    exec.shared_mem = kernel_data.shared_mem;
    exec.stream = kernel_data.stream;
    exec.execution_order = kernel_data.execution_order;
    
    // Read argument pointers
    for (uint32_t i = 0; i < kernel_data.num_args; i++) {
        void* arg_ptr;
        file.read(reinterpret_cast<char*>(&arg_ptr), sizeof(void*));
        exec.arg_ptrs.push_back(arg_ptr);
    }

    // Read pre-states
    for (uint32_t i = 0; i < kernel_data.num_pre_states; i++) {
        size_t state_size;
        file.read(reinterpret_cast<char*>(&state_size), sizeof(size_t));
        
        MemoryState state(state_size);
        if (state_size > 0) {
            file.read(state.data.get(), state_size);
        }
        exec.pre_state.push_back(std::move(state));
    }

    // Read post-states
    for (uint32_t i = 0; i < kernel_data.num_post_states; i++) {
        size_t state_size;
        file.read(reinterpret_cast<char*>(&state_size), sizeof(size_t));
        
        MemoryState state(state_size);
        if (state_size > 0) {
            file.read(state.data.get(), state_size);
        }
        exec.post_state.push_back(std::move(state));
    }
    
    // Read memory changes
    for (uint32_t i = 0; i < kernel_data.num_changes; i++) {
        struct {
            int arg_idx;
            size_t element_index;
            float pre_val;
            float post_val;
        } change_data;
        
        file.read(reinterpret_cast<char*>(&change_data), sizeof(change_data));
        auto& changes = exec.changes_by_arg[change_data.arg_idx];
        changes.push_back(std::make_pair(
            change_data.element_index,
            std::make_pair(change_data.pre_val, change_data.post_val)
        ));
    }
    
    return exec;
}

MemoryOperation Tracer::readMemoryOperation(std::ifstream& file) {
    MemoryOperation op;
    
    struct {
        MemoryOpType type;
        void* dst;
        const void* src;
        size_t size;
        int value;
        hipMemcpyKind kind;
        uint64_t execution_order;
        size_t pre_state_size;
        size_t post_state_size;
    } mem_op_data;
    
    file.read(reinterpret_cast<char*>(&mem_op_data), sizeof(mem_op_data));
    
    op.type = mem_op_data.type;
    op.dst = mem_op_data.dst;
    op.src = mem_op_data.src;
    op.size = mem_op_data.size;
    op.value = mem_op_data.value;
    op.kind = mem_op_data.kind;
    op.execution_order = mem_op_data.execution_order;
    
    // Read pre-state if exists
    if (mem_op_data.pre_state_size > 0) {
        op.pre_state = std::make_shared<MemoryState>(mem_op_data.pre_state_size);
        file.read(op.pre_state->data.get(), mem_op_data.pre_state_size);
    }
    
    // Read post-state if exists
    if (mem_op_data.post_state_size > 0) {
        op.post_state = std::make_shared<MemoryState>(mem_op_data.post_state_size);
        file.read(op.post_state->data.get(), mem_op_data.post_state_size);
    }
    
    return op;
}

} // namespace hip_intercept
