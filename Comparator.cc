#include "Comparator.hh"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <algorithm>

namespace {
    const char* RED = "\033[1;31m";
    const char* YELLOW = "\033[1;33m";
    const char* CYAN = "\033[1;36m";
    const char* RESET = "\033[0m";
}

// Helper function to get number of diffs to show
static int getNumDiffsToShow() {
    const char* env = std::getenv("HIL_NUM_DIFF");
    return env ? std::atoi(env) : 3;  // Default to 3 if not set
}

// Helper function to get operation name
static std::string getOperationName(MemoryOpType type) {
    switch (type) {
        case MemoryOpType::COPY: return "hipMemcpy";
        case MemoryOpType::COPY_ASYNC: return "hipMemcpyAsync";
        case MemoryOpType::SET: return "hipMemset";
        case MemoryOpType::ALLOC: return "hipMalloc";
        default: return "Unknown";
    }
}

// Helper function to get only diff flag
static bool getOnlyDiffFlag() {
    const char* env = std::getenv("HIL_ONLY_DIFF");
    return env && (std::string(env) == "1" || std::string(env) == "ON" || 
                  std::string(env) == "on" || std::string(env) == "true");
}

class ProgressBar {
public:
    ProgressBar(size_t total, size_t width = 50) 
        : total_(total), width_(width), 
          last_print_(std::chrono::steady_clock::now()) {}

    void update(size_t current) {
        if (total_ == 0) return;
        
        // Only update every 100ms to avoid too frequent updates
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_print_).count();
        if (elapsed < 100 && current != total_) return;
        
        float progress = static_cast<float>(current) / total_;
        int pos = static_cast<int>(width_ * progress);
        
        std::cout << "\r[";
        for (int i = 0; i < width_; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << "% "
                 << "(" << current << "/" << total_ << ")"
                 << std::flush;
        
        last_print_ = now;
        
        if (current == total_) {
            std::cout << std::endl;
        }
    }

private:
    size_t total_;
    size_t width_;
    std::chrono::steady_clock::time_point last_print_;
};

Comparator::Comparator(float epsilon) : epsilon_(epsilon) {}

bool Comparator::compareFloats(float a, float b) const {
    if (std::isnan(a) && std::isnan(b)) return true;
    if (std::isinf(a) && std::isinf(b)) return a == b;
    return std::abs(a - b) <= epsilon_;
}

ValueDifference Comparator::compareMemoryChanges(
    const std::vector<MemoryChange>& changes1,
    const std::vector<MemoryChange>& changes2) {
    
    ValueDifference diff;
    diff.matches = true;
    
    std::set<size_t> indices1, indices2;
    for (const auto& change : changes1) indices1.insert(change.element_index);
    for (const auto& change : changes2) indices2.insert(change.element_index);
    
    std::set_difference(indices1.begin(), indices1.end(),
                       indices2.begin(), indices2.end(),
                       std::back_inserter(diff.missing_indices));
    
    for (const auto& change1 : changes1) {
        auto it = std::lower_bound(changes2.begin(), changes2.end(), change1,
            [](const MemoryChange& a, const MemoryChange& b) {
                return a.element_index < b.element_index;
            });
        
        if (it != changes2.end() && it->element_index == change1.element_index) {
            if (!compareFloats(change1.pre_value, it->pre_value)) {
                diff.matches = false;
                diff.pre_value_mismatches.push_back({
                    change1.element_index,
                    change1.pre_value,
                    it->pre_value
                });
            }
            if (!compareFloats(change1.post_value, it->post_value)) {
                diff.matches = false;
                diff.post_value_mismatches.push_back({
                    change1.element_index,
                    change1.post_value,
                    it->post_value
                });
            }
        }
    }
    
    return diff;
}

ValueDifference compareMemoryStates(
    const std::vector<MemoryState>& states1,
    const std::vector<MemoryState>& states2,
    const std::vector<void*>& ptrs1,
    const std::vector<void*>& ptrs2) {
    
    ValueDifference diff;
    diff.matches = true;

    // First check if we have the same number of states
    if (states1.size() != states2.size()) {
        std::cerr << "ERROR: Number of states mismatch!" << std::endl;
        diff.matches = false;
        return diff;
    }

    // Compare each state pair
    for (size_t i = 0; i < states1.size(); i++) {
        const auto& state1 = states1[i];
        const auto& state2 = states2[i];

        // Abort if we find invalid states
        if (!state1.data || state1.size == 0) {
            std::cerr << "FATAL: Invalid state1 detected at index " << i << std::endl;
            std::abort();
        }
        if (!state2.data || state2.size == 0) {
            std::cerr << "FATAL: Invalid state2 detected at index " << i << std::endl;
            std::abort();
        }

        // Check if sizes match
        if (state1.size != state2.size) {
            std::cerr << "ERROR: Size mismatch at index " << i << std::endl;
            diff.matches = false;
            continue;
        }

        // Compare values
        for (size_t j = 0; j < state1.size; j += sizeof(float)) {
            float* val1 = (float*)(state1.data.get() + j);
            float* val2 = (float*)(state2.data.get() + j);

            if (*val1 != *val2) {
                diff.matches = false;
                diff.pre_value_mismatches.push_back({j/sizeof(float), *val1, *val2});
                diff.value_mismatches.push_back({j/sizeof(float), *val1, *val2});
            }
        }
    }

    return diff;
}

KernelComparisonResult compareKernelExecution(
    const KernelExecution& exec1,
    const KernelExecution& exec2) {
    
    KernelComparisonResult result;
    result.matches = true;
    result.kernel_name = exec1.kernel_name;
    result.execution_order = exec1.execution_order;

    // Abort if we detect missing states
    if (exec1.pre_state.empty() || exec1.post_state.empty()) {
        std::cerr << "FATAL: Missing states in exec1" << std::endl;
        std::abort();
    }
    if (exec2.pre_state.empty() || exec2.post_state.empty()) {
        std::cerr << "FATAL: Missing states in exec2" << std::endl;
        std::abort();
    }

    // Compare basic properties
    if (exec1.kernel_name != exec2.kernel_name ||
        exec1.grid_dim.x != exec2.grid_dim.x ||
        exec1.grid_dim.y != exec2.grid_dim.y ||
        exec1.grid_dim.z != exec2.grid_dim.z ||
        exec1.block_dim.x != exec2.block_dim.x ||
        exec1.block_dim.y != exec2.block_dim.y ||
        exec1.block_dim.z != exec2.block_dim.z ||
        exec1.shared_mem != exec2.shared_mem) {
        
        result.matches = false;
        result.differences.push_back("Kernel launch parameters differ");
        return result;
    }

    // Compare pre-states
    auto pre_diff = compareMemoryStates(
        exec1.pre_state, exec2.pre_state,
        exec1.arg_ptrs, exec2.arg_ptrs);
    
    if (!pre_diff.matches) {
        result.matches = false;
        result.differences.push_back("Pre-execution memory states differ");
        for (size_t i = 0; i < exec1.arg_ptrs.size(); i++) {
            result.value_differences[i].pre_value_mismatches = pre_diff.pre_value_mismatches;
        }
    }

    // Compare post-states
    auto post_diff = compareMemoryStates(
        exec1.post_state, exec2.post_state,
        exec1.arg_ptrs, exec2.arg_ptrs);
    
    if (!post_diff.matches) {
        result.matches = false;
        result.differences.push_back("Post-execution memory states differ");
        for (size_t i = 0; i < exec1.arg_ptrs.size(); i++) {
            result.value_differences[i].post_value_mismatches = post_diff.pre_value_mismatches;
        }
    }

    return result;
}

ComparisonResult Comparator::compare(const Trace& trace1, const Trace& trace2) {
    auto total_start = std::chrono::steady_clock::now();
    std::cout << "Starting trace comparison..." << std::endl;
    
    ComparisonResult result{false, SIZE_MAX, "", {}, trace1, trace2};
    result.traces_match = true;

    // Add back the TimelineEvent struct definition
    struct TimelineEvent {
        enum Type { KERNEL, MEMORY } type;
        size_t index;  // Index in original vector
        uint64_t execution_order;
        
        TimelineEvent(Type t, size_t i, uint64_t order) 
            : type(t), index(i), execution_order(order) {}

        bool operator<(const TimelineEvent& other) const {
            if (execution_order != other.execution_order) {
                return execution_order < other.execution_order;
            }
            // If execution orders are equal, maintain stable ordering by type
            return type < other.type;
        }
    };

    auto timeline_start = std::chrono::steady_clock::now();
    // Timeline creation
    std::vector<TimelineEvent> timeline1, timeline2;

    // Merge kernel executions and memory operations into timelines
    for (size_t i = 0; i < trace1.kernel_executions.size(); i++) {
        timeline1.emplace_back(TimelineEvent::KERNEL, i, 
                             trace1.kernel_executions[i].execution_order);
    }
    for (size_t i = 0; i < trace1.memory_operations.size(); i++) {
        timeline1.emplace_back(TimelineEvent::MEMORY, i, 
                             trace1.memory_operations[i].execution_order);
    }

    for (size_t i = 0; i < trace2.kernel_executions.size(); i++) {
        timeline2.emplace_back(TimelineEvent::KERNEL, i, 
                             trace2.kernel_executions[i].execution_order);
    }
    for (size_t i = 0; i < trace2.memory_operations.size(); i++) {
        timeline2.emplace_back(TimelineEvent::MEMORY, i, 
                             trace2.memory_operations[i].execution_order);
    }

    auto timeline_end = std::chrono::steady_clock::now();
    auto timeline_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        timeline_end - timeline_start).count();
    std::cout << "Timeline creation took " << timeline_duration << "ms" << std::endl;

    auto sort_start = std::chrono::steady_clock::now();
    // Sort both timelines by execution order and type
    std::sort(timeline1.begin(), timeline1.end());
    std::sort(timeline2.begin(), timeline2.end());

    auto sort_end = std::chrono::steady_clock::now();
    auto sort_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        sort_end - sort_start).count();
    std::cout << "Timeline sorting took " << sort_duration << "ms" << std::endl;

    // Calculate total events for progress bar
    size_t total_events = timeline1.size();
    std::cout << "Total events to compare: " << total_events << std::endl;
    ProgressBar progress(total_events);
    std::cout << "Comparing traces..." << std::endl;

    auto compare_start = std::chrono::steady_clock::now();
    
    size_t kernel_count = 0;
    size_t i1 = 0, i2 = 0;
    size_t events_processed = 0;
    
    while (i1 < timeline1.size() || i2 < timeline2.size()) {
        progress.update(events_processed++);

        if (i1 >= timeline1.size()) {
            result.traces_match = false;
            result.error_message += "First trace ended early at execution order " + 
                std::to_string(timeline2[i2].execution_order) + "\n";
            break;
        }
        if (i2 >= timeline2.size()) {
            result.traces_match = false;
            result.error_message += "Second trace ended early at execution order " + 
                std::to_string(timeline1[i1].execution_order) + "\n";
            break;
        }

        const auto& event1 = timeline1[i1];
        const auto& event2 = timeline2[i2];

        if (event1.type != event2.type) {
            result.traces_match = false;
            result.error_message += "Event type mismatch at execution order " + 
                std::to_string(event1.execution_order) + "\n";
            i1++; i2++;
            continue;
        }

        if (event1.type == TimelineEvent::KERNEL) {
            auto kernel_result = compareKernelExecution(
                trace1.kernel_executions[event1.index],
                trace2.kernel_executions[event2.index]
            );
            
            result.kernel_results.push_back(kernel_result);
            
            if (!kernel_result.matches && result.first_divergence_point == SIZE_MAX) {
                result.first_divergence_point = kernel_count;
            }
            kernel_count++;
        } else {
            const auto& op1 = trace1.memory_operations[event1.index];
            const auto& op2 = trace2.memory_operations[event2.index];
            
            if (op1.type != op2.type || op1.size != op2.size || 
                (op1.type != MemoryOpType::ALLOC && op1.kind != op2.kind)) {  // Only check kind for non-ALLOC ops
                result.traces_match = false;
                std::stringstream err;
                err << "Memory operation differs at order " << event1.execution_order << ":\n";
                if (op1.type != op2.type) {
                    err << "  - Type mismatch: " << op1.type << " vs " << op2.type << "\n";
                }
                if (op1.size != op2.size) {
                    err << "  - Size mismatch: " << op1.size << " vs " << op2.size 
                        << " bytes\n";
                }
                if (op1.type != MemoryOpType::ALLOC && op1.kind != op2.kind) {
                    err << "  - Kind mismatch: " << op1.kind << " vs " << op2.kind << "\n";
                }
                result.error_message += err.str();
            }
            
            if ((op1.pre_state && !op2.pre_state) || (!op1.pre_state && op2.pre_state)) {
                result.traces_match = false;
                result.error_message += "Memory operation differs in pre-state availability at order " + 
                    std::to_string(event1.execution_order) + "\n";
            }
            
            if (op1.pre_state && op2.pre_state && 
                (op1.pre_state->size != op2.pre_state->size ||
                 memcmp(op1.pre_state->data.get(), op2.pre_state->data.get(),
                        op1.pre_state->size) != 0)) {
                if (op1.pre_state->size != op2.pre_state->size) {
                    result.traces_match = false;
                    result.error_message += "Memory operation differs in pre-state at order " + 
                        std::to_string(event1.execution_order) + "\n";
                } else {
                    result.traces_match = false;
                    result.error_message += "Memory operation differs in pre-state at order " + 
                        std::to_string(event1.execution_order) + "\n";
                }
            }
        }

        i1++;
        i2++;
    }

    // Ensure progress bar shows 100%
    progress.update(total_events);
    
    auto total_end = std::chrono::steady_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        total_end - total_start).count();
    std::cout << "\nTotal comparison took " << total_duration << "ms" << std::endl;
    
    return result;
}

void Comparator::printComparisonResult(const ComparisonResult& result) {
    if (result.traces_match && !getOnlyDiffFlag()) {
        std::cout << "Traces match exactly!\n";
        return;
    }
    
    if (!result.traces_match) {
        std::cout << RED << "Traces differ:" << RESET << "\n";
    }
    
    // Create a timeline of differences
    struct DifferenceEvent {
        enum Type { KERNEL, MEMORY } type;
        size_t kernel_idx;  // Only valid for KERNEL type
        std::string message;
        uint64_t execution_order;
        
        DifferenceEvent(Type t, size_t idx, std::string msg, uint64_t order) 
            : type(t), kernel_idx(idx), message(std::move(msg)), execution_order(order) {}
            
        bool operator<(const DifferenceEvent& other) const {
            return execution_order < other.execution_order;
        }
    };
    
    std::vector<DifferenceEvent> differences;
    
    // Add kernel differences
    size_t kernel_idx = 0;
    for (const auto& kr : result.kernel_results) {

        if (!kr.matches || !getOnlyDiffFlag()) {
            std::stringstream ss;
            bool has_differences = !kr.matches;  // Start with existing match status
            
            // Get kernel configuration from the original executions
            const auto& exec1 = result.trace1.kernel_executions[kernel_idx];
            const auto& exec2 = result.trace2.kernel_executions[kernel_idx];
            
            // Check if configurations differ
            bool config_differs = (exec1.grid_dim.x != exec2.grid_dim.x || 
                                 exec1.grid_dim.y != exec2.grid_dim.y || 
                                 exec1.grid_dim.z != exec2.grid_dim.z || 
                                 exec1.block_dim.x != exec2.block_dim.x || 
                                 exec1.block_dim.y != exec2.block_dim.y || 
                                 exec1.block_dim.z != exec2.block_dim.z || 
                                 exec1.shared_mem != exec2.shared_mem);
            
            has_differences |= config_differs;  // Update difference status
            
            if (has_differences) {
                ss << RED;  // Start red color for differences
            }
            
            ss << "\nOp#" << kr.execution_order << ": Kernel(" << kr.kernel_name << ")";
            
            // Always show kernel configuration (using first trace's values)
            ss << "\n  Config: gridDim=(" << exec1.grid_dim.x << "," << exec1.grid_dim.y << "," << exec1.grid_dim.z 
               << "), blockDim=(" << exec1.block_dim.x << "," << exec1.block_dim.y << "," << exec1.block_dim.z 
               << "), shared=" << exec1.shared_mem;
            
            // If configurations differ, show both
            if (config_differs) {
                ss << "\n  vs: gridDim=(" << exec2.grid_dim.x << "," << exec2.grid_dim.y << "," << exec2.grid_dim.z 
                   << "), blockDim=(" << exec2.block_dim.x << "," << exec2.block_dim.y << "," << exec2.block_dim.z 
                   << "), shared=" << exec2.shared_mem;
            }
            
            // Show configuration differences if any
            if (!kr.differences.empty()) {
                ss << "\n  Config differences: " << kr.differences[0];
                for (size_t i = 1; i < kr.differences.size(); i++) {
                    ss << ", " << kr.differences[i];
                }
            }
            
            // Add argument differences on new lines
            for (const auto& [arg_idx, diff] : kr.value_differences) {
                if (!diff.pre_value_mismatches.empty() || 
                    !diff.post_value_mismatches.empty()) {
                    
                    ss << "\n  Arg " << arg_idx << ": ";
                    if (!diff.pre_value_mismatches.empty()) {
                        const auto& m = diff.pre_value_mismatches[0];
                        ss << diff.pre_value_mismatches.size() 
                           << " pre-exec diffs (first: idx " << m.index 
                           << ": " << YELLOW << std::setprecision(6) << m.value1 
                           << RESET << " vs " << CYAN << m.value2 << RESET << ")";
                    }
                    if (!diff.post_value_mismatches.empty()) {
                        if (!diff.pre_value_mismatches.empty()) ss << ", ";
                        const auto& m = diff.post_value_mismatches[0];
                        ss << diff.post_value_mismatches.size() 
                           << " post-exec diffs (first: idx " << m.index 
                           << ": " << YELLOW << std::setprecision(6) << m.value1 
                           << RESET << " vs " << CYAN << m.value2 << RESET << ")";
                    }
                }
            }
            
            if (!kr.matches) {
                ss << RESET;  // Reset color after differences
            }
            
            uint64_t exec_order = result.trace1.kernel_executions[kernel_idx].execution_order;
            differences.emplace_back(DifferenceEvent::KERNEL, kernel_idx, ss.str(), exec_order);
        }
        kernel_idx++;
    }
    
    // Add operation type counters
    std::unordered_map<MemoryOpType, size_t> op_counters;
    
    // Add memory operation differences
    size_t mem_op_idx = 0;
    for (size_t i = 0; i < result.trace1.memory_operations.size() && 
         i < result.trace2.memory_operations.size(); i++) {
        const auto& op1 = result.trace1.memory_operations[i];
        const auto& op2 = result.trace2.memory_operations[i];
        
        size_t type_count = ++op_counters[op1.type];
        
        // Only consider differences in type, size, and kind (for non-ALLOC operations)
        bool has_differences = (op1.type != op2.type || op1.size != op2.size || 
            (op1.type != MemoryOpType::ALLOC && op1.kind != op2.kind));
            
        // Check for memory state differences
        bool state_differences = op1.pre_state && op2.pre_state && 
                               (op1.pre_state->size != op2.pre_state->size ||
                                memcmp(op1.pre_state->data.get(), op2.pre_state->data.get(),
                                       op1.pre_state->size) != 0);

        if (has_differences || state_differences || !getOnlyDiffFlag()) {
            std::stringstream ss;
            if (has_differences || state_differences) {
                ss << RED;  // Start red color for differences
            }
            
            ss << "\nOp#" << op1.execution_order << " (" 
               << getOperationName(op1.type) << " call #" << type_count << "): "
               << "(size=" << op1.size;
            if (op1.type != MemoryOpType::ALLOC) {
                ss << ", kind=" << memcpyKindToString(op1.kind);
            }
            ss << ", stream=" << op1.stream << ")";

            // Show values if there's a memory state mismatch
            if (state_differences && op1.pre_state && op2.pre_state) {
                const float* data1 = reinterpret_cast<const float*>(op1.pre_state->data.get());
                const float* data2 = reinterpret_cast<const float*>(op2.pre_state->data.get());
                size_t num_floats = std::min(3UL, op1.pre_state->size / sizeof(float));
                
                if (num_floats > 0) {
                    ss << "  " << YELLOW << "[";  // Start yellow for first set
                    for (size_t i = 0; i < num_floats; i++) {
                        if (i > 0) ss << ", ";
                        ss << data1[i];
                    }
                    ss << "]" << RESET << " vs " << CYAN << "[";  // Switch to cyan for second set
                    for (size_t i = 0; i < num_floats; i++) {
                        if (i > 0) ss << ", ";
                        ss << data2[i];
                    }
                    ss << "]" << RESET;  // Reset color after values
                }
            }

            if (has_differences || state_differences) {
                ss << RESET;  // Reset color after differences
            }

            differences.emplace_back(DifferenceEvent::MEMORY, mem_op_idx, 
                ss.str(), op1.execution_order);
        }
        
        mem_op_idx++;
    }
    
    // Sort differences by execution order
    std::sort(differences.begin(), differences.end());
    
    // Print all differences in order
    for (const auto& diff : differences) {
        std::cout << diff.message;
    }
    std::cout << std::endl;
}

ComparisonResult Comparator::compare(const std::string& trace_path1, const std::string& trace_path2) {
    auto start = std::chrono::steady_clock::now();
    std::cout << "Loading traces..." << std::endl;
    
    try {
        // Load traces first
        auto load_start = std::chrono::steady_clock::now();
        Tracer t1(trace_path1);
        auto load_mid = std::chrono::steady_clock::now();
        Tracer t2(trace_path2);
        auto load_end = std::chrono::steady_clock::now();

        auto load1_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            load_mid - load_start).count();
        auto load2_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            load_end - load_mid).count();
        
        std::cout << "Loaded trace 1 in " << load1_duration << "ms" << std::endl;
        std::cout << "Loaded trace 2 in " << load2_duration << "ms" << std::endl;
        
        // Create result with loaded traces
        ComparisonResult result{false, SIZE_MAX, "", {}, std::move(t1.trace_), std::move(t2.trace_)};
        result.traces_match = true;
        
        // Compare the loaded traces
        auto timeline_result = compare(result.trace1, result.trace2);
        
        // Copy comparison results
        result.traces_match = timeline_result.traces_match;
        result.first_divergence_point = timeline_result.first_divergence_point;
        result.error_message = timeline_result.error_message;
        result.kernel_results = timeline_result.kernel_results;
        
        return result;
    } catch (const std::exception& e) {
        // Create an empty result with error message
        ComparisonResult result{false, SIZE_MAX, 
            std::string("Failed to load traces: ") + e.what(), 
            {}, Trace{}, Trace{}};
        return result;
    }
}

int Comparator::getArgumentIndex(void* ptr, const std::vector<void*>& arg_ptrs) const {
    for (size_t i = 0; i < arg_ptrs.size(); i++) {
        if (arg_ptrs[i] == ptr) return static_cast<int>(i);
    }
    return -1;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <trace1> <trace2>\n";
        return 1;
    }

    Comparator comparator;
    ComparisonResult result = comparator.compare(argv[1], argv[2]);
    comparator.printComparisonResult(result);
    return 0;
}