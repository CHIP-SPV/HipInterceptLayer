#ifndef HIP_INTERCEPT_LAYER_CODE_GEN_HH
#define HIP_INTERCEPT_LAYER_CODE_GEN_HH

#include "config.hh"
#include "KernelManager.hh"
#include "Tracer.hh"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_set>
#include <regex>

class CodeGen {
public:
  Tracer tracer;
  std::string trace_file_path_;
  CodeGen(const std::string &trace_file_path)
      : tracer(trace_file_path), operation_index_(-1), trace_file_path_(trace_file_path), kernel_manager_(tracer.getKernelManager()), kernel_lines_(0) {
        tracer.setSerializeTrace(false);
      }
  std::string generateReproducer(std::string kernel_name, int instance_index, bool debug_mode = false) {
    // Find all operations for the given kernel name and instance index
    std::vector<int> operation_indices;
    for (size_t i = 0; i < tracer.getNumOperations(); i++) {
      if (tracer.getOperation(i)->isKernel() &&
          static_cast<const KernelExecution &>(*tracer.getOperation(i))
                  .kernel_name == kernel_name) {
        operation_indices.push_back(i);
      }
    }
    if (operation_indices.empty()) {
      throw std::runtime_error("Kernel not found in trace");
    }
    return generateReproducer(operation_indices[instance_index], debug_mode);
  }

  // Generate complete reproducer code
  std::string generateReproducer(int operation_index, bool debug_mode = false) {
    auto op = tracer.getOperation(operation_index);
    if (!op->isKernel()) {
      throw std::runtime_error("Operation is not a kernel execution");
    }
    auto exec = static_cast<KernelExecution *>(op.get());

    operation_index_ = operation_index;
    std::stringstream ss;

    // Generate includes and main function header
    generateHeader(ss, exec, debug_mode);

    // Generate variable declarations for specific operation
    generateDeclarations(ss, exec, debug_mode);

    // Create data file and update trace file path
    std::string data_file = generateDataFile(exec);
    ss << "    const char* trace_file = \"" << data_file << "\";\n\n";

    // Generate initialization code for specific operation
    generateInitialization(ss, exec, debug_mode);

    // Generate single kernel launch
    generateKernelLaunches(ss, exec, debug_mode);

    // Generate cleanup code
    generateCleanup(ss, debug_mode);

    return ss.str();
  }

  // Generate and write the code to a file
  std::string generateFile(int operation_index,
                           const std::string &output_dir = "/tmp",
                           bool debug_mode = false) {
    std::filesystem::path dir_path(output_dir);
    std::filesystem::create_directories(dir_path);

    // Generate a unique filename using timestamp
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                         now.time_since_epoch())
                         .count();

    std::string filename =
        dir_path / ("kernel_replay_" + std::to_string(timestamp) + ".hip");

    std::ofstream file(filename);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to create output file: " + filename);
    }

    file << generateReproducer(operation_index, debug_mode);
    file.close();

    std::cout << "\n\nGenerated code:\n" << std::ifstream(filename).rdbuf() << std::endl;

    return filename;
  }

  // Compile the generated file using hipcc
  bool compileFile(const std::string &filename,
                   const std::string &output_dir = "/tmp") {
    std::filesystem::path output_path(output_dir);
    std::filesystem::path input_path(filename);
    std::string output_file = (output_path / input_path.stem()).string();

    // Construct the hipcc command
    std::stringstream cmd;
    cmd << HIPCC_PATH << " -O0 -g -w -o " << output_file << " " << filename;
    
    std::cout << "Executing: " << cmd.str() << std::endl;

    // Execute the command
    int result = std::system(cmd.str().c_str());

    if (result != 0) {
      std::cerr << "Compilation failed with error code: " << result
                << std::endl;
      return false;
    }

    return true;
  }

  // Convenience method to generate and compile in one step
  bool generateAndCompile(int operation_index,
                          const std::string &output_dir = "/tmp",
                          bool debug_mode = false) {
    try {
      std::string filename = generateFile(operation_index, output_dir, debug_mode);
      return compileFile(filename, output_dir);
    } catch (const std::exception &e) {
      std::cerr << "Error during generate and compile: " << e.what()
                << std::endl;
      return false;
    }
  }

  std::string generateDataFile(KernelExecution *op) {
    // Create a temporary file to store the raw argument data
    std::string data_file = trace_file_path_ + ".data";
    std::ofstream file(data_file, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to create data file: " + data_file);
    }

    // Write all argument data sequentially
    for (const auto& arg_state : op->pre_args) {
        file.write(arg_state.data.data(), arg_state.total_size());
    }

    file.close();
    return data_file;
  }

private:
  const KernelManager &kernel_manager_;
  std::unordered_set<std::string> declared_vars_;
  int operation_index_;
  size_t kernel_lines_;  // Number of lines in the kernel source for debug mode

  // Split kernel body into statements, properly handling strings and comments
  std::vector<std::string> splitStatements(const std::string& body) {
    std::vector<std::string> statements;
    size_t start = 0;
    bool in_string = false;
    bool in_comment = false;
    int brace_count = 0;
    int paren_count = 0;
    bool in_loop = false;  // Track if we're inside a loop statement
    
    for (size_t i = 0; i < body.length(); i++) {
      // Check if this might be a loop or control statement
      if (i == start || std::isspace(body[i-1])) {
        std::string rest = body.substr(i);
        if (rest.find("for") == 0 || rest.find("while") == 0 || rest.find("if") == 0) {
          in_loop = true;
        }
      }

      if (!in_comment && body[i] == '"' && (i == 0 || body[i-1] != '\\')) {
        in_string = !in_string;
      }
      else if (!in_string && !in_comment && i < body.length() - 1 && body[i] == '/' && body[i+1] == '/') {
        in_comment = true;
      }
      else if (in_comment && body[i] == '\n') {
        in_comment = false;
      }
      else if (!in_string && !in_comment && body[i] == '(') {
        paren_count++;
      }
      else if (!in_string && !in_comment && body[i] == ')') {
        paren_count--;
        if (paren_count == 0 && in_loop) {
          // We've found the end of a loop condition
          if (i + 1 < body.length() && body[i + 1] == '{') {
            // Loop with a block - keep going until we find the matching '}'
            continue;
          } else {
            // Loop with a single statement - keep going until we find ';'
            continue;
          }
        }
      }
      else if (!in_string && !in_comment && body[i] == '{') {
        brace_count++;
      }
      else if (!in_string && !in_comment && body[i] == '}') {
        brace_count--;
        if (brace_count == 0 && in_loop) {
          // We've found the end of a loop block
          // Find the next newline to include it in the statement
          size_t next_newline = i + 1;
          while (next_newline < body.length() && body[next_newline] != '\n') {
            next_newline++;
          }
          std::string stmt = body.substr(start, next_newline - start + 1);
          if (!stmt.empty()) {
            statements.push_back(stmt);
          }
          start = next_newline + 1;
          in_loop = false;
          i = next_newline;
          continue;
        }
      }
      else if (!in_string && !in_comment && body[i] == ';' && brace_count == 0 && paren_count == 0) {
        // Only split on semicolon if we're not inside a brace block or parentheses
        // Find the next newline to include it in the statement
        size_t next_newline = i + 1;
        while (next_newline < body.length() && body[next_newline] != '\n') {
          next_newline++;
        }
        std::string stmt = body.substr(start, next_newline - start + 1);
        if (!stmt.empty()) {
          if (in_loop) {
            // If we're in a loop, keep accumulating until we get the full loop
            continue;
          }
          statements.push_back(stmt);
        }
        if (!in_loop) {
          start = next_newline + 1;
          i = next_newline;
        }
      }
    }
    
    // Handle any remaining text
    if (start < body.length()) {
      std::string stmt = body.substr(start);
      if (!stmt.empty()) {
        statements.push_back(stmt);
      }
    }
    return statements;
  }

  void generateHeader(std::stringstream &ss,
                      KernelExecution *op,
                      bool debug_mode = false) {
    ss << "#include <hip/hip_runtime.h>\n"
       << "#include <iostream>\n"
       << "#include <cstring>\n"
       << "#include <fstream>\n"
       << "#include <iomanip>\n"
       << "#include <sstream>\n"
       << "#include <functional>\n\n";

    const Kernel &kernel = kernel_manager_.getKernelByName(op->kernel_name);
    std::string source = kernel.getModuleSource();

    // Insert the kernel source into the code or generate a placeholder kernel declaration
    if (!source.empty()) {
      // Remove all forms of extern "C" declarations
      std::regex extern_c_regex(R"(extern\s*"C"\s*(\{[^}]*\}|[^;{]*;))");
      source = std::regex_replace(source, extern_c_regex, "$1");
      // Also remove standalone extern "C"
      std::regex standalone_extern_c(R"(extern\s*"C"\s*)");
      source = std::regex_replace(source, standalone_extern_c, "");

      // remove all empty lines from the source
      // source = std::regex_replace(source, std::regex("\\n\s+?\\n"), "");

      std::cout << "Source: " << source << std::endl;

      if (debug_mode) {
        // Find where kernel starts, then first { from there and move backwards to find the parameter list end
        size_t kernel_start = source.find(kernel.getName());
        if (kernel_start != std::string::npos) {
          size_t first_brace = source.find('{', kernel_start);
          if (first_brace != std::string::npos) {
            // Search backwards from the opening brace to find the closing parenthesis
            size_t param_end = source.rfind(')', first_brace);
            if (param_end != std::string::npos) {
              // Find the opening parenthesis by counting backwards
              int paren_count = 1;
              size_t param_start = param_end;
              while (paren_count > 0 && param_start > kernel_start) {
                param_start--;
                if (source[param_start] == ')') paren_count++;
                if (source[param_start] == '(') paren_count--;
              }
              if (paren_count == 0) {
                // Add debug pointer parameter
                source = source.substr(0, param_end) + ", float* dbgPtr" + source.substr(param_end);
              }
            }
          }
          
          // Now inject debug statements after each line in the kernel body
          size_t body_start = first_brace + 1 + sizeof(", float* dbgPtr");
          size_t body_end = source.find_last_of('}');
          if (body_start != std::string::npos && body_end != std::string::npos) {
            std::string body = source.substr(body_start, body_end - body_start);
            std::stringstream modified_body;
            size_t line_number = 0;

            // Split the body into statements using splitStatements()
            auto statements = splitStatements(body);
            
            // Process each statement
            for (const auto& stmt : statements) {
              std::cout << "Processing statement: " << stmt << std::endl;
              // Check if this is an assignment statement
              std::regex assign_pattern(R"((\w+(?:\s*\[[^\]]*\])?)\s*=\s*([^;]+))");
              std::smatch matches;
              
              // First add the original statement
              modified_body << "    " << stmt << "\n";
              
              // Only add debug statement if this is a regular assignment (not in a loop/control statement)
              if (std::regex_search(stmt, matches, assign_pattern) &&
                  stmt.find("for") == std::string::npos && 
                  stmt.find("while") == std::string::npos &&
                  stmt.find("if") == std::string::npos) {
                std::string var_name = matches[1].str();
                // Trim the variable name
                var_name.erase(0, var_name.find_first_not_of(" \t"));
                var_name.erase(var_name.find_last_not_of(" \t") + 1);
                
                modified_body << "    dbgPtr[" << line_number << "] = " << var_name << ";\n\n";
                line_number++;
              }
            }
            
            // Replace the kernel body with the modified one
            source = source.substr(0, body_start) + "\n" + modified_body.str() + source.substr(body_end);
          }
        }
      }

      ss << source << "\n\n";
    } else {
      // Generate kernel declaration with empty body
      ss << "__global__ void " << kernel.getName() << "(";

      const auto &args = kernel.getArguments();
      for (size_t i = 0; i < args.size(); i++) {
        if (i > 0)
          ss << ", ";
        ss << args[i].getType();
        if (!args[i].getName().empty()) {
          ss << " " << args[i].getName();
        } else {
          ss << " arg" << i;
        }
      }

      ss << ") {\n"
         << "    // TODO: Original kernel source not available\n"
         << "    // This is a placeholder implementation\n"
         << "}\n\n";
    }

    // Copy the contents of CodeGenKernelHeaders.hh into the code
    std::ifstream headers_file(KERNEL_HEADERS_PATH);
    if (!headers_file.is_open()) {
        throw std::runtime_error("Failed to open CodeGenKernelHeaders.hh at path: " KERNEL_HEADERS_PATH);
    }
    ss << headers_file.rdbuf();
    ss << "\n\n";
    headers_file.close();

    ss << "int main() {\n"
       << "    hipError_t err;\n\n";

    if (debug_mode) {
      // Count number of lines in kernel source
      size_t line_count = std::count(kernel.getModuleSource().begin(), kernel.getModuleSource().end(), '\n') + 1;
      kernel_lines_ = line_count;
      
      ss << "    // Allocate debug pointer arrays\n";
      ss << "    float* dbgPtr_h = (float*)calloc(" << line_count << ", sizeof(float));\n";
      ss << "    if (!dbgPtr_h) { std::cerr << \"Failed to allocate host debug memory\\n\"; return 1; }\n";
      ss << "    float* dbgPtr_d;\n";
      ss << "    CHECK_HIP(hipMalloc((void**)&dbgPtr_d, " << line_count << " * sizeof(float)));\n";
      ss << "    CHECK_HIP(hipMemset(dbgPtr_d, 0, " << line_count << " * sizeof(float)));\n\n";
    }
  }

  void generateDeclarations(std::stringstream &ss,
                            KernelExecution *op,
                            bool debug_mode) {
    if (!op->isKernel()) {
      throw std::runtime_error("Operation is not a kernel execution");
    }
    const Kernel &kernel = kernel_manager_.getKernelByName(op->kernel_name);
    const auto &args = kernel.getArguments();

    for (size_t i = 0; i < args.size(); i++) {
      const auto &arg = args[i];
      std::string var_name = "arg_" + std::to_string(i) + "_" + op->kernel_name;

      if (declared_vars_.find(var_name) != declared_vars_.end()) {
        continue;
      }

      if (arg.isPointer()) {
        ss << "    " << arg.getType() << " " << var_name << "_h = nullptr;\n";
        ss << "    " << arg.getType() << " " << var_name << "_d = nullptr;\n";
        declared_vars_.insert(var_name + "_h");
        declared_vars_.insert(var_name + "_d");
      } else {
        ss << "    " << arg.getType() << " " << var_name << ";\n";
        declared_vars_.insert(var_name);
      }
    }
    // Remove debug pointer declarations since they're already declared in generateHeader
    ss << "\n";
  }

  void generateInitialization(std::stringstream &ss,
                              KernelExecution *op,
                              bool debug_mode) {
    if (!op->isKernel()) {
        throw std::runtime_error("Operation is not a kernel execution");
    }
    const Kernel &kernel = kernel_manager_.getKernelByName(op->kernel_name);
    const auto &args = kernel.getArguments();

    size_t pointer_arg_idx = 0;  // Track which pointer argument we're on
    size_t current_offset = 0;   // Track offset in data file

    ss << "    std::cout << \"\\nPRE-EXECUTION VALUES:\\n\";\n";
    
    // Add vector to store pre-execution checksums
    ss << "    std::vector<std::pair<size_t, uint64_t>> pre_checksums;\n";
    
    for (size_t i = 0; i < args.size(); i++) {
        const auto &arg = args[i];
        const auto &arg_state = op->pre_args[i];
        std::string var_name = "arg_" + std::to_string(i) + "_" + op->kernel_name;

        if (arg.isPointer()) {
            size_t arg_size = op->arg_sizes[pointer_arg_idx];
            if (arg_size == 0) {
                std::cerr << "Error: Zero size for pointer argument " << i << " of kernel " << op->kernel_name << std::endl;
                throw std::runtime_error("Invalid argument size");
            }
            
            ss << "    // Allocate and initialize " << var_name << "\n";
            ss << "    " << var_name << "_h = (" << arg.getType() << ")malloc(" << arg_size << ");\n";
            ss << "    if (!" << var_name << "_h) { std::cerr << \"Failed to allocate host memory\\n\"; return 1; }\n";
            
            ss << "    CHECK_HIP(hipMalloc((void**)&" << var_name << "_d, " << arg_size << "));\n";

            // Load data from data file
            ss << "    if (!loadTraceData(trace_file, " << current_offset << ", " << arg_state.total_size() 
               << ", (void*)" << var_name << "_h)) { return 1; }\n";
            ss << "    CHECK_HIP(hipMemcpy((void*)" << var_name << "_d, (const void*)" << var_name << "_h, "
               << arg_size << ", hipMemcpyHostToDevice));\n";
            
            // Store pre-execution checksum
            ss << "    pre_checksums.push_back({" << i << ", calculateChecksum((const char*)" << var_name << "_h, " << arg_size << ")});\n";
            
            // Print checksum
            ss << "    std::cout << \"  Arg " << i << " (" << arg.getType() << "): checksum = \" << "
               << "calculateChecksum((const char*)" << var_name << "_h, " << arg_size << ") << std::endl;\n\n";
            
            current_offset += arg_state.total_size();
            pointer_arg_idx++;
        } else {
            ss << "    if (!loadTraceData(trace_file, " << current_offset << ", " << arg_state.total_size()
               << ", (void*)&" << var_name << ")) { return 1; }\n";
            
            // Print value
            ss << "    std::cout << \"  Arg " << i << " (" << arg.getType() << "): \";\n";
            if (arg.getVectorSize() > 0) {
                std::string base_type = arg.getBaseType();
                bool is_integer = base_type.find("char") != std::string::npos || 
                                base_type.find("int") != std::string::npos ||
                                base_type.find("long") != std::string::npos;
                
                ss << "    std::cout << \"(\" << ";
                if (is_integer) {
                    ss << "static_cast<int>(" << var_name << ".x) << \", \" << "
                       << "static_cast<int>(" << var_name << ".y)";
                    if (arg.getVectorSize() >= 3) {
                        ss << " << \", \" << static_cast<int>(" << var_name << ".z)";
                    }
                    if (arg.getVectorSize() >= 4) {
                        ss << " << \", \" << static_cast<int>(" << var_name << ".w)";
                    }
                } else {
                    ss << var_name << ".x << \", \" << " 
                       << var_name << ".y";
                    if (arg.getVectorSize() >= 3) {
                        ss << " << \", \" << " << var_name << ".z";
                    }
                    if (arg.getVectorSize() >= 4) {
                        ss << " << \", \" << " << var_name << ".w";
                    }
                }
                ss << " << \")\" << std::endl;\n";
            } else {
                ss << "    std::cout << " << var_name << " << std::endl;\n";
            }
            
            current_offset += arg_state.total_size();
        }
    }
  }

  void generateKernelLaunches(std::stringstream &ss,
                              KernelExecution *op,
                              bool debug_mode) {
    const Kernel &kernel = kernel_manager_.getKernelByName(op->kernel_name);

    ss << "    // Launch kernel " << kernel.getName() << "\n";
    if (kernel.getModuleSource().empty()) {
        ss << "    std::cout << \"\nWARNING: Launching kernel with placeholder implementation - original source code not available\\n\";\n";
    }
    ss << "    dim3 grid(" << op->grid_dim.x << ", " << op->grid_dim.y << ", "
       << op->grid_dim.z << ");\n";
    ss << "    dim3 block(" << op->block_dim.x << ", " << op->block_dim.y
       << ", " << op->block_dim.z << ");\n";

    ss << "    " << kernel.getName() << "<<<grid, block, " << op->shared_mem
       << ">>>(";

    const auto &args = kernel.getArguments();
    for (size_t i = 0; i < args.size(); i++) {
        if (i > 0)
            ss << ", ";
        std::string var_name = "arg_" + std::to_string(i) + "_" + op->kernel_name;
        ss << (args[i].isPointer() ? var_name + "_d" : var_name);
    }

    if (debug_mode) {
      ss << ", dbgPtr_d";
    }

    ss << ");\n";
    ss << "    CHECK_HIP(hipDeviceSynchronize());\n"
       << "    CHECK_HIP(hipGetLastError());\n\n";
    
    ss << "    std::cout << \"\\nPOST-EXECUTION VALUES:\\n\";\n";
    
    // Add vector to store post-execution checksums
    ss << "    std::vector<std::pair<size_t, uint64_t>> post_checksums;\n";
    
    // Print all arguments after kernel execution
    size_t pointer_arg_idx = 0;
    for (size_t i = 0; i < args.size(); i++) {
        const auto &arg = args[i];
        std::string var_name = "arg_" + std::to_string(i) + "_" + op->kernel_name;
        
        if (arg.isPointer()) {
            size_t arg_size = op->arg_sizes[pointer_arg_idx++];
            
            ss << "    CHECK_HIP(hipMemcpy((void*)" << var_name << "_h, (const void*)" << var_name << "_d, "
               << arg_size << ", hipMemcpyDeviceToHost));\n";
            
            // Store post-execution checksum
            ss << "    post_checksums.push_back({" << i << ", calculateChecksum((const char*)" << var_name << "_h, " << arg_size << ")});\n";
            
            // Print checksum
            ss << "    std::cout << \"  Arg " << i << " (" << arg.getType() << "): checksum = \" << "
               << "calculateChecksum((const char*)" << var_name << "_h, " << arg_size << ") << std::endl;\n";
        } else {
            // Print value
            ss << "    std::cout << \"  Arg " << i << " (" << arg.getType() << "): \";\n";
            if (arg.getVectorSize() > 0) {
                std::string base_type = arg.getBaseType();
                bool is_integer = base_type.find("char") != std::string::npos || 
                                base_type.find("int") != std::string::npos ||
                                base_type.find("long") != std::string::npos;
                
                ss << "    std::cout << \"(\" << ";
                if (is_integer) {
                    ss << "static_cast<int>(" << var_name << ".x) << \", \" << "
                       << "static_cast<int>(" << var_name << ".y)";
                    if (arg.getVectorSize() >= 3) {
                        ss << " << \", \" << static_cast<int>(" << var_name << ".z)";
                    }
                    if (arg.getVectorSize() >= 4) {
                        ss << " << \", \" << static_cast<int>(" << var_name << ".w)";
                    }
                } else {
                    ss << var_name << ".x << \", \" << " 
                       << var_name << ".y";
                    if (arg.getVectorSize() >= 3) {
                        ss << " << \", \" << " << var_name << ".z";
                    }
                    if (arg.getVectorSize() >= 4) {
                        ss << " << \", \" << " << var_name << ".w";
                    }
                }
                ss << " << \")\" << std::endl;\n";
            } else {
                ss << "    std::cout << " << var_name << " << std::endl;\n";
            }
        }
    }
  }

  void generateCleanup(std::stringstream &ss, bool debug_mode) {
    ss << "\n    // Cleanup\n";
    ss << "\n    std::cout << \"\\nSUMMARY OF CHANGES:\\n\";\n";
    ss << "    bool any_changes = false;\n";
    ss << "    for (size_t i = 0; i < pre_checksums.size(); i++) {\n";
    ss << "        if (pre_checksums[i].second != post_checksums[i].second) {\n";
    ss << "            any_changes = true;\n";
    ss << "            std::cout << \"  Arg \" << pre_checksums[i].first << \": checksum changed from \" \n";
    ss << "                      << static_cast<int64_t>(pre_checksums[i].second) << \" to \" \n";
    ss << "                      << static_cast<int64_t>(post_checksums[i].second) << std::endl;\n";
    ss << "        }\n";
    ss << "    }\n";
    ss << "    if (!any_changes) {\n";
    ss << "        std::cout << \"  No changes detected in any pointer arguments\\n\";\n";
    ss << "    }\n\n";
    
    for (const auto &var : declared_vars_) {
        if (var.find("_d") != std::string::npos) {
            ss << "    if (" << var << ") hipFree((void*)" << var << ");\n";
        } else if (var.find("_h") != std::string::npos) {
            ss << "    if (" << var << ") free((void*)" << var << ");\n";
        }
    }

    if (debug_mode) {
      ss << "    // Copy back and print debug information\n";
      ss << "    CHECK_HIP(hipMemcpy(dbgPtr_h, dbgPtr_d, " << kernel_lines_ << " * sizeof(float), hipMemcpyDeviceToHost));\n";
      ss << "    std::cout << \"\\nDEBUG POINTER VALUES:\\n\";\n";
      ss << "    for (size_t i = 0; i < " << kernel_lines_ << "; i++) {\n";
      ss << "       std::cout << \"  Line \" << i << \": value = \" << dbgPtr_h[i] << \"\\n\";\n";
      ss << "    }\n";
      ss << "    if (dbgPtr_h) free(dbgPtr_h);\n";
      ss << "    if (dbgPtr_d) CHECK_HIP(hipFree(dbgPtr_d));\n";
    }

    ss << "\n    return 0;\n"
       << "}\n";
  }
};

#endif // HIP_INTERCEPT_LAYER_CODE_GEN_HH
