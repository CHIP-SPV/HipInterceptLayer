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
      : tracer(trace_file_path), operation_index_(-1), trace_file_path_(trace_file_path), kernel_manager_(tracer.getKernelManager()) {
        tracer.setSerializeTrace(false);
      }
  std::string generateReproducer(std::string kernel_name, int instance_index) {
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
    return generateReproducer(operation_indices[instance_index]);
  }

  // Generate complete reproducer code
  std::string generateReproducer(int operation_index) {
    auto op = tracer.getOperation(operation_index);
    if (!op->isKernel()) {
      throw std::runtime_error("Operation is not a kernel execution");
    }
    auto exec = static_cast<KernelExecution *>(op.get());

    operation_index_ = operation_index;
    std::stringstream ss;

    // Generate includes and main function header
    generateHeader(ss, exec);

    // Generate variable declarations for specific operation
    generateDeclarations(ss, exec);

    // Create data file and update trace file path
    std::string data_file = generateDataFile(exec);
    ss << "    const char* trace_file = \"" << data_file << "\";\n\n";

    // Generate initialization code for specific operation
    generateInitialization(ss, exec);

    // Generate single kernel launch
    generateKernelLaunches(ss, exec);

    // Generate cleanup code
    generateCleanup(ss);

    return ss.str();
  }

  // Generate and write the code to a file
  std::string generateFile(int operation_index,
                           const std::string &output_dir = "/tmp") {
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

    file << generateReproducer(operation_index);
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
                          const std::string &output_dir = "/tmp") {
    try {
      std::string filename = generateFile(operation_index, output_dir);
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

  void generateHeader(std::stringstream &ss,
                      KernelExecution *op) {
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
  }

  void generateDeclarations(std::stringstream &ss,
                            KernelExecution *op) {
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
    ss << "\n";
  }

  void generateInitialization(std::stringstream &ss,
                              KernelExecution *op) {
    if (!op->isKernel()) {
        throw std::runtime_error("Operation is not a kernel execution");
    }
    const Kernel &kernel = kernel_manager_.getKernelByName(op->kernel_name);
    const auto &args = kernel.getArguments();

    size_t pointer_arg_idx = 0;  // Track which pointer argument we're on
    size_t current_offset = 0;   // Track offset in data file

    ss << "    std::cout << \"\\nPRE-EXECUTION VALUES:\\n\";\n";
    
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
                              KernelExecution *op) {
    const Kernel &kernel = kernel_manager_.getKernelByName(op->kernel_name);

    ss << "    // Launch kernel " << kernel.getName() << "\n";
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
    ss << ");\n";
    ss << "    CHECK_HIP(hipDeviceSynchronize());\n"
       << "    CHECK_HIP(hipGetLastError());\n\n";
    
    ss << "    std::cout << \"\\nPOST-EXECUTION VALUES:\\n\";\n";
    
    // Print all arguments after kernel execution
    size_t pointer_arg_idx = 0;
    for (size_t i = 0; i < args.size(); i++) {
        const auto &arg = args[i];
        std::string var_name = "arg_" + std::to_string(i) + "_" + op->kernel_name;
        
        if (arg.isPointer()) {
            size_t arg_size = op->arg_sizes[pointer_arg_idx++];
            
            ss << "    CHECK_HIP(hipMemcpy((void*)" << var_name << "_h, (const void*)" << var_name << "_d, "
               << arg_size << ", hipMemcpyDeviceToHost));\n";
            
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

  void generateCleanup(std::stringstream &ss) {
    ss << "\n    // Cleanup\n";
    for (const auto &var : declared_vars_) {
        if (var.find("_d") != std::string::npos) {
            ss << "    if (" << var << ") hipFree((void*)" << var << ");\n";
        } else if (var.find("_h") != std::string::npos) {
            ss << "    if (" << var << ") free((void*)" << var << ");\n";
        }
    }

    ss << "\n    return 0;\n"
       << "}\n";
  }
};

#endif // HIP_INTERCEPT_LAYER_CODE_GEN_HH
