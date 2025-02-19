#ifndef HIP_INTERCEPT_LAYER_UTIL_HH
#define HIP_INTERCEPT_LAYER_UTIL_HH

#include "Types.hh"
#define __HIP_PLATFORM_SPIRV__
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <cxxabi.h>
#include <dlfcn.h>
#include <cstring>
#include <cerrno>
#include <cstdlib>
#include <cstddef>
#include <unistd.h>
#include <linux/limits.h>
#include <regex>
#include <memory>
#include <queue>
#include <set>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <memory>
#include <link.h>
#include "Tracer.hh"

//Forward declarations
struct dim3;

inline size_t countKernelArgs(void** args) {
    if (!args) return 0;
    
    size_t count = 0;
    while (args[count] != nullptr) {
        count++;
        // Safety check to prevent infinite loop
        if (count > 100) {  // reasonable max number of kernel arguments
            std::cerr << "Warning: Exceeded maximum expected kernel arguments\n";
            break;
        }
    }
    return count;
}

// Helper function to convert hipMemcpyKind to string
inline const char* memcpyKindToString(hipMemcpyKind kind) {
  switch(kind) {
    case hipMemcpyHostToHost: return "hipMemcpyHostToHost";
    case hipMemcpyHostToDevice: return "hipMemcpyHostToDevice"; 
    case hipMemcpyDeviceToHost: return "hipMemcpyDeviceToHost";
    case hipMemcpyDeviceToDevice: return "hipMemcpyDeviceToDevice";
    case hipMemcpyDefault: return "hipMemcpyDefault";
    default: return "Unknown";
  }
}

// Helper function to convert dim3 to string
inline std::string dim3ToString(dim3 d) {
  std::stringstream ss;
  ss << "{" << d.x << "," << d.y << "," << d.z << "}";
  return ss.str();
}

// Helper for hipDeviceProp_t
inline std::string devicePropsToString(const hipDeviceProp_t* props) {
  if (!props) return "null";
  std::stringstream ss;
  ss << "{name=" << props->name << ", totalGlobalMem=" << props->totalGlobalMem << "}";
  return ss.str();
}

#endif // HIP_INTERCEPT_LAYER_UTIL_HH