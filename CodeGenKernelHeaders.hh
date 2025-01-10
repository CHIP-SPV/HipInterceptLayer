#include <cstddef>
#include <fstream>
#include <iostream>
#include <hip/hip_runtime.h>
#include <iomanip>

inline float calculateChecksum(const char* data, std::size_t size) {
    float checksum = 0.0f;
    for (std::size_t i = 0; i < size; ++i) {
        checksum += static_cast<float>(data[i]);
    }
    return checksum;
}

inline bool loadTraceData(const char* filename, std::size_t offset, std::size_t size, void* dest) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open trace file: " << filename << std::endl;
        return false;
    }
    
    // First seek to the offset in the pre_state data
    file.seekg(offset);
    if (!file) {
        std::cerr << "Failed to seek to offset " << offset << std::endl;
        return false;
    }
    
    // Read the data directly into the destination
    file.read(static_cast<char*>(dest), size);
    if (!file) {
        std::cerr << "Failed to read " << size << " bytes at offset " << offset << std::endl;
        return false;
    }
    
    // Calculate and print the checksum
    float checksum = calculateChecksum(static_cast<char*>(dest), size);
    // std::cout << "Checksum: " << std::hex << std::setprecision(8) << checksum << std::dec << std::endl;
    return true;
}

// Add error checking macro
#define CHECK_HIP(cmd) \
    do { \
        hipError_t error = (cmd); \
        if (error != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return 1; \
        } \
    } while (0)