#pragma once

namespace hip_intercept {

enum class MemoryOpType {
    COPY,
    COPY_ASYNC,
    SET
};

inline static const char* memOpTypeToString(MemoryOpType type) {
    switch (type) {
        case MemoryOpType::COPY: return "COPY";
        case MemoryOpType::COPY_ASYNC: return "COPY_ASYNC";
        case MemoryOpType::SET: return "SET";
        default: return "UNKNOWN";
    }
}

} // namespace hip_intercept 