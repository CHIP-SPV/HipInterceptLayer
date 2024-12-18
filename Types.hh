#pragma once

#include <ostream>

namespace hip_intercept {

enum class MemoryOpType {
    COPY,
    COPY_ASYNC,
    SET,
    ALLOC
};

inline std::ostream& operator<<(std::ostream& os, const MemoryOpType& type) {
    switch (type) {
        case MemoryOpType::COPY:
            return os << "COPY";
        case MemoryOpType::COPY_ASYNC:
            return os << "COPY_ASYNC";
        case MemoryOpType::SET:
            return os << "SET";
        case MemoryOpType::ALLOC:
            return os << "ALLOC";
        default:
            return os << "UNKNOWN";
    }
}

} // namespace hip_intercept 