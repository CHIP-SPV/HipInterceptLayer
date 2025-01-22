#include "TypeMap.hh"
#include <hip/hip_runtime.h>
#include <sstream>
#include <algorithm>

void TypeMap::registerType(const std::string& typeName, size_t size, size_t alignment) {
    if (typeName.empty()) {
        throw std::invalid_argument("Type name cannot be empty");
    }
    if (size == 0) {
        throw std::invalid_argument("Type size cannot be 0");
    }
    
    types[typeName] = TypeInfo{size, alignment, ""};
}

void TypeMap::registerTypedef(const std::string& newType, const std::string& existingType) {
    if (newType.empty() || existingType.empty()) {
        throw std::invalid_argument("Type names cannot be empty");
    }
    
    // Check if the target type exists or is resolvable
    if (!isTypeRegistered(existingType)) {
        throw std::runtime_error("Target type '" + existingType + "' is not registered");
    }
    
    types[newType] = TypeInfo{0, 0, existingType};
}

size_t TypeMap::getTypeSize(const std::string& typeName) const {
    std::string resolvedType = resolveTypedef(typeName);
    
    auto it = types.find(resolvedType);
    if (it == types.end()) {
        throw std::runtime_error("Type '" + typeName + "' is not registered");
    }
    
    return it->second.size;
}

size_t TypeMap::getAlignment(const std::string& typeName) const {
    std::string resolvedType = resolveTypedef(typeName);
    
    auto it = types.find(resolvedType);
    if (it == types.end()) {
        throw std::runtime_error("Type '" + typeName + "' is not registered");
    }
    
    return it->second.alignment;
}

bool TypeMap::isTypeRegistered(const std::string& typeName) const {
    try {
        std::string resolvedType = resolveTypedef(typeName);
        return types.find(resolvedType) != types.end();
    } catch (const std::runtime_error&) {
        return false;
    }
}

void TypeMap::clear() {
    types.clear();
    initializeBuiltinTypes();
}

std::string TypeMap::resolveTypedef(const std::string& typeName) const {
    std::string current = typeName;
    std::set<std::string> visited;
    
    while (true) {
        auto it = types.find(current);
        if (it == types.end()) {
            return current;  // Type not found, return as is
        }
        
        if (it->second.baseType.empty()) {
            return current;  // Found a concrete type
        }
        
        // Check for circular typedefs
        if (visited.find(current) != visited.end()) {
            throw std::runtime_error("Circular typedef detected for type '" + typeName + "'");
        }
        
        visited.insert(current);
        current = it->second.baseType;
    }
}

void TypeMap::initializeBuiltinTypes() {
    // Basic types
    registerType("char", sizeof(char));
    registerType("short", sizeof(short));
    registerType("int", sizeof(int));
    registerType("long", sizeof(long));
    registerType("float", sizeof(float));
    registerType("double", sizeof(double));
    
    // Unsigned variants
    registerType("unsigned char", sizeof(unsigned char));
    registerType("unsigned short", sizeof(unsigned short));
    registerType("unsigned int", sizeof(unsigned int));
    registerType("unsigned long", sizeof(unsigned long));
    
    // Fixed-width types
    registerType("int8_t", sizeof(int8_t));
    registerType("uint8_t", sizeof(uint8_t));
    registerType("int16_t", sizeof(int16_t));
    registerType("uint16_t", sizeof(uint16_t));
    registerType("int32_t", sizeof(int32_t));
    registerType("uint32_t", sizeof(uint32_t));
    registerType("int64_t", sizeof(int64_t));
    registerType("uint64_t", sizeof(uint64_t));
    
    // HIP vector types
    registerType("char2", sizeof(char2));
    registerType("char3", sizeof(char3));
    registerType("char4", sizeof(char4));
    registerType("uchar2", sizeof(uchar2));
    registerType("uchar3", sizeof(uchar3));
    registerType("uchar4", sizeof(uchar4));
    registerType("short2", sizeof(short2));
    registerType("short3", sizeof(short3));
    registerType("short4", sizeof(short4));
    registerType("ushort2", sizeof(ushort2));
    registerType("ushort3", sizeof(ushort3));
    registerType("ushort4", sizeof(ushort4));
    registerType("int2", sizeof(int2));
    registerType("int3", sizeof(int3));
    registerType("int4", sizeof(int4));
    registerType("uint2", sizeof(uint2));
    registerType("uint3", sizeof(uint3));
    registerType("uint4", sizeof(uint4));
    registerType("long2", sizeof(long2));
    registerType("long3", sizeof(long3));
    registerType("long4", sizeof(long4));
    registerType("ulong2", sizeof(ulong2));
    registerType("ulong3", sizeof(ulong3));
    registerType("ulong4", sizeof(ulong4));
    registerType("float2", sizeof(float2));
    registerType("float3", sizeof(float3));
    registerType("float4", sizeof(float4));
    registerType("double2", sizeof(double2));
    registerType("double3", sizeof(double3));
    registerType("double4", sizeof(double4));
}

void TypeMap::parseSource(const std::string& source) {
    std::istringstream stream(source);
    std::string line;
    std::string current_block;
    bool in_struct = false;
    size_t brace_count = 0;

    while (std::getline(stream, line)) {
        // Remove comments
        size_t comment_pos = line.find("//");
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }

        // Skip empty lines
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }

        // Count braces to track struct definitions
        brace_count += std::count(line.begin(), line.end(), '{');
        brace_count -= std::count(line.begin(), line.end(), '}');

        if (line.find("typedef") != std::string::npos) {
            if (line.find("struct") != std::string::npos) {
                in_struct = true;
            }
            current_block += line + "\n";
        } else if (in_struct) {
            current_block += line + "\n";
            if (brace_count == 0) {
                // End of struct definition
                parseTypedef(current_block);
                current_block.clear();
                in_struct = false;
            }
        } else if (line.find("typedef") != std::string::npos) {
            // Simple typedef
            parseTypedef(line);
        }
    }
}

void TypeMap::parseTypedef(const std::string& typedef_str) {
    static std::regex struct_regex(R"(typedef\s+struct\s+(?:alignas\s*\((\d+)\))?\s*\{([^}]+)\}\s*(\w+)\s*;)");
    static std::regex simple_typedef_regex(R"(typedef\s+([^;]+)\s+(\w+)\s*;)");

    std::smatch match;
    if (std::regex_search(typedef_str, match, struct_regex)) {
        // Struct typedef
        size_t alignment = match[1].matched ? std::stoul(match[1].str()) : 0;
        std::string struct_body = match[2].str();
        std::string type_name = match[3].str();
        parseStruct(struct_body, type_name, alignment);
    } else if (std::regex_search(typedef_str, match, simple_typedef_regex)) {
        // Simple typedef
        std::string existing_type = match[1].str();
        std::string new_type = match[2].str();
        registerTypedef(new_type, existing_type);
    }
}

void TypeMap::parseStruct(const std::string& struct_str, const std::string& name, size_t alignment) {
    auto members = parseStructMembers(struct_str);
    
    TypeInfo info;
    info.is_struct = true;
    info.members = members;
    info.alignment = alignment;
    info.baseType = "";
    
    // Calculate struct size based on members and alignment
    info.size = calculateStructSize(info);
    
    types[name] = info;
}

size_t TypeMap::calculateStructSize(const TypeInfo& structInfo) const {
    size_t total_size = 0;
    size_t max_align = structInfo.alignment > 0 ? structInfo.alignment : 1;

    for (const auto& member : structInfo.members) {
        const std::string& member_type = member.second;
        
        // Get member size and alignment
        size_t member_size = getTypeSize(member_type);
        size_t member_align = getAlignment(member_type);
        if (member_align == 0) member_align = member_size; // Default alignment
        
        // Update max alignment
        max_align = std::max(max_align, member_align);
        
        // Add padding for alignment
        total_size = (total_size + member_align - 1) & ~(member_align - 1);
        total_size += member_size;
    }

    // Final size must be multiple of max alignment
    total_size = (total_size + max_align - 1) & ~(max_align - 1);
    
    return total_size;
}

std::vector<std::pair<std::string, std::string>> TypeMap::parseStructMembers(const std::string& struct_body) {
    std::vector<std::pair<std::string, std::string>> members;
    std::istringstream stream(struct_body);
    std::string line;

    static std::regex member_regex(R"((\w+(?:\s*[*])?)\s+([^;]+);)");
    
    while (std::getline(stream, line)) {
        // Remove comments and leading/trailing whitespace
        size_t comment_pos = line.find("//");
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }
        
        line = std::regex_replace(line, std::regex("^\\s+|\\s+$"), "");
        if (line.empty()) continue;

        std::smatch match;
        if (std::regex_search(line, match, member_regex)) {
            std::string type = match[1].str();
            std::string names = match[2].str();
            
            // Split multiple declarations (e.g., "x, y, z")
            std::istringstream name_stream(names);
            std::string name;
            while (std::getline(name_stream, name, ',')) {
                name = std::regex_replace(name, std::regex("^\\s+|\\s+$"), "");
                members.emplace_back(name, type);
            }
        }
    }

    return members;
} 