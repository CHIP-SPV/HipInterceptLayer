#pragma once

#include <string>
#include <map>
#include <set>
#include <stdexcept>
#include <cstddef>
#include <regex>
#include <vector>
#include <sstream>
#include <algorithm>
#include <hip/hip_runtime.h>

class TypeMap {
private:
    struct TypeInfo {
        size_t size;
        size_t array_size = 1;
        size_t alignment;
        std::string baseType;  // empty for direct types, contains target type for typedefs/defines
        bool is_struct;        // true if this is a struct type
        // member name, type, and array size (1 for non-array members)
        std::vector<std::tuple<std::string, std::string, size_t>> members;
    };
    
    std::map<std::string, TypeInfo> types;

public:
    TypeMap() {
        initializeBuiltinTypes();
    }

    // Register a type with its size
    void registerType(const std::string& typeName, size_t size, size_t alignment = 0, size_t array_size = 1) {
        if (typeName.empty()) {
            throw std::invalid_argument("Type name cannot be empty");
        }
        if (size == 0) {
            throw std::invalid_argument("Type size cannot be 0");
        }
        
        // fprintf(stderr, "Registering type: %s (size=%zu, align=%zu, array_size=%zu)\n", typeName.c_str(), size, alignment, array_size);
        TypeInfo info;
        info.size = size;
        info.alignment = alignment;
        info.array_size = array_size;
        info.baseType = "";
        types[typeName] = info;
    }
    
    // Register a typedef/define that points to another type
    void registerTypedef(const std::string& newType, const std::string& existingType) {
        if (newType.empty() || existingType.empty()) {
            throw std::invalid_argument("Type names cannot be empty");
        }
        
        // Check if the target type exists or is resolvable
        if (!isTypeRegistered(existingType)) {
            throw std::runtime_error("Target type '" + existingType + "' is not registered");
        }
        
        // fprintf(stderr, "Registering typedef/define: %s -> %s\n", newType.c_str(), existingType.c_str());
        auto targetType = types[resolveTypedef(existingType)];
        types[newType] = TypeInfo{0, targetType.array_size, 0, existingType};
    }
    
    // Get the size of a type
    size_t getTypeSize(const std::string& typeName) const {
        std::string resolvedType = resolveTypedef(typeName);
        
        auto it = types.find(resolvedType);
        if (it == types.end()) {
            throw std::runtime_error("Type '" + typeName + "' is not registered");
        }

        return it->second.size * it->second.array_size;
    }
    
    // Get the alignment of a type (returns 0 if not specified)
    size_t getAlignment(const std::string& typeName) const {
        std::string resolvedType = resolveTypedef(typeName);
        
        auto it = types.find(resolvedType);
        if (it == types.end()) {
            throw std::runtime_error("Type '" + typeName + "' is not registered");
        }
        
        return it->second.alignment;
    }

    // Get the array size of a type (returns 1 for non-array types)
    size_t getArraySize(const std::string& typeName) const {
        std::string resolvedType = resolveTypedef(typeName);
        
        auto it = types.find(resolvedType);
        if (it == types.end()) {
            throw std::runtime_error("Type '" + typeName + "' is not registered");
        }
        
        return it->second.array_size;
    }
    
    // Check if a type is registered
    bool isTypeRegistered(const std::string& typeName) const {
        try {
            std::string resolvedType = resolveTypedef(typeName);
            return types.find(resolvedType) != types.end();
        } catch (const std::runtime_error&) {
            return false;
        }
    }
    
    // Clear all registered types
    void clear() {
        types.clear();
        initializeBuiltinTypes();
    }

    /**
     * @brief Represents a typedef declaration during source parsing
     * 
     * This structure tracks both simple typedefs and array typedefs:
     * - For simple typedefs (e.g., 'typedef float real'):
     *   - newType: The new type name ('real')
     *   - targetType: The existing type being aliased ('float')
     *   - isArray: false
     * 
     * - For array typedefs (e.g., 'typedef real vector[4]'):
     *   - newType: The new array type name ('vector')
     *   - baseType: The element type ('real')
     *   - isArray: true
     *   - arraySize: Number of elements (4)
     */
    struct TypedefDeclaration {
        std::string newType;
        std::string targetType;
        bool isArray;
        size_t arraySize;
        std::string baseType;  // For array types, stores the element type
    };

    /**
     * @brief Parse source code to extract and register type information
     * 
     * This implementation uses a wave-based approach to handle complex typedef chains:
     * 1. First pass: Collect all typedef declarations and struct blocks without processing
     * 2. Process typedefs in waves until all are resolved:
     *    - Wave 1: Process typedefs that only depend on built-in/known types
     *    - Wave 2: Process typedefs that depend on types registered in wave 1
     *    - Continue until all typedefs are processed or no progress can be made
     * 3. Finally process struct blocks which may depend on the typedefs
     * 
     * Example of wave processing:
     * ```cpp
     * typedef float real;          // Wave 1: depends on built-in 'float'
     * typedef real real4[4];       // Wave 2: depends on 'real' from Wave 1
     * typedef real4 matrix[4];     // Wave 3: depends on 'real4' from Wave 2
     * ```
     * 
     * This approach correctly handles:
     * - Chains of typedefs where later definitions depend on earlier ones
     * - Array types where the base type is itself a typedef
     * - Structs that use previously defined typedefs
     * 
     * @param source The source code to parse
     * @throws std::runtime_error if circular dependencies are detected
     */
    void parseSource(const std::string& source) {
        std::vector<TypedefDeclaration> typedefs;
        std::vector<std::string> structBlocks;
        
        std::istringstream stream(source);
        std::string line;
        std::string current_block;
        bool in_struct = false;
        size_t brace_count = 0;

        while (std::getline(stream, line)) {
            size_t comment_pos = line.find("//");
            if (comment_pos != std::string::npos) {
                line = line.substr(0, comment_pos);
            }
            line = std::regex_replace(line, std::regex("^\\s+|\\s+$"), "");
            if (line.empty()) continue;

            brace_count += std::count(line.begin(), line.end(), '{');
            brace_count -= std::count(line.begin(), line.end(), '}');

            if (line.find("#define") != std::string::npos) {
                collectDefine(line);
            } else if (line.find("typedef") != std::string::npos) {
                if (line.find("struct") != std::string::npos) {
                    in_struct = true;
                    current_block = line + "\n";
                } else {
                    collectTypedef(line, typedefs);
                }
            } else if (in_struct) {
                current_block += line + "\n";
                if (brace_count == 0) {
                    structBlocks.push_back(current_block);
                    current_block.clear();
                    in_struct = false;
                }
            }
        }

        bool progress;
        do {
            progress = false;
            auto it = typedefs.begin();
            while (it != typedefs.end()) {
                if (it->isArray) {
                    if (isTypeRegistered(it->baseType)) {
                        size_t baseSize = getTypeSize(it->baseType);
                        registerType(it->newType, baseSize, 0, it->arraySize);
                        progress = true;
                        it = typedefs.erase(it);
                        continue;
                    }
                } else {
                    if (isTypeRegistered(it->targetType)) {
                        registerTypedef(it->newType, it->targetType);
                        progress = true;
                        it = typedefs.erase(it);
                        continue;
                    }
                }
                ++it;
            }
        } while (progress && !typedefs.empty());

        for (const auto& block : structBlocks) {
            parseTypedef(block);
        }

        if (!typedefs.empty()) {
            std::string unresolved;
            for (const auto& td : typedefs) {
                unresolved += td.newType + ", ";
            }
            throw std::runtime_error("Unresolved typedefs (possible circular dependencies): " + unresolved);
        }
    }

    void collectDefine(const std::string& define_str) {
        static std::regex define_regex(R"(#define\s+(\w+)\s+(\w+))");
        
        std::smatch match;
        if (std::regex_search(define_str, match, define_regex)) {
            std::string name = match[1].str();
            std::string value = match[2].str();
            
            if (isTypeRegistered(value)) {
                registerTypedef(name, value);
            }
        }
    }

    /**
     * @brief Parse and collect a single typedef declaration
     * 
     * Handles two forms of typedefs:
     * 1. Simple typedefs: typedef existing_type new_type;
     * 2. Array typedefs: typedef element_type new_type[size];
     * 
     * The collected declarations are stored for later processing in waves
     * to handle dependencies correctly.
     * 
     * @param typedef_str The typedef declaration string to parse
     * @param typedefs Vector to store the collected typedef declarations
     */
    void collectTypedef(const std::string& typedef_str, std::vector<TypedefDeclaration>& typedefs) {
        static std::regex simple_typedef_regex(R"(typedef\s+(\w+)\s+(\w+)\s*;)");
        static std::regex array_typedef_regex(R"(typedef\s+(\w+)\s+(\w+)\s*\[(\d+)\]\s*;)");

        std::smatch match;
        if (std::regex_search(typedef_str, match, array_typedef_regex)) {
            TypedefDeclaration decl;
            decl.baseType = match[1].str();
            decl.newType = match[2].str();
            decl.arraySize = std::stoul(match[3].str());
            decl.isArray = true;
            
            if (isTypeRegistered(decl.baseType)) {
                size_t baseSize = getTypeSize(decl.baseType);
                registerType(decl.newType, baseSize, 0, decl.arraySize);
            } else {
                typedefs.push_back(decl);
            }
        } else if (std::regex_search(typedef_str, match, simple_typedef_regex)) {
            TypedefDeclaration decl;
            decl.targetType = match[1].str();
            decl.newType = match[2].str();
            decl.isArray = false;
            typedefs.push_back(decl);
        }
    }

private:
    std::string resolveTypedef(const std::string& typeName) const {
        std::string current = typeName;
        std::set<std::string> visited;
        
        while (true) {
            auto it = types.find(current);
            if (it == types.end()) {
                return current;
            }
            
            if (it->second.baseType.empty()) {
                return current;
            }
            
            if (visited.find(current) != visited.end()) {
                throw std::runtime_error("Circular typedef detected for type '" + typeName + "'");
            }
            
            visited.insert(current);
            current = it->second.baseType;
        }
    }
    
    void initializeBuiltinTypes() {
        registerType("char", sizeof(char));
        registerType("short", sizeof(short));
        registerType("int", sizeof(int));
        registerType("long", sizeof(long));
        registerType("float", sizeof(float));
        registerType("double", sizeof(double));
        
        registerType("unsigned char", sizeof(unsigned char));
        registerType("unsigned short", sizeof(unsigned short));
        registerType("unsigned int", sizeof(unsigned int));
        registerType("unsigned long", sizeof(unsigned long));
        
        registerType("int8_t", sizeof(int8_t));
        registerType("uint8_t", sizeof(uint8_t));
        registerType("int16_t", sizeof(int16_t));
        registerType("uint16_t", sizeof(uint16_t));
        registerType("int32_t", sizeof(int32_t));
        registerType("uint32_t", sizeof(uint32_t));
        registerType("int64_t", sizeof(int64_t));
        registerType("uint64_t", sizeof(uint64_t));
        
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

    void parseTypedef(const std::string& typedef_str) {
        static std::regex struct_regex(R"(typedef\s+struct\s+(?:alignas\s*\((\d+)\))?\s*\{([^}]+)\}\s*(\w+)\s*;)");
        static std::regex simple_typedef_regex(R"(typedef\s+(\w+(?:\s*\[\d+\])?)\s+(\w+(?:\s*\[\d+\])?)\s*;)");

        std::smatch match;
        if (std::regex_search(typedef_str, match, struct_regex)) {
            size_t alignment = match[1].matched ? std::stoul(match[1].str()) : 0;
            std::string struct_body = match[2].str();
            std::string type_name = match[3].str();
            parseStruct(struct_body, type_name, alignment);
        } else if (std::regex_search(typedef_str, match, simple_typedef_regex)) {
            std::string existing_type = match[1].str();
            std::string new_type = match[2].str();

            std::regex array_regex(R"((\w+)\s*\[(\d+)\])");
            std::smatch array_match;
            if (std::regex_search(existing_type, array_match, array_regex)) {
                std::string base_type = array_match[1].str();
                size_t array_size = std::stoul(array_match[2].str());
                size_t base_size = getTypeSize(base_type);
                registerType(new_type, base_size, 0, array_size);
            } else {
                registerTypedef(new_type, existing_type);
            }
        }
    }

    void parseStruct(const std::string& struct_str, const std::string& name, size_t alignment) {
        auto members = parseStructMembers(struct_str);
        
        TypeInfo info;
        info.is_struct = true;
        info.members = members;
        info.alignment = alignment;
        info.baseType = "";
        info.array_size = 1;
        
        fprintf(stderr, "Parsing struct %s with alignment %zu\n", name.c_str(), alignment);
        
        info.size = calculateStructSize(info);
        
        fprintf(stderr, "Final struct %s size: %zu\n", name.c_str(), info.size);
        
        types[name] = info;
    }

    size_t calculateStructSize(const TypeInfo& structInfo) const {
        size_t current_offset = 0;
        size_t struct_alignment = structInfo.alignment > 0 ? structInfo.alignment : 1;

        for (const auto& member : structInfo.members) {
            const auto& [member_name, member_type, array_size] = member;
            
            // Get member's base size and natural alignment
            size_t base_size = getTypeSize(member_type);
            size_t member_alignment = getAlignment(member_type);
            
            // If member has no explicit alignment, use its size as natural alignment
            // Vector types (float4, int2, etc.) typically align to their size
            if (member_alignment == 0) {
                member_alignment = base_size >= 16 ? 16 : 
                                 base_size >= 8 ? 8 :
                                 base_size >= 4 ? 4 :
                                 base_size >= 2 ? 2 : 1;
            }

            // Align current offset to member's alignment requirement
            size_t padding = (member_alignment - (current_offset % member_alignment)) % member_alignment;
            current_offset += padding;

            size_t member_size = base_size * array_size;
            fprintf(stderr, "Member %s size: %zu (base=%zu, array_size=%zu)\n", 
                    member_name.c_str(), member_size, base_size, array_size);
            
            current_offset += member_size;
        }

        // Final alignment to struct's alignment requirement
        size_t remainder = current_offset % struct_alignment;
        if (remainder != 0) {
            fprintf(stderr, "Padding struct from %zu to %zu\n", current_offset, current_offset + struct_alignment - remainder);
            current_offset += struct_alignment - remainder;
        }
        
        return current_offset;
    }

    std::vector<std::tuple<std::string, std::string, size_t>> parseStructMembers(const std::string& struct_body) {
        std::vector<std::tuple<std::string, std::string, size_t>> members;
        std::istringstream stream(struct_body);
        std::string line;

        // Updated regex to better handle array declarations
        static std::regex member_regex(R"((\w+(?:\s*[*])?)\s+(\w+)(?:\[(\d+)\])?;)");
        
        while (std::getline(stream, line)) {
            size_t comment_pos = line.find("//");
            if (comment_pos != std::string::npos) {
                line = line.substr(0, comment_pos);
            }
            
            line = std::regex_replace(line, std::regex("^\\s+|\\s+$"), "");
            if (line.empty()) continue;

            std::smatch match;
            if (std::regex_search(line, match, member_regex)) {
                std::string type = match[1].str();
                std::string name = match[2].str();
                size_t array_size = match[3].matched ? std::stoul(match[3].str()) : 1;
                
                fprintf(stderr, "Parsing member: type=%s, name=%s, array_size=%zu\n", 
                        type.c_str(), name.c_str(), array_size);
                members.emplace_back(name, type, array_size);
            }
        }

        return members;
    }
};
