#pragma once

#include <string>
#include <map>
#include <set>
#include <stdexcept>
#include <cstddef>
#include <regex>
#include <vector>

class TypeMap {
public:
    TypeMap() {
        initializeBuiltinTypes();
    }

    // Register a type with its size
    void registerType(const std::string& typeName, size_t size, size_t alignment = 0);
    
    // Register a typedef that points to another type
    void registerTypedef(const std::string& newType, const std::string& existingType);
    
    // Get the size of a type
    size_t getTypeSize(const std::string& typeName) const;
    
    // Get the alignment of a type (returns 0 if not specified)
    size_t getAlignment(const std::string& typeName) const;
    
    // Check if a type is registered
    bool isTypeRegistered(const std::string& typeName) const;
    
    // Clear all registered types
    void clear();

    // Parse source code to extract type information
    void parseSource(const std::string& source);

private:
    struct TypeInfo {
        size_t size;
        size_t alignment;
        std::string baseType;  // empty for direct types, contains target type for typedefs
        bool is_struct;        // true if this is a struct type
        std::vector<std::pair<std::string, std::string>> members;  // member name and type pairs for structs
    };
    
    std::map<std::string, TypeInfo> types;
    
    // Helper to resolve typedefs
    std::string resolveTypedef(const std::string& typeName) const;
    
    // Initialize built-in types
    void initializeBuiltinTypes();

    // Source parsing helpers
    void parseTypedef(const std::string& typedef_str);
    void parseStruct(const std::string& struct_str, const std::string& name, size_t alignment);
    size_t calculateStructSize(const TypeInfo& structInfo) const;
    std::string extractAlignment(const std::string& struct_str, size_t& alignment);
    std::vector<std::pair<std::string, std::string>> parseStructMembers(const std::string& struct_body);
}; 