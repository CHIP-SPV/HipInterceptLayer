#include <gtest/gtest.h>
#include "../TypeMap.hh"

// Test fixture for TypeMap tests
class TypeMapTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test basic built-in type registration and querying
TEST_F(TypeMapTest, BasicBuiltinTypes) {
    TypeMap typeMap;
    
    // Test built-in types that should be pre-registered
    EXPECT_EQ(typeMap.getTypeSize("int"), sizeof(int));
    EXPECT_EQ(typeMap.getTypeSize("float"), sizeof(float));
    EXPECT_EQ(typeMap.getTypeSize("double"), sizeof(double));
    EXPECT_EQ(typeMap.getTypeSize("char"), sizeof(char));
}

// Test custom type registration and querying
TEST_F(TypeMapTest, CustomTypeRegistration) {
    TypeMap typeMap;
    
    // Register a custom type
    typeMap.registerType("MyType", 16, 8);  // 16 bytes with 8-byte alignment
    
    // Verify registration
    EXPECT_TRUE(typeMap.isTypeRegistered("MyType"));
    EXPECT_EQ(typeMap.getTypeSize("MyType"), 16);
    EXPECT_EQ(typeMap.getAlignment("MyType"), 8);
}

// Test error handling for invalid operations
TEST_F(TypeMapTest, ErrorHandling) {
    TypeMap typeMap;
    
    // Test invalid type registration
    EXPECT_THROW(typeMap.registerType("", 8), std::invalid_argument);  // Empty name
    EXPECT_THROW(typeMap.registerType("InvalidSize", 0), std::invalid_argument);  // Zero size
    
    // Test querying non-existent type
    EXPECT_FALSE(typeMap.isTypeRegistered("NonExistentType"));
    EXPECT_THROW(typeMap.getTypeSize("NonExistentType"), std::runtime_error);
    EXPECT_THROW(typeMap.getAlignment("NonExistentType"), std::runtime_error);
}

// Test simple typedef handling
TEST_F(TypeMapTest, SimpleTypedef) {
    TypeMap typeMap;
    
    // Create a simple typedef
    typeMap.registerType("BaseType", 8, 4);
    typeMap.registerTypedef("AliasType", "BaseType");
    
    // Verify typedef resolution
    EXPECT_TRUE(typeMap.isTypeRegistered("AliasType"));
    EXPECT_EQ(typeMap.getTypeSize("AliasType"), 8);
    EXPECT_EQ(typeMap.getAlignment("AliasType"), 4);
}

// Test parsing of simple type definitions
TEST_F(TypeMapTest, SimpleTypeParsing) {
    TypeMap typeMap;
    
    // Test parsing a simple typedef
    std::string source = "typedef int MyInt;";
    typeMap.parseSource(source);
    
    EXPECT_TRUE(typeMap.isTypeRegistered("MyInt"));
    EXPECT_EQ(typeMap.getTypeSize("MyInt"), sizeof(int));
} 