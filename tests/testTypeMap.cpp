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
    
    // Test parsing a chain of typedefs that resolve to a known type
    std::string source = R"(
        typedef int MyInt;           // Direct mapping to known type
        typedef MyInt MyInt2;        // One level of indirection
        typedef MyInt2 MyInt3;       // Two levels of indirection
        typedef float real;          // Common scientific computing typedef
        typedef real real4[4];       // Array typedef
        typedef real4 matrix[4];     // Nested array typedef
    )";
    
    typeMap.parseSource(source);
    
    // Verify direct typedef to known type
    EXPECT_TRUE(typeMap.isTypeRegistered("MyInt"));
    EXPECT_EQ(typeMap.getTypeSize("MyInt"), sizeof(int));
    
    // Verify chain of typedefs
    EXPECT_TRUE(typeMap.isTypeRegistered("MyInt2"));
    EXPECT_EQ(typeMap.getTypeSize("MyInt2"), sizeof(int));
    EXPECT_TRUE(typeMap.isTypeRegistered("MyInt3"));
    EXPECT_EQ(typeMap.getTypeSize("MyInt3"), sizeof(int));
    
    // Verify scientific computing typedefs
    EXPECT_TRUE(typeMap.isTypeRegistered("real"));
    EXPECT_EQ(typeMap.getTypeSize("real"), sizeof(float));
    EXPECT_TRUE(typeMap.isTypeRegistered("real4"));
    // real4 is an array of 4 floats, sizeof(float) = 4, so 4*4 = 16
    EXPECT_EQ(typeMap.getTypeSize("real4"), 16);
    EXPECT_TRUE(typeMap.isTypeRegistered("matrix"));
    // matrix is an array of 4 float4s, sizeof(float4) = 16, so 16*4 = 64
    EXPECT_EQ(typeMap.getTypeSize("matrix"), 64);
}

// Test parsing of complex type definitions
TEST_F(TypeMapTest, ComplexTypeParsing) {
    TypeMap typeMap;
    
    // Test parsing complex type definitions that mix known types
    std::string source = R"(
        #define real float                  // Base scientific type
        #define real4 float4 
        typedef float myType;               // First define myType as float
        typedef real4 vector;               // Then use real4 (which expands to float4)
        typedef vector positions[1000];      // Array of vectors
        typedef struct {
            vector pos;                      // float4 position
            float mass;                      // Known type
            int2 indices;                    // HIP's int2
        } Particle;
    )";
    
    typeMap.parseSource(source);

    EXPECT_TRUE(typeMap.isTypeRegistered("myType"));
    EXPECT_EQ(typeMap.getTypeSize("myType"), sizeof(float));
    
    // Verify resolution to known types
    EXPECT_TRUE(typeMap.isTypeRegistered("real"));
    EXPECT_EQ(typeMap.getTypeSize("real"), sizeof(float));
    
    // Verify vector typedef resolves to float4
    EXPECT_TRUE(typeMap.isTypeRegistered("vector"));
    EXPECT_EQ(typeMap.getTypeSize("vector"), sizeof(float4));
    
    // Verify array type
    EXPECT_TRUE(typeMap.isTypeRegistered("positions"));
    EXPECT_EQ(typeMap.getTypeSize("positions"), sizeof(float4) * 1000);
    
    // Verify struct with mixed types
    EXPECT_TRUE(typeMap.isTypeRegistered("Particle"));
    struct TestParticle {
        float4 pos;
        float mass;
        int2 indices;
    };
    EXPECT_EQ(typeMap.getTypeSize("Particle"), sizeof(TestParticle));
}

// Test HIP vector types
TEST_F(TypeMapTest, HIPVectorTypes) {
    TypeMap typeMap;
    
    // Test float vector types
    EXPECT_EQ(typeMap.getTypeSize("float2"), sizeof(float2));
    EXPECT_EQ(typeMap.getTypeSize("float3"), sizeof(float3));
    EXPECT_EQ(typeMap.getTypeSize("float4"), sizeof(float4));
    
    // Test int vector types
    EXPECT_EQ(typeMap.getTypeSize("int2"), sizeof(int2));
    EXPECT_EQ(typeMap.getTypeSize("int3"), sizeof(int3));
    EXPECT_EQ(typeMap.getTypeSize("int4"), sizeof(int4));
    
    // Test double vector types
    EXPECT_EQ(typeMap.getTypeSize("double2"), sizeof(double2));
    EXPECT_EQ(typeMap.getTypeSize("double3"), sizeof(double3));
    EXPECT_EQ(typeMap.getTypeSize("double4"), sizeof(double4));
}

// Test struct parsing and size calculation
TEST_F(TypeMapTest, StructParsing) {
    TypeMap typeMap;
    
    std::string source = R"(
        typedef struct {
            float x;
            float y;
            float z;
            float w;
        } Vector4;

        typedef struct alignas(16) {
            double position[3];
            float mass;
            int id;
            int tid;
            double *ptr;
            Vector4 *whatever;
        } Particle;

        typedef struct alignas(16) {
            Particle particles[1000];
        } ParticleArray;
    )";
    
    typeMap.parseSource(source);
    
    // Verify Vector4 struct
    EXPECT_TRUE(typeMap.isTypeRegistered("Vector4"));
    EXPECT_EQ(typeMap.getTypeSize("Vector4"), 4 * sizeof(float));
    
    // Verify Particle struct with explicit alignment
    EXPECT_TRUE(typeMap.isTypeRegistered("Particle"));
    EXPECT_EQ(typeMap.getAlignment("Particle"), 16);
    // Size should account for alignment and padding
    // 3 doubles (24 bytes) + 1 float (4 bytes) + 1 int (4 bytes) + 1 int (4 bytes) = 36 bytes
    // 36 bytes is not a multiple of 16, so we need to pad to the next multiple of 16 which is 48
    size_t expected_size = 48;
    size_t actual_size = typeMap.getTypeSize("Particle");
    fprintf(stderr, "Particle size: expected=%zu, actual=%zu\n", expected_size, actual_size);
    EXPECT_EQ(actual_size, expected_size);

    // Verify ParticleArray struct with explicit alignment
    EXPECT_TRUE(typeMap.isTypeRegistered("ParticleArray"));
    EXPECT_EQ(typeMap.getAlignment("ParticleArray"), 16);
    // Size should account for alignment and padding
    expected_size = 1000 * 48;
    actual_size = typeMap.getTypeSize("ParticleArray");
    fprintf(stderr, "ParticleArray size: expected=%zu, actual=%zu\n", expected_size, actual_size);
    EXPECT_EQ(actual_size, expected_size);

    // Verify ParticleArray with pointer to Particle
    EXPECT_TRUE(typeMap.isTypeRegistered("ptr"));
    EXPECT_EQ(typeMap.getTypeSize("ptr"), sizeof(double*));

    typedef struct {
            float x;
            float y;
            float z;
            float w;
        } Vector4;

    // Verify Vector4 type is registered and pointer member has correct size
    EXPECT_TRUE(typeMap.isTypeRegistered("Vector4"));
    EXPECT_EQ(typeMap.getTypeSize("ptr"), sizeof(Vector4*));
}

// Test type resolution with qualifiers
TEST_F(TypeMapTest, QualifierHandling) {
    TypeMap typeMap;
    
    std::string source = R"(
        typedef const int ConstInt;
        typedef volatile float VolatileFloat;
        typedef const volatile double ConstVolatileDouble;
        typedef int* restrict RestrictIntPtr;
    )";
    
    typeMap.parseSource(source);
    
    // Verify that qualifiers don't affect size/alignment
    EXPECT_EQ(typeMap.getTypeSize("ConstInt"), sizeof(int));
    EXPECT_EQ(typeMap.getTypeSize("VolatileFloat"), sizeof(float));
    EXPECT_EQ(typeMap.getTypeSize("ConstVolatileDouble"), sizeof(double));
    EXPECT_EQ(typeMap.getTypeSize("RestrictIntPtr"), sizeof(int*));
}

// Test complex struct alignment and padding
TEST_F(TypeMapTest, ComplexStructAlignment) {
    TypeMap typeMap;
    
    std::string source = R"(
        typedef struct {
            char a;     // 1 byte
            double b;   // 8 bytes, needs alignment
            short c;    // 2 bytes
            int d;      // 4 bytes
        } MixedStruct;

        typedef struct alignas(32) {
            float vec[3];    // 12 bytes
            char flag;       // 1 byte
            double value;    // 8 bytes
        } AlignedStruct;
    )";
    
    typeMap.parseSource(source);
    
    // Verify MixedStruct
    EXPECT_TRUE(typeMap.isTypeRegistered("MixedStruct"));
    // Size should account for natural alignment and padding
    size_t expected_mixed_size = ((1 + 7) & ~7) + 8 + ((2 + 3) & ~3) + 4;
    expected_mixed_size = (expected_mixed_size + 7) & ~7;  // Final alignment
    EXPECT_EQ(typeMap.getTypeSize("MixedStruct"), expected_mixed_size);
    
    // Verify AlignedStruct with explicit alignment
    EXPECT_TRUE(typeMap.isTypeRegistered("AlignedStruct"));
    EXPECT_EQ(typeMap.getAlignment("AlignedStruct"), 32);
    // Size should be rounded up to multiple of 32 due to alignas(32)
    size_t expected_aligned_size = ((12 + 1 + 7 + 8 + 31) / 32) * 32;
    EXPECT_EQ(typeMap.getTypeSize("AlignedStruct"), expected_aligned_size);
}

// Test clear functionality
TEST_F(TypeMapTest, ClearFunctionality) {
    TypeMap typeMap;
    
    // Register some custom types
    typeMap.registerType("CustomType1", 16);
    typeMap.registerType("CustomType2", 32);
    typeMap.registerTypedef("AliasType", "CustomType1");
    
    // Verify registration
    EXPECT_TRUE(typeMap.isTypeRegistered("CustomType1"));
    EXPECT_TRUE(typeMap.isTypeRegistered("CustomType2"));
    EXPECT_TRUE(typeMap.isTypeRegistered("AliasType"));
    
    // Clear all types
    typeMap.clear();
    
    // Custom types should be gone
    EXPECT_FALSE(typeMap.isTypeRegistered("CustomType1"));
    EXPECT_FALSE(typeMap.isTypeRegistered("CustomType2"));
    EXPECT_FALSE(typeMap.isTypeRegistered("AliasType"));
    
    // But built-in types should still be available
    EXPECT_TRUE(typeMap.isTypeRegistered("int"));
    EXPECT_TRUE(typeMap.isTypeRegistered("float"));
    EXPECT_TRUE(typeMap.isTypeRegistered("double"));
}
