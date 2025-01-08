#include <gtest/gtest.h>
#include "../KernelManager.hh"
#include "../Tracer.hh"
#include <fstream>
#include <sstream>

class KernelManagerTest : public ::testing::Test {
protected:
    KernelManager manager;
    const std::string test_source = R"(
        __global__ void simpleKernel(int* a, float b) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            a[idx] = b;
        }

        __global__ void complexKernel(float4* vectors, double* results, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                results[idx] = vectors[idx].x + vectors[idx].y;
            }
        }

        __global__ void matrixMulKernel(float* A, float* B, float* C, int N) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            float sum = 0.0f;
            if (row < N && col < N) {
                for (int i = 0; i < N; i++) {
                    sum += A[row * N + i] * B[i * N + col];
                }
                C[row * N + col] = sum;
            }
        }

        __global__ void vectorAddKernel(float* a, float* b, float* c, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                c[i] = a[i] + b[i];
            }
        }

        __global__ void complexDataKernel(
            float* scalar_array,                    
            HIP_vector_type<float, 4>* vec4_array, 
            HIP_vector_type<float, 2>* vec2_array, 
            float4* float4_array,                  
            int scalar1,                           
            float scalar2,                         
            double scalar3,                        
            bool flag,                             
            unsigned int uint_val,                 
            size_t n) {}

    )";

    void verifyKernelEquality(const Kernel& k1, const Kernel& k2) {
        EXPECT_EQ(k1.getName(), k2.getName());
        EXPECT_EQ(k1.getSignature(), k2.getSignature());
        EXPECT_EQ(k1.getSource(), k2.getSource());
        
        auto args1 = k1.getArguments();
        auto args2 = k2.getArguments();
        EXPECT_EQ(args1.size(), args2.size());
        
        for (size_t i = 0; i < args1.size(); i++) {
            EXPECT_EQ(args1[i].getName(), args2[i].getName());
            EXPECT_EQ(args1[i].getType(), args2[i].getType());
            EXPECT_EQ(args1[i].getSize(), args2[i].getSize());
            EXPECT_EQ(args1[i].isPointer(), args2[i].isPointer());
        }
    }

    void verifyKernelManagerState(const KernelManager& m1, const KernelManager& m2) {
        EXPECT_EQ(m1.getNumKernels(), m2.getNumKernels());
        
        std::vector<std::string> kernel_names = {
            "simpleKernel", "complexKernel", "matrixMulKernel", "vectorAddKernel"
        };

        for (const auto& name : kernel_names) {
            verifyKernelEquality(m1.getKernelByName(name), m2.getKernelByName(name));
        }
    }
};

TEST_F(KernelManagerTest, ComplexDataTypesParsing) {
  manager.addFromModuleSource(test_source);

  // Get the complexDataKernel
  Kernel complex = manager.getKernelByName("complexDataKernel");
  auto args = complex.getArguments();

  // Verify we have all 10 arguments
  ASSERT_EQ(args.size(), 10) << "Expected 10 arguments in complexDataKernel";

  // Test float* array
  EXPECT_EQ(args[0].getName(), "scalar_array");
  EXPECT_EQ(args[0].getType(), "float*");
  EXPECT_TRUE(args[0].isPointer());

  // Test HIP_vector_type<float, 4>* array
  EXPECT_EQ(args[1].getName(), "vec4_array");
  EXPECT_EQ(args[1].getType(), "HIP_vector_type<float, 4>*");
  EXPECT_TRUE(args[1].isPointer());
  EXPECT_EQ(args[1].getVectorSize(), 4);

  // Test HIP_vector_type<float, 2>* array
  EXPECT_EQ(args[2].getName(), "vec2_array");
  EXPECT_EQ(args[2].getType(), "HIP_vector_type<float, 2>*");
  EXPECT_TRUE(args[2].isPointer());
  EXPECT_EQ(args[2].getVectorSize(), 2);

  // Test float4* array
  EXPECT_EQ(args[3].getName(), "float4_array");
  EXPECT_EQ(args[3].getType(), "float4*");
  EXPECT_TRUE(args[3].isPointer());
  EXPECT_EQ(args[3].getVectorSize(), 4);

  // Test scalar types
  EXPECT_EQ(args[4].getName(), "scalar1");
  EXPECT_EQ(args[4].getType(), "int");
  EXPECT_FALSE(args[4].isPointer());

  EXPECT_EQ(args[5].getName(), "scalar2");
  EXPECT_EQ(args[5].getType(), "float");
  EXPECT_FALSE(args[5].isPointer());

  EXPECT_EQ(args[6].getName(), "scalar3");
  EXPECT_EQ(args[6].getType(), "double");
  EXPECT_FALSE(args[6].isPointer());

  EXPECT_EQ(args[7].getName(), "flag");
  EXPECT_EQ(args[7].getType(), "bool");
  EXPECT_FALSE(args[7].isPointer());

  EXPECT_EQ(args[8].getName(), "uint_val");
  EXPECT_EQ(args[8].getType(), "unsigned int");
  EXPECT_FALSE(args[8].isPointer());

  EXPECT_EQ(args[9].getName(), "n");
  EXPECT_EQ(args[9].getType(), "size_t");
  EXPECT_FALSE(args[9].isPointer());

  // Test serialization/deserialization of the complex types
  std::string temp_filename = "complex_types_test.bin";
  {
    std::ofstream outfile(temp_filename, std::ios::binary);
    ASSERT_TRUE(outfile.is_open());
    manager.serialize(outfile);
  }

  // Create new manager and deserialize
  KernelManager new_manager;
  {
    std::ifstream infile(temp_filename, std::ios::binary);
    ASSERT_TRUE(infile.is_open());
    new_manager.deserialize(infile);
  }

  // Get the deserialized kernel and verify its arguments
  Kernel deserialized_kernel = new_manager.getKernelByName("complexDataKernel");
  auto deserialized_args = deserialized_kernel.getArguments();

  // Verify all arguments are preserved after serialization
  ASSERT_EQ(deserialized_args.size(), args.size());
  for (size_t i = 0; i < args.size(); i++) {
    EXPECT_EQ(deserialized_args[i].getName(), args[i].getName());
    EXPECT_EQ(deserialized_args[i].getType(), args[i].getType());
    EXPECT_EQ(deserialized_args[i].isPointer(), args[i].isPointer());
    EXPECT_EQ(deserialized_args[i].getVectorSize(), args[i].getVectorSize());
  }

  // Clean up
  std::remove(temp_filename.c_str());
}

TEST_F(KernelManagerTest, TraceFileTest) {
  // Load the trace file generated during build
  std::string trace_file =
      std::string(CMAKE_BINARY_DIR) + "/tests/test_kernels-0.trace";

  // Create a Tracer instance to read the trace file
  Tracer tracer(trace_file);

  // Get the KernelManager from the trace
  const KernelManager &traced_manager = tracer.getKernelManager();

  // Verify we have the expected kernels
  ASSERT_EQ(traced_manager.getNumKernels(), 5);

  // Verify vectorAdd kernel
  Kernel vector = traced_manager.getKernelByName("vectorAdd");
  EXPECT_EQ(vector.getName(), "vectorAdd");
  EXPECT_EQ(vector.getArguments().size(), 4);
  EXPECT_EQ(vector.getArguments()[0].getName(), "arg0");
  EXPECT_EQ(vector.getArguments()[0].getType(), "float*");
  EXPECT_EQ(vector.getArguments()[1].getName(), "arg1");
  EXPECT_EQ(vector.getArguments()[1].getType(), "float*");
  EXPECT_EQ(vector.getArguments()[2].getName(), "arg2");
  EXPECT_EQ(vector.getArguments()[2].getType(), "float*");
  EXPECT_EQ(vector.getArguments()[3].getName(), "arg3");
  EXPECT_EQ(vector.getArguments()[3].getType(), "int");

  // Verify scalarKernel
  Kernel scalar = traced_manager.getKernelByName("scalarKernel");
  EXPECT_EQ(scalar.getName(), "scalarKernel");
  EXPECT_EQ(scalar.getArguments().size(), 3);
  EXPECT_EQ(scalar.getArguments()[0].getName(), "arg0");
  EXPECT_EQ(scalar.getArguments()[0].getType(), "float*");
  EXPECT_EQ(scalar.getArguments()[1].getName(), "arg1");
  EXPECT_EQ(scalar.getArguments()[1].getType(), "int");
  EXPECT_EQ(scalar.getArguments()[2].getName(), "arg2");
  EXPECT_EQ(scalar.getArguments()[2].getType(), "float");

  // Verify simpleKernel
  Kernel simple = traced_manager.getKernelByName("simpleKernel");
  EXPECT_EQ(simple.getName(), "simpleKernel");
  EXPECT_EQ(simple.getArguments().size(), 1);
  EXPECT_EQ(simple.getArguments()[0].getName(), "arg0");
  EXPECT_EQ(simple.getArguments()[0].getType(), "float*");

  // Verify simpleKernelWithN
  Kernel simpleN = traced_manager.getKernelByName("simpleKernelWithN");
  EXPECT_EQ(simpleN.getName(), "simpleKernelWithN");
  EXPECT_EQ(simpleN.getArguments().size(), 2);
  EXPECT_EQ(simpleN.getArguments()[0].getName(), "arg0");
  EXPECT_EQ(simpleN.getArguments()[0].getType(), "float*");
  EXPECT_EQ(simpleN.getArguments()[1].getName(), "arg1");
  EXPECT_EQ(simpleN.getArguments()[1].getType(), "int");
}

TEST_F(KernelManagerTest, InvalidDeserialization) {
    // Test with corrupted data
    std::string temp_filename = "corrupted_kernels.bin";
    {
        std::ofstream outfile(temp_filename, std::ios::binary);
        ASSERT_TRUE(outfile.is_open());
        // Write invalid kernel count
        uint32_t invalid_count = 999999;
        outfile.write(reinterpret_cast<const char*>(&invalid_count), sizeof(uint32_t));
    }

    KernelManager corrupt_manager;
    {
        std::ifstream infile(temp_filename, std::ios::binary);
        ASSERT_TRUE(infile.is_open());
        EXPECT_THROW(corrupt_manager.deserialize(infile), std::runtime_error);
    }

    // Clean up
    std::remove(temp_filename.c_str());
}

TEST_F(KernelManagerTest, EmptyManagerSerialization) {
    std::string temp_filename = "empty_kernels.bin";
    
    // Serialize empty manager
    {
        std::ofstream outfile(temp_filename, std::ios::binary);
        ASSERT_TRUE(outfile.is_open());
        manager.serialize(outfile);
    }

    // Deserialize and verify it's empty
    KernelManager empty_manager;
    {
        std::ifstream infile(temp_filename, std::ios::binary);
        ASSERT_TRUE(infile.is_open());
        empty_manager.deserialize(infile);
    }

    EXPECT_EQ(empty_manager.getNumKernels(), 0);

    // Clean up
    std::remove(temp_filename.c_str());
}

TEST_F(KernelManagerTest, MultipleKernelsSerialization) {
    // Add kernels to manager
    manager.addFromModuleSource(test_source);
    ASSERT_EQ(manager.getNumKernels(), 5);

    // Verify initial kernel loading
    EXPECT_NO_THROW(manager.getKernelByName("simpleKernel"));
    EXPECT_NO_THROW(manager.getKernelByName("complexKernel"));
    EXPECT_NO_THROW(manager.getKernelByName("matrixMulKernel"));
    EXPECT_NO_THROW(manager.getKernelByName("vectorAddKernel"));

    // Serialize to temporary file
    std::string temp_filename = "test_kernels.bin";
    {
        std::ofstream outfile(temp_filename, std::ios::binary);
        ASSERT_TRUE(outfile.is_open());
        manager.serialize(outfile);
    }

    // Create new manager and deserialize
    KernelManager new_manager;
    {
        std::ifstream infile(temp_filename, std::ios::binary);
        ASSERT_TRUE(infile.is_open());
        new_manager.deserialize(infile);
    }

    // Verify kernel count and contents
    verifyKernelManagerState(manager, new_manager);

    // Clean up
    std::remove(temp_filename.c_str());
}

TEST_F(KernelManagerTest, KernelArgumentVerification) {
    manager.addFromModuleSource(test_source);

    // Verify simpleKernel arguments
    Kernel simple = manager.getKernelByName("simpleKernel");
    auto simple_args = simple.getArguments();
    ASSERT_EQ(simple_args.size(), 2);
    EXPECT_TRUE(simple_args[0].isPointer());
    EXPECT_FALSE(simple_args[1].isPointer());
    EXPECT_EQ(simple_args[0].getType(), "int*");
    EXPECT_EQ(simple_args[1].getType(), "float");

    // Verify complexKernel arguments
    Kernel complex = manager.getKernelByName("complexKernel");
    auto complex_args = complex.getArguments();
    ASSERT_EQ(complex_args.size(), 3);
    EXPECT_TRUE(complex_args[0].isPointer());
    EXPECT_GT(complex_args[0].getVectorSize(), 0) << "Expected first argument to be a vector type";
    EXPECT_EQ(complex_args[0].getType(), "float4*");
    EXPECT_EQ(complex_args[0].getVectorSize(), 4);
    EXPECT_EQ(complex_args[0].getBaseType(), "float4");

    // Verify matrixMulKernel arguments
    Kernel matrix = manager.getKernelByName("matrixMulKernel");
    auto matrix_args = matrix.getArguments();
    ASSERT_EQ(matrix_args.size(), 4);
    EXPECT_TRUE(matrix_args[0].isPointer());
    EXPECT_TRUE(matrix_args[1].isPointer());
    EXPECT_TRUE(matrix_args[2].isPointer());
    EXPECT_FALSE(matrix_args[3].isPointer());

    // Test vector type detection
    EXPECT_GT(complex_args[0].getVectorSize(), 0) << "Expected first argument to be a vector type";
    EXPECT_EQ(complex_args[1].getVectorSize(), 0) << "Expected second argument to be a scalar type";
    EXPECT_EQ(complex_args[2].getVectorSize(), 0) << "Expected third argument to be a scalar type";
}

TEST_F(KernelManagerTest, SerializationConsistency) {
    manager.addFromModuleSource(test_source);

    // First serialization
    std::string temp_file1 = "test_kernels1.bin";
    {
        std::ofstream outfile(temp_file1, std::ios::binary);
        ASSERT_TRUE(outfile.is_open());
        manager.serialize(outfile);
    }

    // Second serialization
    std::string temp_file2 = "test_kernels2.bin";
    {
        std::ofstream outfile(temp_file2, std::ios::binary);
        ASSERT_TRUE(outfile.is_open());
        manager.serialize(outfile);
    }

    // Compare file contents
    std::ifstream file1(temp_file1, std::ios::binary);
    std::ifstream file2(temp_file2, std::ios::binary);
    ASSERT_TRUE(file1.is_open() && file2.is_open());

    file1.seekg(0, std::ios::end);
    file2.seekg(0, std::ios::end);
    EXPECT_EQ(file1.tellg(), file2.tellg());

    file1.seekg(0);
    file2.seekg(0);
    std::vector<char> content1((std::istreambuf_iterator<char>(file1)),
                              std::istreambuf_iterator<char>());
    std::vector<char> content2((std::istreambuf_iterator<char>(file2)),
                              std::istreambuf_iterator<char>());
    EXPECT_EQ(content1, content2);

    // Clean up
    std::remove(temp_file1.c_str());
    std::remove(temp_file2.c_str());
}

TEST_F(KernelManagerTest, VectorArgumentHandling) {
    manager.addFromModuleSource(test_source);

    // Get the complexKernel which uses float4 vectors
    Kernel complex = manager.getKernelByName("complexKernel");
    auto args = complex.getArguments();
    
    // Verify vector argument properties
    ASSERT_EQ(args.size(), 3);
    
    // Check float4* argument
    EXPECT_TRUE(args[0].isPointer());
    EXPECT_GT(args[0].getVectorSize(), 0) << "Expected first argument to be a vector type";
    EXPECT_EQ(args[0].getType(), "float4*");
    EXPECT_EQ(args[0].getVectorSize(), 4);
    EXPECT_EQ(args[0].getBaseType(), "float4");
    
    // Check double* argument (non-vector)
    EXPECT_TRUE(args[1].isPointer());
    EXPECT_EQ(args[1].getVectorSize(), 0) << "Expected second argument to be a scalar type";
    EXPECT_EQ(args[1].getType(), "double*");
    
    // Check int argument
    EXPECT_FALSE(args[2].isPointer());
    EXPECT_EQ(args[2].getVectorSize(), 0) << "Expected third argument to be a scalar type";
    EXPECT_EQ(args[2].getType(), "int");

    // Test serialization of vector arguments
    std::string temp_filename = "vector_test.bin";
    {
        std::ofstream outfile(temp_filename, std::ios::binary);
        ASSERT_TRUE(outfile.is_open());
        manager.serialize(outfile);
    }

    // Deserialize and verify vector properties are preserved
    KernelManager new_manager;
    {
        std::ifstream infile(temp_filename, std::ios::binary);
        ASSERT_TRUE(infile.is_open());
        new_manager.deserialize(infile);
    }

    Kernel deserialized_kernel = new_manager.getKernelByName("complexKernel");
    auto deserialized_args = deserialized_kernel.getArguments();
    
    // Verify vector properties after deserialization
    EXPECT_TRUE(deserialized_args[0].isPointer());
    EXPECT_GT(deserialized_args[0].getVectorSize(), 0) << "Expected first deserialized argument to be a vector type";
    EXPECT_EQ(deserialized_args[0].getType(), "float4*");
    EXPECT_EQ(deserialized_args[0].getVectorSize(), 4);
    EXPECT_EQ(deserialized_args[0].getBaseType(), "float4");

    // Clean up
    std::remove(temp_filename.c_str());
}

TEST_F(KernelManagerTest, HIPVectorTypeHandling) {
    // Test source with HIP vector type
    const std::string hip_vector_source = R"(
        __global__ void vectorKernel(HIP_vector_type<float, 4u>* vec) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            vec[idx].x = 1.0f;
        }
    )";

    KernelManager hip_manager;
    hip_manager.addFromModuleSource(hip_vector_source);

    // Get the vectorKernel
    Kernel vector_kernel = hip_manager.getKernelByName("vectorKernel");
    auto args = vector_kernel.getArguments();
    
    // Verify argument properties
    ASSERT_EQ(args.size(), 1);
    
    // Check HIP_vector_type argument
    EXPECT_TRUE(args[0].isPointer());
    EXPECT_GT(args[0].getVectorSize(), 0) << "Expected first argument to be a vector type";
    EXPECT_EQ(args[0].getType(), "HIP_vector_type<float, 4u>*");
    EXPECT_EQ(args[0].getVectorSize(), 4);
    EXPECT_EQ(args[0].getBaseType(), "float");

    // Test serialization of HIP vector type
    std::string temp_filename = "hip_vector_test.bin";
    {
        std::ofstream outfile(temp_filename, std::ios::binary);
        ASSERT_TRUE(outfile.is_open());
        hip_manager.serialize(outfile);
    }

    // Deserialize and verify vector properties are preserved
    KernelManager new_manager;
    {
        std::ifstream infile(temp_filename, std::ios::binary);
        ASSERT_TRUE(infile.is_open());
        new_manager.deserialize(infile);
    }

    Kernel deserialized_kernel = new_manager.getKernelByName("vectorKernel");
    auto deserialized_args = deserialized_kernel.getArguments();
    
    // Verify vector properties after deserialization
    EXPECT_TRUE(deserialized_args[0].isPointer());
    EXPECT_GT(deserialized_args[0].getVectorSize(), 0) << "Expected first argument to be a vector type";
    EXPECT_EQ(deserialized_args[0].getType(), "HIP_vector_type<float, 4u>*");
    EXPECT_EQ(deserialized_args[0].getVectorSize(), 4);
    EXPECT_EQ(deserialized_args[0].getBaseType(), "float");

    // Clean up
    std::remove(temp_filename.c_str());
}

TEST(KernelExecutionTest, ArgumentHandling) {
    // Create test arguments
    std::vector<ArgState> pre_args;
    std::vector<ArgState> post_args;

    // Create a scalar argument
    ArgState scalar_arg(sizeof(int), 1);
    int scalar_value = 42;
    std::memcpy(scalar_arg.data.data(), &scalar_value, sizeof(int));
    pre_args.push_back(scalar_arg);

    // Create an array argument
    ArgState array_arg(sizeof(float), 4);
    float array_values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::memcpy(array_arg.data.data(), array_values, sizeof(float) * 4);
    pre_args.push_back(array_arg);

    // Create a kernel execution with these arguments
    KernelExecution exec(
        pre_args,
        post_args,
        nullptr,
        "test_kernel",
        dim3(1, 1, 1),
        dim3(1, 1, 1),
        0,
        nullptr
    );

    // Test serialization
    std::ofstream out_file("test_kernel.bin", std::ios::binary);
    exec.serialize(out_file);
    out_file.close();

    // Test deserialization
    std::ifstream in_file("test_kernel.bin", std::ios::binary);
    auto deserialized = KernelExecution::create_from_file(in_file);
    in_file.close();

    // Verify the deserialized data
    ASSERT_EQ(deserialized->kernel_name, "test_kernel");
    ASSERT_EQ(deserialized->pre_args.size(), 2);
    
    // Verify scalar argument
    const auto& deserialized_scalar = deserialized->pre_args[0];
    ASSERT_EQ(deserialized_scalar.data_type_size, sizeof(int));
    ASSERT_EQ(deserialized_scalar.array_size, 1);
    int deserialized_value;
    std::memcpy(&deserialized_value, deserialized_scalar.data.data(), sizeof(int));
    ASSERT_EQ(deserialized_value, 42);

    // Verify array argument
    const auto& deserialized_array_arg = deserialized->pre_args[1];
    ASSERT_EQ(deserialized_array_arg.data_type_size, sizeof(float));
    ASSERT_EQ(deserialized_array_arg.array_size, 4);
    
    std::vector<float> deserialized_values(4);
    std::memcpy(deserialized_values.data(), deserialized_array_arg.data.data(), sizeof(float) * 4);
    for (int i = 0; i < 4; i++) {
        ASSERT_FLOAT_EQ(deserialized_values[i], array_values[i]);
    }

    // Clean up
    std::remove("test_kernel.bin");
}

TEST_F(KernelManagerTest, CompoundTypesParsing) {
    KernelManager manager;
    
    // Create a test kernel source with various compound types
    const std::string test_source = R"(
        __global__ void compoundTypesKernel(
            unsigned int value,
            unsigned long long counter,
            unsigned int *ptr,
            unsigned int unnamed,
            int x,
            float *fptr,
            HIP_vector_type<float, 4> vec,
            unsigned short ushort_val,
            unsigned char uchar_val,
            uint64_t uint64_val,
            int64_t int64_val,
            uint32_t uint32_val,
            int32_t int32_val,
            uint16_t uint16_val,
            int16_t int16_val,
            uint8_t uint8_val,
            int8_t int8_val,
            unsigned int,
            __restrict__ unsigned int*,
            __restrict__ unsigned int* test_ptr,
            float4* in_scalar_float4,
            char2,
            HIP_vector_type<float, 4>
        ) {}
    )";

    manager.addFromModuleSource(test_source);
    Kernel kernel = manager.getKernelByName("compoundTypesKernel");
    auto args = kernel.getArguments();

    // Test compound types
    EXPECT_EQ(args[0].getType(), "unsigned int");
    EXPECT_EQ(args[0].getName(), "value");

    EXPECT_EQ(args[1].getType(), "unsigned long long");
    EXPECT_EQ(args[1].getName(), "counter");

    EXPECT_EQ(args[2].getType(), "unsigned int*");
    EXPECT_EQ(args[2].getName(), "ptr");

    // Test auto-naming when no name provided
    EXPECT_EQ(args[3].getType(), "unsigned int");
    EXPECT_EQ(args[3].getName(), "unnamed");

    // Test regular types
    EXPECT_EQ(args[4].getType(), "int");
    EXPECT_EQ(args[4].getName(), "x");

    // Test pointer types
    EXPECT_EQ(args[5].getType(), "float*");
    EXPECT_EQ(args[5].getName(), "fptr");

    // Test template types
    EXPECT_EQ(args[6].getType(), "HIP_vector_type<float, 4>");
    EXPECT_EQ(args[6].getName(), "vec");

    // Test additional integer types
    EXPECT_EQ(args[7].getType(), "unsigned short");
    EXPECT_EQ(args[7].getName(), "ushort_val");

    EXPECT_EQ(args[8].getType(), "unsigned char");
    EXPECT_EQ(args[8].getName(), "uchar_val");

    EXPECT_EQ(args[9].getType(), "uint64_t");
    EXPECT_EQ(args[9].getName(), "uint64_val");

    EXPECT_EQ(args[10].getType(), "int64_t");
    EXPECT_EQ(args[10].getName(), "int64_val");

    EXPECT_EQ(args[11].getType(), "uint32_t");
    EXPECT_EQ(args[11].getName(), "uint32_val");

    EXPECT_EQ(args[12].getType(), "int32_t");
    EXPECT_EQ(args[12].getName(), "int32_val");

    EXPECT_EQ(args[13].getType(), "uint16_t");
    EXPECT_EQ(args[13].getName(), "uint16_val");

    EXPECT_EQ(args[14].getType(), "int16_t");
    EXPECT_EQ(args[14].getName(), "int16_val");

    EXPECT_EQ(args[15].getType(), "uint8_t");
    EXPECT_EQ(args[15].getName(), "uint8_val");

    EXPECT_EQ(args[16].getType(), "int8_t");
    EXPECT_EQ(args[16].getName(), "int8_val");

    EXPECT_EQ(args[17].getType(), "unsigned int");
    EXPECT_EQ(args[17].getName(), "arg17");

    EXPECT_EQ(args[18].getType(), "__restrict__ unsigned int*"); 
    EXPECT_EQ(args[18].getName(), "arg18");

    EXPECT_EQ(args[19].getType(), "__restrict__ unsigned int*");
    EXPECT_EQ(args[19].getName(), "test_ptr");

    EXPECT_EQ(args[20].getType(), "float4*");
    EXPECT_EQ(args[20].getName(), "in_scalar_float4");

    EXPECT_EQ(args[21].getType(), "char2");
    EXPECT_EQ(args[21].getName(), "arg21");
}

TEST_F(KernelManagerTest, ArgumentSerialization) {
    KernelManager manager;
    
    // Create a test kernel source with various compound types
    const std::string test_source = R"(
        __global__ void compoundTypesKernel(
            unsigned int value,
            unsigned long long counter,
            unsigned int *ptr,
            unsigned int unnamed,
            int x,
            float *fptr,
            HIP_vector_type<float, 4> vec,
            unsigned short ushort_val,
            unsigned char uchar_val,
            uint64_t uint64_val,
            int64_t int64_val,
            uint32_t uint32_val,
            int32_t int32_val,
            uint16_t uint16_val,
            int16_t int16_val,
            uint8_t uint8_val,
            int8_t int8_val,
            unsigned int,
            __restrict__ unsigned int*,
            __restrict__ unsigned int* test_ptr,
            float4* in_scalar_float4,
            char2,
            HIP_vector_type<float, 4>
        ) {}
    )";

    // Add the kernel to the manager
    manager.addFromModuleSource(test_source);
    Kernel original_kernel = manager.getKernelByName("compoundTypesKernel");
    auto original_args = original_kernel.getArguments();

    // Serialize to a temporary file
    std::string temp_filename = "argument_test.bin";
    {
        std::ofstream outfile(temp_filename, std::ios::binary);
        ASSERT_TRUE(outfile.is_open());
        manager.serialize(outfile);
    }

    // Create new manager and deserialize
    KernelManager new_manager;
    {
        std::ifstream infile(temp_filename, std::ios::binary);
        ASSERT_TRUE(infile.is_open());
        new_manager.deserialize(infile);
    }

    // Get the deserialized kernel and its arguments
    Kernel deserialized_kernel = new_manager.getKernelByName("compoundTypesKernel");
    auto deserialized_args = deserialized_kernel.getArguments();

    // Verify the number of arguments matches
    ASSERT_EQ(deserialized_args.size(), original_args.size());

    // Verify each argument's properties
    for (size_t i = 0; i < original_args.size(); i++) {
        const auto& orig = original_args[i];
        const auto& deser = deserialized_args[i];

        // Test type preservation
        EXPECT_EQ(deser.getType(), orig.getType()) 
            << "Type mismatch at argument " << i;
        
        // Test name preservation
        EXPECT_EQ(deser.getName(), orig.getName()) 
            << "Name mismatch at argument " << i;
        
        // Test pointer property preservation
        EXPECT_EQ(deser.isPointer(), orig.isPointer()) 
            << "Pointer property mismatch at argument " << i;
        
        // Test vector size preservation
        EXPECT_EQ(deser.getVectorSize(), orig.getVectorSize()) 
            << "Vector size mismatch at argument " << i;
        
        // For vector types, test base type preservation
        if (orig.getVectorSize() > 0) {
            EXPECT_EQ(deser.getBaseType(), orig.getBaseType()) 
                << "Base type mismatch at argument " << i;
        }
    }

    // Clean up
    std::remove(temp_filename.c_str());
}