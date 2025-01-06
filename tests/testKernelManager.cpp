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

TEST_F(KernelManagerTest, TraceFileTest) {
    // Load the trace file generated during build
    std::string trace_file = std::string(CMAKE_BINARY_DIR) + "/tests/test_kernels-0.trace";
    
    // Create a Tracer instance to read the trace file
    Tracer tracer(trace_file);
    
    // Get the KernelManager from the trace
    const KernelManager& traced_manager = tracer.getKernelManager();
    
    // Verify we have the expected kernels
    ASSERT_EQ(traced_manager.getNumKernels(), 5);
    
    // Verify vectorAdd kernel
    Kernel vector = traced_manager.getKernelByName("vectorAdd");
    EXPECT_EQ(vector.getName(), "vectorAdd");
    EXPECT_EQ(vector.getArguments().size(), 4);
    EXPECT_EQ(vector.getArguments()[0].getName(), "arg1");
    EXPECT_EQ(vector.getArguments()[0].getType(), "float*");
    EXPECT_EQ(vector.getArguments()[1].getName(), "arg2");
    EXPECT_EQ(vector.getArguments()[1].getType(), "float*");
    EXPECT_EQ(vector.getArguments()[2].getName(), "arg3");
    EXPECT_EQ(vector.getArguments()[2].getType(), "float*");
    EXPECT_EQ(vector.getArguments()[3].getName(), "arg4");
    EXPECT_EQ(vector.getArguments()[3].getType(), "int");

    // Verify scalarKernel
    Kernel scalar = traced_manager.getKernelByName("scalarKernel");
    EXPECT_EQ(scalar.getName(), "scalarKernel");
    EXPECT_EQ(scalar.getArguments().size(), 3);
    EXPECT_EQ(scalar.getArguments()[0].getName(), "arg1");
    EXPECT_EQ(scalar.getArguments()[0].getType(), "float*");
    EXPECT_EQ(scalar.getArguments()[1].getName(), "arg2");
    EXPECT_EQ(scalar.getArguments()[1].getType(), "int");
    EXPECT_EQ(scalar.getArguments()[2].getName(), "arg3");
    EXPECT_EQ(scalar.getArguments()[2].getType(), "float");

    // Verify simpleKernel
    Kernel simple = traced_manager.getKernelByName("simpleKernel");
    EXPECT_EQ(simple.getName(), "simpleKernel");
    EXPECT_EQ(simple.getArguments().size(), 1);
    EXPECT_EQ(simple.getArguments()[0].getName(), "arg1");
    EXPECT_EQ(simple.getArguments()[0].getType(), "float*");

    // Verify simpleKernelWithN
    Kernel simpleN = traced_manager.getKernelByName("simpleKernelWithN");
    EXPECT_EQ(simpleN.getName(), "simpleKernelWithN");
    EXPECT_EQ(simpleN.getArguments().size(), 2);
    EXPECT_EQ(simpleN.getArguments()[0].getName(), "arg1");
    EXPECT_EQ(simpleN.getArguments()[0].getType(), "float*");
    EXPECT_EQ(simpleN.getArguments()[1].getName(), "arg2");
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
    ASSERT_EQ(manager.getNumKernels(), 4);

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
    EXPECT_TRUE(complex_args[1].isPointer());
    EXPECT_FALSE(complex_args[2].isPointer());
    EXPECT_TRUE(complex_args[0].isVector());
    EXPECT_EQ(complex_args[0].getType(), "float4*");

    // Verify matrixMulKernel arguments
    Kernel matrix = manager.getKernelByName("matrixMulKernel");
    auto matrix_args = matrix.getArguments();
    ASSERT_EQ(matrix_args.size(), 4);
    EXPECT_TRUE(matrix_args[0].isPointer());
    EXPECT_TRUE(matrix_args[1].isPointer());
    EXPECT_TRUE(matrix_args[2].isPointer());
    EXPECT_FALSE(matrix_args[3].isPointer());
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
    EXPECT_TRUE(args[0].isVector());
    EXPECT_EQ(args[0].getType(), "float4*");
    EXPECT_EQ(args[0].getVectorSize(), 4);
    EXPECT_EQ(args[0].getBaseType(), "float4");
    
    // Check double* argument (non-vector)
    EXPECT_TRUE(args[1].isPointer());
    EXPECT_FALSE(args[1].isVector());
    EXPECT_EQ(args[1].getType(), "double*");
    EXPECT_EQ(args[1].getVectorSize(), 1);
    
    // Check int argument
    EXPECT_FALSE(args[2].isPointer());
    EXPECT_FALSE(args[2].isVector());
    EXPECT_EQ(args[2].getType(), "int");
    EXPECT_EQ(args[2].getVectorSize(), 1);

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
    EXPECT_TRUE(deserialized_args[0].isVector());
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
    EXPECT_TRUE(args[0].isVector());
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
    EXPECT_TRUE(deserialized_args[0].isVector());
    EXPECT_EQ(deserialized_args[0].getType(), "HIP_vector_type<float, 4u>*");
    EXPECT_EQ(deserialized_args[0].getVectorSize(), 4);
    EXPECT_EQ(deserialized_args[0].getBaseType(), "float");

    // Clean up
    std::remove(temp_filename.c_str());
}

TEST(KernelExecutionTest, MemoryStateHandling) {
    // Create test memory states
    auto pre_state = MemoryState(1024);  // 1KB pre state
    auto post_state = MemoryState(2048); // 2KB post state

    // Initialize data with patterns
    {
        // Initialize pre-state data directly in the chunk
        char* pre_data = pre_state.chunks[0].data.get();
        for (size_t i = 0; i < 1024; i++) {
            pre_data[i] = static_cast<char>(i & 0xFF);
        }
    }
    {
        // Initialize post-state data directly in the chunk
        char* post_data = post_state.chunks[0].data.get();
        for (size_t i = 0; i < 2048; i++) {
            post_data[i] = static_cast<char>((i * 2) & 0xFF);
        }
    }

    // Setup kernel execution parameters
    std::string kernel_name = "test_kernel";
    std::cout << "Original kernel name: " << kernel_name << " length: " << kernel_name.length() << std::endl;
    
    void* function_ptr = reinterpret_cast<void*>(0x12345678);  // Dummy function pointer
    dim3 grid(2, 2, 1);
    dim3 block(32, 1, 1);
    size_t shared_mem = 1024;
    hipStream_t stream = nullptr;

    // Setup kernel arguments
    std::vector<void*> arg_ptrs = {reinterpret_cast<void*>(0x1000), reinterpret_cast<void*>(0x2000)};
    std::vector<size_t> arg_sizes = {sizeof(int*), sizeof(float*)};
    
    // Setup scalar values
    int scalar1 = 42;
    float scalar2 = 3.14f;
    std::vector<std::vector<char>> scalar_values;
    scalar_values.emplace_back(sizeof(int));
    scalar_values.emplace_back(sizeof(float));
    std::memcpy(scalar_values[0].data(), &scalar1, sizeof(int));
    std::memcpy(scalar_values[1].data(), &scalar2, sizeof(float));

    // Create kernel execution with these states
    KernelExecution exec(pre_state, post_state, function_ptr, kernel_name,
                        grid, block, shared_mem, stream);
    exec.arg_ptrs = arg_ptrs;
    exec.arg_sizes = arg_sizes;
    exec.scalar_values = std::move(scalar_values);

    std::cout << "Before serialization, kernel name: " << exec.kernel_name << " length: " << exec.kernel_name.length() << std::endl;

    // Test serialization/deserialization
    std::string temp_filename = "test_kernel.bin";
    {
        std::ofstream file(temp_filename, std::ios::binary);
        ASSERT_TRUE(file.is_open());
        exec.serialize(file);
        file.close();
    }

    // Verify the file was written and has content
    {
        std::ifstream check_file(temp_filename, std::ios::binary | std::ios::ate);
        ASSERT_TRUE(check_file.is_open());
        std::streamsize size = check_file.tellg();
        std::cout << "Serialized file size: " << size << " bytes" << std::endl;
        ASSERT_GT(size, 0);
        check_file.close();
    }

    std::shared_ptr<KernelExecution> exec2;
    {
        std::ifstream file(temp_filename, std::ios::binary);
        ASSERT_TRUE(file.is_open());
        exec2 = KernelExecution::create_from_file(file);
        file.close();
    }

    ASSERT_NE(exec2, nullptr);
    std::cout << "After deserialization, kernel name: " << exec2->kernel_name << " length: " << exec2->kernel_name.length() << std::endl;

    // Verify states are preserved after deserialization
    ASSERT_EQ(exec2->pre_state.total_size, 1024);
    ASSERT_EQ(exec2->post_state.total_size, 2048);
    ASSERT_EQ(exec2->function_address, function_ptr);
    ASSERT_EQ(exec2->grid_dim.x, grid.x);
    ASSERT_EQ(exec2->grid_dim.y, grid.y);
    ASSERT_EQ(exec2->block_dim.x, block.x);
    ASSERT_EQ(exec2->block_dim.y, block.y);
    ASSERT_EQ(exec2->shared_mem, shared_mem);
    ASSERT_EQ(exec2->stream, stream);
    ASSERT_EQ(exec2->arg_ptrs.size(), 2);
    ASSERT_EQ(exec2->arg_sizes.size(), 2);
    ASSERT_EQ(exec2->scalar_values.size(), 2);

    // Verify scalar values are preserved
    ASSERT_EQ(exec2->scalar_values[0].size(), sizeof(int));
    ASSERT_EQ(exec2->scalar_values[1].size(), sizeof(float));
    int deserialized_scalar1;
    float deserialized_scalar2;
    std::memcpy(&deserialized_scalar1, exec2->scalar_values[0].data(), sizeof(int));
    std::memcpy(&deserialized_scalar2, exec2->scalar_values[1].data(), sizeof(float));
    ASSERT_EQ(deserialized_scalar1, scalar1);
    ASSERT_FLOAT_EQ(deserialized_scalar2, scalar2);

    // Verify data is preserved after deserialization
    {
        auto pre_data = exec2->pre_state.getData();
        auto post_data = exec2->post_state.getData();
        
        // Verify pre-state data
        for (size_t i = 0; i < 1024; i++) {
            ASSERT_EQ(static_cast<unsigned char>(pre_data.get()[i]), 
                     static_cast<unsigned char>(i & 0xFF));
        }
        
        // Verify post-state data
        for (size_t i = 0; i < 2048; i++) {
            ASSERT_EQ(static_cast<unsigned char>(post_data.get()[i]), 
                     static_cast<unsigned char>((i * 2) & 0xFF));
        }
    }

    // Clean up
    std::remove(temp_filename.c_str());
}