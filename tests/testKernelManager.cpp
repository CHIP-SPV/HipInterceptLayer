#include <gtest/gtest.h>
#include "../KernelManager.hh"
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

TEST_F(KernelManagerTest, SerializationTest) {
    // Add kernels to manager
    manager.addFromModuleSource(test_source);
    ASSERT_EQ(manager.getNumKernels(), 4);

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

    // Verify kernel count
    ASSERT_EQ(new_manager.getNumKernels(), 4);

    // Verify first kernel (simpleKernel)
    Kernel simple = new_manager.getKernelByName("simpleKernel");
    EXPECT_EQ(simple.getName(), "simpleKernel");
    EXPECT_EQ(simple.getArguments().size(), 2);
    EXPECT_EQ(simple.getArguments()[0].getName(), "a");
    EXPECT_EQ(simple.getArguments()[0].getType(), "int*");
    EXPECT_EQ(simple.getArguments()[1].getName(), "b");
    EXPECT_EQ(simple.getArguments()[1].getType(), "float");

    // Verify second kernel (complexKernel)
    Kernel complex = new_manager.getKernelByName("complexKernel");
    EXPECT_EQ(complex.getName(), "complexKernel");
    EXPECT_EQ(complex.getArguments().size(), 3);
    EXPECT_EQ(complex.getArguments()[0].getName(), "vectors");
    EXPECT_EQ(complex.getArguments()[0].getType(), "float4*");
    EXPECT_EQ(complex.getArguments()[1].getName(), "results");
    EXPECT_EQ(complex.getArguments()[1].getType(), "double*");
    EXPECT_EQ(complex.getArguments()[2].getName(), "n");
    EXPECT_EQ(complex.getArguments()[2].getType(), "int");

    // Verify third kernel (matrixMulKernel)
    Kernel matrix = new_manager.getKernelByName("matrixMulKernel");
    EXPECT_EQ(matrix.getName(), "matrixMulKernel");
    EXPECT_EQ(matrix.getArguments().size(), 4);
    EXPECT_EQ(matrix.getArguments()[0].getName(), "A");
    EXPECT_EQ(matrix.getArguments()[0].getType(), "float*");
    EXPECT_EQ(matrix.getArguments()[1].getName(), "B");
    EXPECT_EQ(matrix.getArguments()[1].getType(), "float*");
    EXPECT_EQ(matrix.getArguments()[2].getName(), "C");
    EXPECT_EQ(matrix.getArguments()[2].getType(), "float*");
    EXPECT_EQ(matrix.getArguments()[3].getName(), "N");
    EXPECT_EQ(matrix.getArguments()[3].getType(), "int");

    // Verify fourth kernel (vectorAddKernel)
    Kernel vector = new_manager.getKernelByName("vectorAddKernel");
    EXPECT_EQ(vector.getName(), "vectorAddKernel");
    EXPECT_EQ(vector.getArguments().size(), 4);
    EXPECT_EQ(vector.getArguments()[0].getName(), "a");
    EXPECT_EQ(vector.getArguments()[0].getType(), "float*");
    EXPECT_EQ(vector.getArguments()[1].getName(), "b");
    EXPECT_EQ(vector.getArguments()[1].getType(), "float*");
    EXPECT_EQ(vector.getArguments()[2].getName(), "c");
    EXPECT_EQ(vector.getArguments()[2].getType(), "float*");
    EXPECT_EQ(vector.getArguments()[3].getName(), "n");
    EXPECT_EQ(vector.getArguments()[3].getType(), "int");

    // Clean up
    std::remove(temp_filename.c_str());
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
