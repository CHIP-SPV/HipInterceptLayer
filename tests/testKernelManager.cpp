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
    )";
};

TEST_F(KernelManagerTest, SerializationTest) {
    // Add kernels to manager
    manager.addFromModuleSource(test_source);
    ASSERT_EQ(manager.getNumKernels(), 2);

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
    ASSERT_EQ(new_manager.getNumKernels(), 2);

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
