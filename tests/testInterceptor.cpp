#include "../Interceptor.hh"
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>

// Define tileflags type as used in computeNonbondedRepro.cpp
typedef unsigned int tileflags;

class InterceptorTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Set up code if needed
  }
};

// Test to verify vectorAdd kernel behavior from trace
TEST_F(InterceptorTest, CompareVectorAddTraceWithSource) {
  // Load and verify the trace file
  std::string trace_file =
      std::string(CMAKE_BINARY_DIR) + "/tests/vectorAdd-0.trace";
  Tracer tracer(trace_file);

  // Find the kernel operation by name
  auto kernel_indices = tracer.getOperationsIdxByName("vectorIncrementKernel");
  ASSERT_FALSE(kernel_indices.empty())
      << "No vectorIncrementKernel found in trace";

  // Get the first instance of the kernel
  auto kernel_op = std::dynamic_pointer_cast<KernelExecution>(
      tracer.getOperation(kernel_indices[0]));
  ASSERT_NE(kernel_op, nullptr);
  std::cout << "Found kernel operation at index " << kernel_indices[0]
            << std::endl;

  // Add debug output for kernel operation details
  std::cout << "Kernel operation details:" << std::endl;
  std::cout << "  Name: " << kernel_op->kernel_name << std::endl;
  std::cout << "  Grid dim: " << kernel_op->grid_dim.x << ","
            << kernel_op->grid_dim.y << "," << kernel_op->grid_dim.z
            << std::endl;
  std::cout << "  Block dim: " << kernel_op->block_dim.x << ","
            << kernel_op->block_dim.y << "," << kernel_op->block_dim.z
            << std::endl;
  std::cout << "  Shared mem: " << kernel_op->shared_mem << std::endl;
  std::cout << "  Number of arguments: " << kernel_op->arg_ptrs.size()
            << std::endl;

  // Verify kernel launch parameters
  EXPECT_EQ(kernel_op->kernel_name, "vectorIncrementKernel");
  EXPECT_EQ(kernel_op->block_dim.x, 256); // From vectorAdd.cpp
  EXPECT_EQ(kernel_op->block_dim.y, 1);
  EXPECT_EQ(kernel_op->block_dim.z, 1);
  EXPECT_EQ(kernel_op->shared_mem, 0);

  // Verify pre_args - should have 4 arguments
  ASSERT_EQ(kernel_op->pre_args.size(), 4)
      << "Expected 4 arguments (in_scalar_float4, in_scalar_float, "
         "inout_vector_float4, inout_vector_float)";

  // First argument: in_scalar_float4 (float4)
  ASSERT_EQ(kernel_op->pre_args[0].total_size(), sizeof(float4))
      << "Expected float4 to be 16 bytes";
  const float4 *in_scalar_float4 =
      reinterpret_cast<const float4 *>(kernel_op->pre_args[0].data.data());
  EXPECT_FLOAT_EQ(in_scalar_float4->x, 1.0f);
  EXPECT_FLOAT_EQ(in_scalar_float4->y, 2.0f);
  EXPECT_FLOAT_EQ(in_scalar_float4->z, 3.0f);
  EXPECT_FLOAT_EQ(in_scalar_float4->w, 4.0f);

  // Second argument: in_scalar_float (float)
  ASSERT_EQ(kernel_op->pre_args[1].total_size(), sizeof(float))
      << "Expected float to be 4 bytes";
  const float *in_scalar_float =
      reinterpret_cast<const float *>(kernel_op->pre_args[1].data.data());
  EXPECT_FLOAT_EQ(*in_scalar_float, 0.5f);

  // Third argument: inout_vector_float4 (float4*)
  const size_t N = 1024; // Size from vectorAdd.cpp
  ASSERT_EQ(kernel_op->pre_args[2].total_size(), N * sizeof(float4))
      << "Expected array of 1024 float4s";
  const float4 *inout_vector_float4 =
      reinterpret_cast<const float4 *>(kernel_op->pre_args[2].data.data());

  // Fourth argument: inout_vector_float (float*)
  ASSERT_EQ(kernel_op->pre_args[3].total_size(), N * sizeof(float))
      << "Expected array of 1024 floats";
  const float *inout_vector_float =
      reinterpret_cast<const float *>(kernel_op->pre_args[3].data.data());

  // Print first few pre-state values for debugging
  std::cout << "\nPre-execution state:" << std::endl;
  std::cout << "  in_scalar_float4: (" << in_scalar_float4->x << ", "
            << in_scalar_float4->y << ", " << in_scalar_float4->z << ", "
            << in_scalar_float4->w << ")" << std::endl;
  std::cout << "  in_scalar_float: " << *in_scalar_float << std::endl;
  std::cout << "\nFirst few vector values:" << std::endl;
  for (int i = 0; i < 5; i++) {
    std::cout << "  Index " << i << ":" << std::endl;
    std::cout << "    inout_vector_float4: (" << inout_vector_float4[i].x
              << ", " << inout_vector_float4[i].y << ", "
              << inout_vector_float4[i].z << ", " << inout_vector_float4[i].w
              << ")" << std::endl;
    std::cout << "    inout_vector_float: " << inout_vector_float[i]
              << std::endl;
  }

  // Post-execution state
  ASSERT_FALSE(kernel_op->post_args.empty()) << "No post-execution arguments";
  ASSERT_EQ(kernel_op->post_args.size(), 4)
      << "Expected 4 post-execution arguments";

  // Get post-execution arrays (only the output vectors change)
  const float4 *post_vector_float4 =
      reinterpret_cast<const float4 *>(kernel_op->post_args[2].data.data());
  const float *post_vector_float =
      reinterpret_cast<const float *>(kernel_op->post_args[3].data.data());

  // Print first few post-state values and verify results
  std::cout << "\nPost-execution state (first 5 elements):" << std::endl;
  for (int i = 0; i < 5; i++) {
    std::cout << "  Index " << i << ":" << std::endl;
    std::cout << "    inout_vector_float4: (" << post_vector_float4[i].x << ", "
              << post_vector_float4[i].y << ", " << post_vector_float4[i].z
              << ", " << post_vector_float4[i].w << ")" << std::endl;
    std::cout << "    inout_vector_float: " << post_vector_float[i]
              << std::endl;

    // Verify the results match the kernel computation
    EXPECT_FLOAT_EQ(post_vector_float4[i].x,
                    in_scalar_float4->x + *in_scalar_float);
    EXPECT_FLOAT_EQ(post_vector_float4[i].y,
                    in_scalar_float4->y + *in_scalar_float);
    EXPECT_FLOAT_EQ(post_vector_float4[i].z,
                    in_scalar_float4->z + *in_scalar_float);
    EXPECT_FLOAT_EQ(post_vector_float4[i].w,
                    in_scalar_float4->w + *in_scalar_float);
    EXPECT_FLOAT_EQ(post_vector_float[i], *in_scalar_float * 2.0f);
  }
}

// Test to verify computeNonbonded kernel behavior from trace
TEST_F(InterceptorTest, CompareComputeNonbondedReproTraceWithSource) {
  // Load and verify the trace file
  std::string trace_file =
      std::string(CMAKE_BINARY_DIR) + "/tests/computeNonbondedRepro-0.trace";
  Tracer tracer(trace_file);

  // Find the kernel operation by name
  auto kernel_indices = tracer.getOperationsIdxByName("computeNonbonded");
  ASSERT_FALSE(kernel_indices.empty())
      << "No computeNonbonded kernel found in trace";

  // Get the first instance of the kernel
  auto kernel_op = std::dynamic_pointer_cast<KernelExecution>(
      tracer.getOperation(kernel_indices[0]));
  ASSERT_NE(kernel_op, nullptr);
  std::cout << "Found kernel operation at index " << kernel_indices[0]
            << std::endl;

  // Add debug output for kernel operation details
  std::cout << "Kernel operation details:" << std::endl;
  std::cout << "  Name: " << kernel_op->kernel_name << std::endl;
  std::cout << "  Grid dim: " << kernel_op->grid_dim.x << ","
            << kernel_op->grid_dim.y << "," << kernel_op->grid_dim.z
            << std::endl;
  std::cout << "  Block dim: " << kernel_op->block_dim.x << ","
            << kernel_op->block_dim.y << "," << kernel_op->block_dim.z
            << std::endl;
  std::cout << "  Shared mem: " << kernel_op->shared_mem << std::endl;
  std::cout << "  Number of arguments: " << kernel_op->arg_ptrs.size()
            << std::endl;

  // Verify kernel launch parameters
  EXPECT_EQ(kernel_op->kernel_name, "computeNonbonded");
  EXPECT_EQ(kernel_op->grid_dim.x, 2); // NUM_BLOCKS = (128 + 64 - 1) / 64 = 2
  EXPECT_EQ(kernel_op->grid_dim.y, 1);
  EXPECT_EQ(kernel_op->grid_dim.z, 1);
  EXPECT_EQ(kernel_op->block_dim.x, 64); // BLOCK_SIZE from source
  EXPECT_EQ(kernel_op->block_dim.y, 1);
  EXPECT_EQ(kernel_op->block_dim.z, 1);
  EXPECT_EQ(kernel_op->shared_mem, 0);

  // Verify pre_args - should have all kernel arguments
  const int N = 128; // Size from source
  ASSERT_EQ(kernel_op->pre_args.size(), 21)
      << "Expected 21 arguments for computeNonbonded kernel";

  // Verify each argument's size and initial values based on
  // computeNonbondedRepro.cpp
  // 1. forceBuffers (unsigned long long*)
  ASSERT_EQ(kernel_op->pre_args[0].total_size(),
            N * sizeof(unsigned long long));
  const unsigned long long *forceBuffers =
      reinterpret_cast<const unsigned long long *>(
          kernel_op->pre_args[0].data.data());
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(forceBuffers[i], i) << "forceBuffers[" << i << "] mismatch";
  }

  // 2. energyBuffer (float*)
  ASSERT_EQ(kernel_op->pre_args[1].total_size(), N * sizeof(float));
  const float *energyBuffer =
      reinterpret_cast<const float *>(kernel_op->pre_args[1].data.data());
  for (int i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(energyBuffer[i], static_cast<float>(i))
        << "energyBuffer[" << i << "] mismatch";
  }

  // 3. posq (float4*)
  ASSERT_EQ(kernel_op->pre_args[2].total_size(), N * sizeof(float4));
  const float4 *posq =
      reinterpret_cast<const float4 *>(kernel_op->pre_args[2].data.data());
  for (int i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(posq[i].x, i) << "posq[" << i << "].x mismatch";
    EXPECT_FLOAT_EQ(posq[i].y, i + 0.1f) << "posq[" << i << "].y mismatch";
    EXPECT_FLOAT_EQ(posq[i].z, i + 0.2f) << "posq[" << i << "].z mismatch";
    EXPECT_FLOAT_EQ(posq[i].w, i + 0.3f) << "posq[" << i << "].w mismatch";
  }

  // 4. exclusions (tileflags*)
  ASSERT_EQ(kernel_op->pre_args[3].total_size(), N * sizeof(tileflags));
  const tileflags *exclusions =
      reinterpret_cast<const tileflags *>(kernel_op->pre_args[3].data.data());
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(exclusions[i], static_cast<tileflags>(i))
        << "exclusions[" << i << "] mismatch";
  }

  // 5. exclusionTiles (int2*)
  ASSERT_EQ(kernel_op->pre_args[4].total_size(), N * sizeof(int2));
  const int2 *exclusionTiles =
      reinterpret_cast<const int2 *>(kernel_op->pre_args[4].data.data());
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(exclusionTiles[i].x, i)
        << "exclusionTiles[" << i << "].x mismatch";
    EXPECT_EQ(exclusionTiles[i].y, i + 1)
        << "exclusionTiles[" << i << "].y mismatch";
  }

  // 6. startTileIndex (unsigned int)
  ASSERT_EQ(kernel_op->pre_args[5].total_size(), sizeof(unsigned int));
  const unsigned int *startTileIndex = reinterpret_cast<const unsigned int *>(
      kernel_op->pre_args[5].data.data());
  EXPECT_EQ(*startTileIndex, 0u) << "startTileIndex mismatch";

  // 7. numTileIndices (unsigned long long)
  ASSERT_EQ(kernel_op->pre_args[6].total_size(), sizeof(unsigned long long));
  const unsigned long long *numTileIndices =
      reinterpret_cast<const unsigned long long *>(
          kernel_op->pre_args[6].data.data());
  EXPECT_EQ(*numTileIndices, N) << "numTileIndices mismatch";

  // 8. tiles (int*)
  ASSERT_EQ(kernel_op->pre_args[7].total_size(), N * sizeof(int));
  const int *tiles =
      reinterpret_cast<const int *>(kernel_op->pre_args[7].data.data());
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(tiles[i], i) << "tiles[" << i << "] mismatch";
  }

  // 9. interactionCount (unsigned int*)
  ASSERT_EQ(kernel_op->pre_args[8].total_size(), N * sizeof(unsigned int));
  const unsigned int *interactionCount = reinterpret_cast<const unsigned int *>(
      kernel_op->pre_args[8].data.data());
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(interactionCount[i], static_cast<unsigned int>(i))
        << "interactionCount[" << i << "] mismatch";
  }

  // 10. periodicBoxSize (float4)
  ASSERT_EQ(kernel_op->pre_args[9].total_size(), sizeof(float4));
  const float4 *periodicBoxSize =
      reinterpret_cast<const float4 *>(kernel_op->pre_args[9].data.data());
  EXPECT_FLOAT_EQ(periodicBoxSize->x, 10.0f) << "periodicBoxSize.x mismatch";
  EXPECT_FLOAT_EQ(periodicBoxSize->y, 10.0f) << "periodicBoxSize.y mismatch";
  EXPECT_FLOAT_EQ(periodicBoxSize->z, 10.0f) << "periodicBoxSize.z mismatch";
  EXPECT_FLOAT_EQ(periodicBoxSize->w, 0.0f) << "periodicBoxSize.w mismatch";

  // 11. invPeriodicBoxSize (float4)
  ASSERT_EQ(kernel_op->pre_args[10].total_size(), sizeof(float4));
  const float4 *invPeriodicBoxSize =
      reinterpret_cast<const float4 *>(kernel_op->pre_args[10].data.data());
  EXPECT_FLOAT_EQ(invPeriodicBoxSize->x, 0.1f)
      << "invPeriodicBoxSize.x mismatch";
  EXPECT_FLOAT_EQ(invPeriodicBoxSize->y, 0.1f)
      << "invPeriodicBoxSize.y mismatch";
  EXPECT_FLOAT_EQ(invPeriodicBoxSize->z, 0.1f)
      << "invPeriodicBoxSize.z mismatch";
  EXPECT_FLOAT_EQ(invPeriodicBoxSize->w, 0.0f)
      << "invPeriodicBoxSize.w mismatch";

  // 12-14. periodicBoxVecX/Y/Z (float4)
  const float4 *periodicBoxVecX =
      reinterpret_cast<const float4 *>(kernel_op->pre_args[11].data.data());
  const float4 *periodicBoxVecY =
      reinterpret_cast<const float4 *>(kernel_op->pre_args[12].data.data());
  const float4 *periodicBoxVecZ =
      reinterpret_cast<const float4 *>(kernel_op->pre_args[13].data.data());

  EXPECT_FLOAT_EQ(periodicBoxVecX->x, 10.0f) << "periodicBoxVecX.x mismatch";
  EXPECT_FLOAT_EQ(periodicBoxVecX->y, 0.0f) << "periodicBoxVecX.y mismatch";
  EXPECT_FLOAT_EQ(periodicBoxVecX->z, 0.0f) << "periodicBoxVecX.z mismatch";
  EXPECT_FLOAT_EQ(periodicBoxVecX->w, 0.0f) << "periodicBoxVecX.w mismatch";

  EXPECT_FLOAT_EQ(periodicBoxVecY->x, 0.0f) << "periodicBoxVecY.x mismatch";
  EXPECT_FLOAT_EQ(periodicBoxVecY->y, 10.0f) << "periodicBoxVecY.y mismatch";
  EXPECT_FLOAT_EQ(periodicBoxVecY->z, 0.0f) << "periodicBoxVecY.z mismatch";
  EXPECT_FLOAT_EQ(periodicBoxVecY->w, 0.0f) << "periodicBoxVecY.w mismatch";

  EXPECT_FLOAT_EQ(periodicBoxVecZ->x, 0.0f) << "periodicBoxVecZ.x mismatch";
  EXPECT_FLOAT_EQ(periodicBoxVecZ->y, 0.0f) << "periodicBoxVecZ.y mismatch";
  EXPECT_FLOAT_EQ(periodicBoxVecZ->z, 10.0f) << "periodicBoxVecZ.z mismatch";
  EXPECT_FLOAT_EQ(periodicBoxVecZ->w, 0.0f) << "periodicBoxVecZ.w mismatch";

  // 15. maxTiles (unsigned int)
  ASSERT_EQ(kernel_op->pre_args[14].total_size(), sizeof(unsigned int));
  const unsigned int *maxTiles = reinterpret_cast<const unsigned int *>(
      kernel_op->pre_args[14].data.data());
  EXPECT_EQ(*maxTiles, N) << "maxTiles mismatch";

  // 16. blockCenter (float4*)
  ASSERT_EQ(kernel_op->pre_args[15].total_size(), N * sizeof(float4));
  const float4 *blockCenter =
      reinterpret_cast<const float4 *>(kernel_op->pre_args[15].data.data());
  for (int i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(blockCenter[i].x, i + 0.5f)
        << "blockCenter[" << i << "].x mismatch";
    EXPECT_FLOAT_EQ(blockCenter[i].y, i + 0.6f)
        << "blockCenter[" << i << "].y mismatch";
    EXPECT_FLOAT_EQ(blockCenter[i].z, i + 0.7f)
        << "blockCenter[" << i << "].z mismatch";
    EXPECT_FLOAT_EQ(blockCenter[i].w, i + 0.8f)
        << "blockCenter[" << i << "].w mismatch";
  }

  // 17. blockSize (float4*)
  ASSERT_EQ(kernel_op->pre_args[16].total_size(), N * sizeof(float4));
  const float4 *blockSize =
      reinterpret_cast<const float4 *>(kernel_op->pre_args[16].data.data());
  for (int i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(blockSize[i].x, 1.0f)
        << "blockSize[" << i << "].x mismatch";
    EXPECT_FLOAT_EQ(blockSize[i].y, 1.0f)
        << "blockSize[" << i << "].y mismatch";
    EXPECT_FLOAT_EQ(blockSize[i].z, 1.0f)
        << "blockSize[" << i << "].z mismatch";
    EXPECT_FLOAT_EQ(blockSize[i].w, 1.0f)
        << "blockSize[" << i << "].w mismatch";
  }

  // 18. interactingAtoms (unsigned int*)
  ASSERT_EQ(kernel_op->pre_args[17].total_size(), N * sizeof(unsigned int));
  const unsigned int *interactingAtoms = reinterpret_cast<const unsigned int *>(
      kernel_op->pre_args[17].data.data());
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(interactingAtoms[i], static_cast<unsigned int>(i))
        << "interactingAtoms[" << i << "] mismatch";
  }

  // 19. maxSinglePairs (unsigned int)
  ASSERT_EQ(kernel_op->pre_args[18].total_size(), sizeof(unsigned int));
  const unsigned int *maxSinglePairs = reinterpret_cast<const unsigned int *>(
      kernel_op->pre_args[18].data.data());
  EXPECT_EQ(*maxSinglePairs, N) << "maxSinglePairs mismatch";

  // 20. singlePairs (int2*)
  ASSERT_EQ(kernel_op->pre_args[19].total_size(), N * sizeof(int2));
  const int2 *singlePairs =
      reinterpret_cast<const int2 *>(kernel_op->pre_args[19].data.data());
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(singlePairs[i].x, i) << "singlePairs[" << i << "].x mismatch";
    EXPECT_EQ(singlePairs[i].y, i + 2) << "singlePairs[" << i << "].y mismatch";
  }

  // 21. global_nonbonded0_sigmaEpsilon (float2*)
  ASSERT_EQ(kernel_op->pre_args[20].total_size(), N * sizeof(float2));
  const float2 *sigmaEpsilon =
      reinterpret_cast<const float2 *>(kernel_op->pre_args[20].data.data());
  for (int i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(sigmaEpsilon[i].x, 0.5f + i)
        << "sigmaEpsilon[" << i << "].x mismatch";
    EXPECT_FLOAT_EQ(sigmaEpsilon[i].y, 1.0f + i)
        << "sigmaEpsilon[" << i << "].y mismatch";
  }
}

// Test typeSub function for type substitutions
TEST_F(InterceptorTest, TypeSubstitution) {
    // Test case 1: Basic typedef substitution
    std::string source1 = R"(
typedef int MyInt;
MyInt x = 5;
)";
    std::string expected1 = R"(

int x = 5;
)";
    EXPECT_EQ(typeSub(source1), expected1);

    // Test case 2: Using declaration substitution
    std::string source2 = R"(
using MyFloat = float;
MyFloat y = 3.14;
)";
    std::string expected2 = R"(

float y = 3.14;
)";
    EXPECT_EQ(typeSub(source2), expected2);

    // Test case 3: #define substitution
    std::string source3 = R"(
#define CUSTOM_TYPE double
CUSTOM_TYPE z = 2.718;
)";
    std::string expected3 = R"(

double z = 2.718;
)";
    EXPECT_EQ(typeSub(source3), expected3);

    // Test case 4: Chained typedef substitution
    std::string source4 = R"(
typedef int BaseType;
typedef BaseType IntermediateType;
typedef IntermediateType FinalType;
FinalType value = 42;
)";
    std::string expected4 = R"(



int value = 42;
)";
    EXPECT_EQ(typeSub(source4), expected4);

    // Test case 5: Complex type with multiple substitutions
    std::string source5 = R"(
typedef unsigned int uint;
typedef uint* uint_ptr;
using IntPtr = int*;
#define PTR_TYPE IntPtr
uint_ptr x = nullptr;
PTR_TYPE y = nullptr;
)";
    std::string expected5 = R"(



unsigned int* x = nullptr;
int* y = nullptr;
)";
    EXPECT_EQ(typeSub(source5), expected5);
}
