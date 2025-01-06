#include <gtest/gtest.h>
#include "../Interceptor.hh"
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
    std::string trace_file = std::string(CMAKE_BINARY_DIR) + "/tests/vectorAdd-0.trace";
    Tracer tracer(trace_file);
    
    // Find the kernel operation by name
    auto kernel_indices = tracer.getOperationsIdxByName("vectorIncrementKernel");
    ASSERT_FALSE(kernel_indices.empty()) << "No vectorIncrementKernel found in trace";
    
    // Get the first instance of the kernel
    auto kernel_op = std::dynamic_pointer_cast<KernelExecution>(tracer.getOperation(kernel_indices[0]));
    ASSERT_NE(kernel_op, nullptr);
    std::cout << "Found kernel operation at index " << kernel_indices[0] << std::endl;
    
    // Add debug output for kernel operation details
    std::cout << "Kernel operation details:" << std::endl;
    std::cout << "  Name: " << kernel_op->kernel_name << std::endl;
    std::cout << "  Grid dim: " << kernel_op->grid_dim.x << "," << kernel_op->grid_dim.y << "," << kernel_op->grid_dim.z << std::endl;
    std::cout << "  Block dim: " << kernel_op->block_dim.x << "," << kernel_op->block_dim.y << "," << kernel_op->block_dim.z << std::endl;
    std::cout << "  Shared mem: " << kernel_op->shared_mem << std::endl;
    std::cout << "  Number of arguments: " << kernel_op->arg_ptrs.size() << std::endl;
    std::cout << "  Number of scalar values: " << kernel_op->scalar_values.size() << std::endl;
    
    // Verify kernel launch parameters
    EXPECT_EQ(kernel_op->kernel_name, "vectorIncrementKernel");
    EXPECT_EQ(kernel_op->block_dim.x, 256);  // From vectorAdd.cpp
    EXPECT_EQ(kernel_op->block_dim.y, 1);
    EXPECT_EQ(kernel_op->block_dim.z, 1);
    EXPECT_EQ(kernel_op->shared_mem, 0);

    // Verify scalar values
    ASSERT_EQ(kernel_op->scalar_values.size(), 2) << "Expected 2 scalar values (float4 and float)";
    
    // First scalar value should be float4 (16 bytes)
    ASSERT_EQ(kernel_op->scalar_values[0].size(), 16) << "Expected float4 to be 16 bytes";
    const float4* input_vec4 = reinterpret_cast<const float4*>(kernel_op->scalar_values[0].data());
    std::cout << "Input float4: (" << input_vec4->x << ", " << input_vec4->y << ", " 
              << input_vec4->z << ", " << input_vec4->w << ")" << std::endl;
    
    // Second scalar value should be float (4 bytes)
    ASSERT_EQ(kernel_op->scalar_values[1].size(), 8) << "Expected float to be 8 bytes";
    const float* input_scalar = reinterpret_cast<const float*>(kernel_op->scalar_values[1].data());
    std::cout << "Input scalar: " << *input_scalar << std::endl;

    // Expected values from vectorAdd.cpp
    const float4 expected_input_vec4 = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    const float expected_input_scalar = 0.5f;

    // Verify input values
    EXPECT_FLOAT_EQ(input_vec4->x, expected_input_vec4.x);
    EXPECT_FLOAT_EQ(input_vec4->y, expected_input_vec4.y);
    EXPECT_FLOAT_EQ(input_vec4->z, expected_input_vec4.z);
    EXPECT_FLOAT_EQ(input_vec4->w, expected_input_vec4.w);
    EXPECT_FLOAT_EQ(*input_scalar, expected_input_scalar);

    // Pre-execution state
    auto pre_state = kernel_op->pre_state;
    std::cout << "Pre-state total_size: " << pre_state.total_size << std::endl;
    std::cout << "Pre-state chunks size: " << pre_state.chunks.size() << std::endl;
    ASSERT_GT(pre_state.total_size, 0);
    ASSERT_GT(pre_state.chunks.size(), 0) << "No chunks in pre_state";

    // Get raw data from the first chunk of pre_state
    const auto& pre_chunk = pre_state.chunks[0];
    const float4* pre_vec4 = reinterpret_cast<const float4*>(pre_chunk.data.get());
    const float* pre_scalar = reinterpret_cast<const float*>(pre_vec4 + 1024);  // N=1024

    // Print first few pre-state values
    std::cout << "\nFirst few pre-state values:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "  Index " << i << ": vec4=("
                  << pre_vec4[i].x << ", " 
                  << pre_vec4[i].y << ", "
                  << pre_vec4[i].z << ", "
                  << pre_vec4[i].w << "), "
                  << "scalar=" << pre_scalar[i] << std::endl;
    }

    // Verify all pre-state values (should be uninitialized or zero)
    for (int i = 0; i < 1024; i++) {
        // We don't verify exact values since they're uninitialized
        // but we can verify they exist and are accessible
        volatile float x = pre_vec4[i].x;
        volatile float y = pre_vec4[i].y;
        volatile float z = pre_vec4[i].z;
        volatile float w = pre_vec4[i].w;
        volatile float s = pre_scalar[i];
        (void)x; (void)y; (void)z; (void)w; (void)s;  // Prevent unused variable warnings
    }

    // Post-execution state
    auto post_state = kernel_op->post_state;
    std::cout << "Post-state total_size: " << post_state.total_size << std::endl;
    std::cout << "Post-state chunks size: " << post_state.chunks.size() << std::endl;
    ASSERT_GT(post_state.total_size, 0);
    ASSERT_GT(post_state.chunks.size(), 0) << "No chunks in post_state";

    // Get raw data from the first chunk of post_state
    const auto& post_chunk = post_state.chunks[0];
    const float4* output_vec4 = reinterpret_cast<const float4*>(post_chunk.data.get());
    const float* output_scalar = reinterpret_cast<const float*>(output_vec4 + 1024);  // N=1024

    // Expected output values
    const float4 expected_output_vec4 = make_float4(
        expected_input_vec4.x + expected_input_scalar,
        expected_input_vec4.y + expected_input_scalar,
        expected_input_vec4.z + expected_input_scalar,
        expected_input_vec4.w + expected_input_scalar
    );
    const float expected_output_scalar = expected_input_scalar * 2.0f;

    // Print first few output values
    std::cout << "\nFirst few post-state values:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "  Index " << i << ": vec4=("
                  << output_vec4[i].x << ", " 
                  << output_vec4[i].y << ", "
                  << output_vec4[i].z << ", "
                  << output_vec4[i].w << "), "
                  << "scalar=" << output_scalar[i] << std::endl;
    }

    std::cout << "\nExpected output vec4: ("
              << expected_output_vec4.x << ", "
              << expected_output_vec4.y << ", "
              << expected_output_vec4.z << ", "
              << expected_output_vec4.w << ")" << std::endl;
    std::cout << "Expected output scalar: " << expected_output_scalar << std::endl;

    // Verify all output values
    for (int i = 0; i < 1024; i++) {
        // Verify vec4 components
        EXPECT_FLOAT_EQ(output_vec4[i].x, expected_output_vec4.x)
            << "Vec4.x mismatch at index " << i;
        EXPECT_FLOAT_EQ(output_vec4[i].y, expected_output_vec4.y)
            << "Vec4.y mismatch at index " << i;
        EXPECT_FLOAT_EQ(output_vec4[i].z, expected_output_vec4.z)
            << "Vec4.z mismatch at index " << i;
        EXPECT_FLOAT_EQ(output_vec4[i].w, expected_output_vec4.w)
            << "Vec4.w mismatch at index " << i;

        // Verify scalar value
        EXPECT_FLOAT_EQ(output_scalar[i], expected_output_scalar)
            << "Scalar mismatch at index " << i;
    }
}

// Test to verify computeNonbonded kernel behavior from trace
TEST_F(InterceptorTest, CompareComputeNonbondedReproTraceWithSource) {
    // Load and verify the trace file
    std::string trace_file = std::string(CMAKE_BINARY_DIR) + "/tests/computeNonbondedRepro-0.trace";
    Tracer tracer(trace_file);
    
    // Find the kernel operation by name
    auto kernel_indices = tracer.getOperationsIdxByName("computeNonbonded");
    ASSERT_FALSE(kernel_indices.empty()) << "No computeNonbonded kernel found in trace";
    
    // Get the first instance of the kernel
    auto kernel_op = std::dynamic_pointer_cast<KernelExecution>(tracer.getOperation(kernel_indices[0]));
    ASSERT_NE(kernel_op, nullptr);
    std::cout << "Found kernel operation at index " << kernel_indices[0] << std::endl;
    
    // Add debug output for kernel operation details
    std::cout << "Kernel operation details:" << std::endl;
    std::cout << "  Name: " << kernel_op->kernel_name << std::endl;
    std::cout << "  Grid dim: " << kernel_op->grid_dim.x << "," << kernel_op->grid_dim.y << "," << kernel_op->grid_dim.z << std::endl;
    std::cout << "  Block dim: " << kernel_op->block_dim.x << "," << kernel_op->block_dim.y << "," << kernel_op->block_dim.z << std::endl;
    std::cout << "  Shared mem: " << kernel_op->shared_mem << std::endl;
    std::cout << "  Number of arguments: " << kernel_op->arg_ptrs.size() << std::endl;
    std::cout << "  Number of scalar values: " << kernel_op->scalar_values.size() << std::endl;
    
    // Verify kernel launch parameters
    EXPECT_EQ(kernel_op->kernel_name, "computeNonbonded");
    EXPECT_EQ(kernel_op->grid_dim.x, 2);
    EXPECT_EQ(kernel_op->grid_dim.y, 1);
    EXPECT_EQ(kernel_op->grid_dim.z, 1);
    EXPECT_EQ(kernel_op->block_dim.x, 64);
    EXPECT_EQ(kernel_op->block_dim.y, 1);
    EXPECT_EQ(kernel_op->block_dim.z, 1);
    EXPECT_EQ(kernel_op->shared_mem, 0);

    // Verify scalar arguments
    const float4 expected_periodic_box_size = make_float4(10.0f, 10.0f, 10.0f, 0.0f);
    const float4 expected_inv_periodic_box_size = make_float4(0.1f, 0.1f, 0.1f, 0.0f);
    const float4 expected_periodic_box_vec_x = make_float4(10.0f, 0.0f, 0.0f, 0.0f);
    const float4 expected_periodic_box_vec_y = make_float4(0.0f, 10.0f, 0.0f, 0.0f);
    const float4 expected_periodic_box_vec_z = make_float4(0.0f, 0.0f, 10.0f, 0.0f);
    const unsigned int expected_max_tiles = 128u;
    const unsigned int expected_max_single_pairs = 128u;
    const unsigned int expected_start_tile_index = 0u;
    const unsigned long long expected_num_tile_indices = 128ull;

    // Print and verify scalar values
    std::cout << "\nScalar values:" << std::endl;
    for (size_t i = 0; i < kernel_op->scalar_values.size(); i++) {
        std::cout << "  Value " << i << ": ";
        switch(i) {
            case 0: // startTileIndex (unsigned int)
                {
                    unsigned int val = *reinterpret_cast<const unsigned int*>(kernel_op->scalar_values[i].data());
                    std::cout << "startTileIndex=" << val;
                    EXPECT_EQ(val, expected_start_tile_index) << "startTileIndex should be " << expected_start_tile_index;
                }
                break;
            case 1: // numTileIndices (unsigned long long)
                {
                    unsigned long long val = *reinterpret_cast<const unsigned long long*>(kernel_op->scalar_values[i].data());
                    std::cout << "numTileIndices=" << val;
                    EXPECT_EQ(val, expected_num_tile_indices) << "numTileIndices should be " << expected_num_tile_indices;
                }
                break;
            case 2: // periodicBoxSize (float4)
                {
                    const float4* vec = reinterpret_cast<const float4*>(kernel_op->scalar_values[i].data());
                    std::cout << "periodicBoxSize=(" << vec->x << ", " << vec->y << ", " 
                             << vec->z << ", " << vec->w << ")";
                    EXPECT_FLOAT_EQ(vec->x, expected_periodic_box_size.x);
                    EXPECT_FLOAT_EQ(vec->y, expected_periodic_box_size.y);
                    EXPECT_FLOAT_EQ(vec->z, expected_periodic_box_size.z);
                    EXPECT_FLOAT_EQ(vec->w, expected_periodic_box_size.w);
                }
                break;
            case 3: // invPeriodicBoxSize (float4)
                {
                    const float4* vec = reinterpret_cast<const float4*>(kernel_op->scalar_values[i].data());
                    std::cout << "invPeriodicBoxSize=(" << vec->x << ", " << vec->y << ", " 
                             << vec->z << ", " << vec->w << ")";
                    EXPECT_FLOAT_EQ(vec->x, expected_inv_periodic_box_size.x);
                    EXPECT_FLOAT_EQ(vec->y, expected_inv_periodic_box_size.y);
                    EXPECT_FLOAT_EQ(vec->z, expected_inv_periodic_box_size.z);
                    EXPECT_FLOAT_EQ(vec->w, expected_inv_periodic_box_size.w);
                }
                break;
            case 4: // periodicBoxVecX (float4)
                {
                    const float4* vec = reinterpret_cast<const float4*>(kernel_op->scalar_values[i].data());
                    std::cout << "periodicBoxVecX=(" << vec->x << ", " << vec->y << ", " 
                             << vec->z << ", " << vec->w << ")";
                    EXPECT_FLOAT_EQ(vec->x, expected_periodic_box_vec_x.x);
                    EXPECT_FLOAT_EQ(vec->y, expected_periodic_box_vec_x.y);
                    EXPECT_FLOAT_EQ(vec->z, expected_periodic_box_vec_x.z);
                    EXPECT_FLOAT_EQ(vec->w, expected_periodic_box_vec_x.w);
                }
                break;
            case 5: // periodicBoxVecY (float4)
                {
                    const float4* vec = reinterpret_cast<const float4*>(kernel_op->scalar_values[i].data());
                    std::cout << "periodicBoxVecY=(" << vec->x << ", " << vec->y << ", " 
                             << vec->z << ", " << vec->w << ")";
                    EXPECT_FLOAT_EQ(vec->x, expected_periodic_box_vec_y.x);
                    EXPECT_FLOAT_EQ(vec->y, expected_periodic_box_vec_y.y);
                    EXPECT_FLOAT_EQ(vec->z, expected_periodic_box_vec_y.z);
                    EXPECT_FLOAT_EQ(vec->w, expected_periodic_box_vec_y.w);
                }
                break;
            case 6: // periodicBoxVecZ (float4)
                {
                    const float4* vec = reinterpret_cast<const float4*>(kernel_op->scalar_values[i].data());
                    std::cout << "periodicBoxVecZ=(" << vec->x << ", " << vec->y << ", " 
                             << vec->z << ", " << vec->w << ")";
                    EXPECT_FLOAT_EQ(vec->x, expected_periodic_box_vec_z.x);
                    EXPECT_FLOAT_EQ(vec->y, expected_periodic_box_vec_z.y);
                    EXPECT_FLOAT_EQ(vec->z, expected_periodic_box_vec_z.z);
                    EXPECT_FLOAT_EQ(vec->w, expected_periodic_box_vec_z.w);
                }
                break;
            case 7: // maxTiles (unsigned int)
                {
                    unsigned int val = *reinterpret_cast<const unsigned int*>(kernel_op->scalar_values[i].data());
                    std::cout << "maxTiles=" << val;
                    EXPECT_EQ(val, expected_max_tiles) << "maxTiles should be " << expected_max_tiles;
                }
                break;
            case 8: // maxSinglePairs (unsigned int)
                {
                    unsigned int val = *reinterpret_cast<const unsigned int*>(kernel_op->scalar_values[i].data());
                    std::cout << "maxSinglePairs=" << val;
                    EXPECT_EQ(val, expected_max_single_pairs) << "maxSinglePairs should be " << expected_max_single_pairs;
                }
                break;
            default:
                std::cout << "unknown type";
        }
        std::cout << std::endl;
    }

    // Pre-execution state
    auto pre_state = kernel_op->pre_state;
    std::cout << "Pre-state total_size: " << pre_state.total_size << std::endl;
    std::cout << "Pre-state chunks size: " << pre_state.chunks.size() << std::endl;
    ASSERT_GT(pre_state.total_size, 0);
    ASSERT_GT(pre_state.chunks.size(), 0) << "No chunks in pre_state";

    // Get pointers to the different arrays in pre_state using the first chunk
    const auto& pre_chunk = pre_state.chunks[0];
    const char* pre_data = pre_chunk.data.get();
    const unsigned long long* force_buffers = reinterpret_cast<const unsigned long long*>(pre_data);
    const float* energy_buffer = reinterpret_cast<const float*>(force_buffers + 128);  // N=128
    const float4* posq = reinterpret_cast<const float4*>(energy_buffer + 128);
    const unsigned int* exclusions = reinterpret_cast<const unsigned int*>(posq + 128);
    const int2* exclusion_tiles = reinterpret_cast<const int2*>(exclusions + 128);
    const int* tiles = reinterpret_cast<const int*>(exclusion_tiles + 128);
    const unsigned int* interaction_count = reinterpret_cast<const unsigned int*>(tiles + 128);
    const float4* block_center = reinterpret_cast<const float4*>(interaction_count + 128);
    const float4* block_size = reinterpret_cast<const float4*>(block_center + 128);
    const unsigned int* interacting_atoms = reinterpret_cast<const unsigned int*>(block_size + 128);
    const int2* single_pairs = reinterpret_cast<const int2*>(interacting_atoms + 128);
    const float2* sigma_epsilon = reinterpret_cast<const float2*>(single_pairs + 128);

    // Print first few values of each array for debugging
    std::cout << "\nFirst few pre-state values:" << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << "  Index " << i << ":" << std::endl;
        std::cout << "    force_buffer: " << force_buffers[i] << std::endl;
        std::cout << "    energy_buffer: " << energy_buffer[i] << std::endl;
        std::cout << "    posq: (" << posq[i].x << ", " << posq[i].y << ", " 
                  << posq[i].z << ", " << posq[i].w << ")" << std::endl;
        std::cout << "    exclusions: " << exclusions[i] << std::endl;
        std::cout << "    exclusion_tiles: (" << exclusion_tiles[i].x << ", " 
                  << exclusion_tiles[i].y << ")" << std::endl;
        std::cout << "    tiles: " << tiles[i] << std::endl;
        std::cout << "    interaction_count: " << interaction_count[i] << std::endl;
        std::cout << "    block_center: (" << block_center[i].x << ", " << block_center[i].y << ", "
                  << block_center[i].z << ", " << block_center[i].w << ")" << std::endl;
        std::cout << "    block_size: (" << block_size[i].x << ", " << block_size[i].y << ", "
                  << block_size[i].z << ", " << block_size[i].w << ")" << std::endl;
        std::cout << "    interacting_atoms: " << interacting_atoms[i] << std::endl;
        std::cout << "    single_pairs: (" << single_pairs[i].x << ", " << single_pairs[i].y << ")" << std::endl;
        std::cout << "    sigma_epsilon: (" << sigma_epsilon[i].x << ", " << sigma_epsilon[i].y << ")" << std::endl;
    }

    // Verify initial values based on computeNonbondedRepro.cpp initialization
    for (int i = 0; i < 128; i++) {
        // Check force buffers: initialized to i
        EXPECT_EQ(force_buffers[i], static_cast<unsigned long long>(i))
            << "Force buffer mismatch at index " << i;

        // Check energy buffer: initialized to (float)i
        EXPECT_FLOAT_EQ(energy_buffer[i], static_cast<float>(i))
            << "Energy buffer mismatch at index " << i;

        // Check posq: initialized to make_float4(i, i+0.1f, i+0.2f, i+0.3f)
        EXPECT_FLOAT_EQ(posq[i].x, static_cast<float>(i))
            << "posq.x mismatch at index " << i;
        EXPECT_FLOAT_EQ(posq[i].y, static_cast<float>(i) + 0.1f)
            << "posq.y mismatch at index " << i;
        EXPECT_FLOAT_EQ(posq[i].z, static_cast<float>(i) + 0.2f)
            << "posq.z mismatch at index " << i;
        EXPECT_FLOAT_EQ(posq[i].w, static_cast<float>(i) + 0.3f)
            << "posq.w mismatch at index " << i;

        // Check exclusions: initialized to i
        EXPECT_EQ(exclusions[i], static_cast<unsigned int>(i))
            << "Exclusions mismatch at index " << i;

        // Check exclusion tiles: initialized to make_int2(i, i+1)
        EXPECT_EQ(exclusion_tiles[i].x, i)
            << "Exclusion tiles x mismatch at index " << i;
        EXPECT_EQ(exclusion_tiles[i].y, i + 1)
            << "Exclusion tiles y mismatch at index " << i;

        // Check tiles: initialized to i
        EXPECT_EQ(tiles[i], i)
            << "Tiles mismatch at index " << i;

        // Check interaction count: initialized to i
        EXPECT_EQ(interaction_count[i], static_cast<unsigned int>(i))
            << "Interaction count mismatch at index " << i;

        // Check block center: initialized to make_float4(i+0.5f, i+0.6f, i+0.7f, i+0.8f)
        EXPECT_FLOAT_EQ(block_center[i].x, i + 0.5f)
            << "Block center x mismatch at index " << i;
        EXPECT_FLOAT_EQ(block_center[i].y, i + 0.6f)
            << "Block center y mismatch at index " << i;
        EXPECT_FLOAT_EQ(block_center[i].z, i + 0.7f)
            << "Block center z mismatch at index " << i;
        EXPECT_FLOAT_EQ(block_center[i].w, i + 0.8f)
            << "Block center w mismatch at index " << i;

        // Check block size: initialized to make_float4(1.0f, 1.0f, 1.0f, 1.0f)
        EXPECT_FLOAT_EQ(block_size[i].x, 1.0f)
            << "Block size x mismatch at index " << i;
        EXPECT_FLOAT_EQ(block_size[i].y, 1.0f)
            << "Block size y mismatch at index " << i;
        EXPECT_FLOAT_EQ(block_size[i].z, 1.0f)
            << "Block size z mismatch at index " << i;
        EXPECT_FLOAT_EQ(block_size[i].w, 1.0f)
            << "Block size w mismatch at index " << i;

        // Check interacting atoms: initialized to i
        EXPECT_EQ(interacting_atoms[i], static_cast<unsigned int>(i))
            << "Interacting atoms mismatch at index " << i;

        // Check single pairs: initialized to make_int2(i, i+2)
        EXPECT_EQ(single_pairs[i].x, i)
            << "Single pairs x mismatch at index " << i;
        EXPECT_EQ(single_pairs[i].y, i + 2)
            << "Single pairs y mismatch at index " << i;

        // Check sigma epsilon: initialized to make_float2(0.5f + i, 1.0f + i)
        EXPECT_FLOAT_EQ(sigma_epsilon[i].x, 0.5f + i)
            << "Sigma epsilon x mismatch at index " << i;
        EXPECT_FLOAT_EQ(sigma_epsilon[i].y, 1.0f + i)
            << "Sigma epsilon y mismatch at index " << i;
    }

    // Post-execution state
    auto post_state = kernel_op->post_state;
    std::cout << "Post-state total_size: " << post_state.total_size << std::endl;
    std::cout << "Post-state chunks size: " << post_state.chunks.size() << std::endl;
    ASSERT_GT(post_state.total_size, 0);
    ASSERT_GT(post_state.chunks.size(), 0) << "No chunks in post_state";

    // Get pointers to the arrays in post_state using the first chunk
    const auto& post_chunk = post_state.chunks[0];
    const char* post_data = post_chunk.data.get();
    const unsigned long long* post_force_buffers = reinterpret_cast<const unsigned long long*>(post_data);
    const float* post_energy_buffer = reinterpret_cast<const float*>(post_force_buffers + 128);
    const float4* post_posq = reinterpret_cast<const float4*>(post_energy_buffer + 128);
    const tileflags* post_exclusions = reinterpret_cast<const tileflags*>(post_posq + 128);
    const int2* post_exclusion_tiles = reinterpret_cast<const int2*>(post_exclusions + 128);
    const int* post_tiles = reinterpret_cast<const int*>(post_exclusion_tiles + 128);
    const unsigned int* post_interaction_count = reinterpret_cast<const unsigned int*>(post_tiles + 128);
    const float4* post_block_center = reinterpret_cast<const float4*>(post_interaction_count + 128);
    const float4* post_block_size = reinterpret_cast<const float4*>(post_block_center + 128);
    const unsigned int* post_interacting_atoms = reinterpret_cast<const unsigned int*>(post_block_size + 128);
    const int2* post_single_pairs = reinterpret_cast<const int2*>(post_interacting_atoms + 128);
    const float2* post_sigma_epsilon = reinterpret_cast<const float2*>(post_single_pairs + 128);

    // Print first few post-state values for debugging
    std::cout << "\nFirst few post-state values:" << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << "  Index " << i << ":" << std::endl;
        std::cout << "    force_buffer: " << post_force_buffers[i] << std::endl;
        std::cout << "    energy_buffer: " << post_energy_buffer[i] << std::endl;
        std::cout << "    posq: (" << post_posq[i].x << ", " << post_posq[i].y << ", "
                  << post_posq[i].z << ", " << post_posq[i].w << ")" << std::endl;
        std::cout << "    exclusions: " << post_exclusions[i] << std::endl;
        std::cout << "    exclusion_tiles: (" << post_exclusion_tiles[i].x << ", "
                  << post_exclusion_tiles[i].y << ")" << std::endl;
        std::cout << "    tiles: " << post_tiles[i] << std::endl;
        std::cout << "    interaction_count: " << post_interaction_count[i] << std::endl;
        std::cout << "    block_center: (" << post_block_center[i].x << ", " << post_block_center[i].y << ", "
                  << post_block_center[i].z << ", " << post_block_center[i].w << ")" << std::endl;
        std::cout << "    block_size: (" << post_block_size[i].x << ", " << post_block_size[i].y << ", "
                  << post_block_size[i].z << ", " << post_block_size[i].w << ")" << std::endl;
        std::cout << "    interacting_atoms: " << post_interacting_atoms[i] << std::endl;
        std::cout << "    single_pairs: (" << post_single_pairs[i].x << ", " << post_single_pairs[i].y << ")" << std::endl;
        std::cout << "    sigma_epsilon: (" << post_sigma_epsilon[i].x << ", " << post_sigma_epsilon[i].y << ")" << std::endl;
    }

    // Verify all output values
    for (int i = 0; i < 128; i++) {
        // Verify force buffers
        EXPECT_EQ(post_force_buffers[i], static_cast<unsigned long long>(i + 1))
            << "Force buffer mismatch at index " << i;
        
        // Verify energy buffer
        EXPECT_FLOAT_EQ(post_energy_buffer[i], static_cast<float>(i + 1))
            << "Energy buffer mismatch at index " << i;
        
        // Verify tiles
        EXPECT_EQ(post_tiles[i], i + 1)
            << "Post tiles mismatch at index " << i;
        
        // Verify interaction count
        EXPECT_EQ(post_interaction_count[i], static_cast<unsigned int>(i + 1))
            << "Post interaction count mismatch at index " << i;

        // Verify interacting atoms
        EXPECT_EQ(post_interacting_atoms[i], static_cast<unsigned int>(i + 1))
            << "Post interacting atoms mismatch at index " << i;

        // Verify that other arrays remain unchanged
        // posq should remain unchanged
        EXPECT_FLOAT_EQ(post_posq[i].x, static_cast<float>(i))
            << "Post posq.x should remain unchanged at index " << i;
        EXPECT_FLOAT_EQ(post_posq[i].y, static_cast<float>(i) + 0.1f)
            << "Post posq.y should remain unchanged at index " << i;
        EXPECT_FLOAT_EQ(post_posq[i].z, static_cast<float>(i) + 0.2f)
            << "Post posq.z should remain unchanged at index " << i;
        EXPECT_FLOAT_EQ(post_posq[i].w, static_cast<float>(i) + 0.3f)
            << "Post posq.w should remain unchanged at index " << i;

        // block_center should remain unchanged
        EXPECT_FLOAT_EQ(post_block_center[i].x, i + 0.5f)
            << "Post block center x should remain unchanged at index " << i;
        EXPECT_FLOAT_EQ(post_block_center[i].y, i + 0.6f)
            << "Post block center y should remain unchanged at index " << i;
        EXPECT_FLOAT_EQ(post_block_center[i].z, i + 0.7f)
            << "Post block center z should remain unchanged at index " << i;
        EXPECT_FLOAT_EQ(post_block_center[i].w, i + 0.8f)
            << "Post block center w should remain unchanged at index " << i;

        // block_size should remain unchanged
        EXPECT_FLOAT_EQ(post_block_size[i].x, 1.0f)
            << "Post block size x should remain unchanged at index " << i;
        EXPECT_FLOAT_EQ(post_block_size[i].y, 1.0f)
            << "Post block size y should remain unchanged at index " << i;
        EXPECT_FLOAT_EQ(post_block_size[i].z, 1.0f)
            << "Post block size z should remain unchanged at index " << i;
        EXPECT_FLOAT_EQ(post_block_size[i].w, 1.0f)
            << "Post block size w should remain unchanged at index " << i;

        // single_pairs should remain unchanged
        EXPECT_EQ(post_single_pairs[i].x, i)
            << "Post single pairs x should remain unchanged at index " << i;
        EXPECT_EQ(post_single_pairs[i].y, i + 2)
            << "Post single pairs y should remain unchanged at index " << i;

        // sigma_epsilon should remain unchanged
        EXPECT_FLOAT_EQ(post_sigma_epsilon[i].x, 0.5f + i)
            << "Post sigma epsilon x should remain unchanged at index " << i;
        EXPECT_FLOAT_EQ(post_sigma_epsilon[i].y, 1.0f + i)
            << "Post sigma epsilon y should remain unchanged at index " << i;
    }
}