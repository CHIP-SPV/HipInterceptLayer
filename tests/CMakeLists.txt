# Add test executable
add_executable(unit_tests
  ../Tracer.cc
  testKernelManager.cpp 
  testCodeGen.cpp
)

# Add compile definition for build directory path
target_compile_definitions(unit_tests PRIVATE
  CMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}"
)

# Compile test kernels
add_executable(test_kernels data/Kernels.cpp)

# Add compiler flags to preserve symbols
target_compile_options(test_kernels PRIVATE -g)

# Add linker flags to preserve symbols
set_target_properties(test_kernels PROPERTIES 
    LINK_FLAGS "-Wl,--export-dynamic")

target_link_libraries(test_kernels
  PRIVATE
  hip::device)

target_link_libraries(unit_tests
  PRIVATE
  GTest::GTest
  GTest::Main
  HIPIntercept
)

# Add custom target to run test_kernels with LD_PRELOAD
add_custom_target(run_test_kernels
    COMMAND ${CMAKE_COMMAND} -E remove -f test_kernels-0.trace
    COMMAND ${CMAKE_COMMAND} -E env "LD_PRELOAD=$<TARGET_FILE:HIPIntercept>" "HIP_TRACE_LOCATION=${CMAKE_CURRENT_BINARY_DIR}" $<TARGET_FILE:test_kernels>
    DEPENDS test_kernels HIPIntercept
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running test_kernels with HIPIntercept preloaded"
)

# Make test_kernels.trace available for unit tests
add_dependencies(unit_tests run_test_kernels)

# Make sure run_test_kernels is built by default
add_custom_target(all_tests ALL DEPENDS run_test_kernels)

# Make sure the trace file exists before running tests
add_dependencies(unit_tests run_test_kernels)

gtest_discover_tests(unit_tests)