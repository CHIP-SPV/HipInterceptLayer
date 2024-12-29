# Add test executable
add_executable(unit_tests
  ../Tracer.cc
  testKernelManager.cpp 
  testCodeGen.cpp
  testComparator.cpp
  testInterceptor.cpp
)

target_link_libraries(unit_tests
  PRIVATE
  hip::device
)

# Add compile definition for build directory path
target_compile_definitions(unit_tests PRIVATE
  CMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}"
)

# Compile test kernels
add_executable(test_kernels data/test_kernels.cpp)

# Add compiler flags to preserve symbols
target_compile_options(test_kernels PRIVATE -g)

# Add linker flags to preserve symbols
set_target_properties(test_kernels PROPERTIES 
    LINK_FLAGS "-Wl,--export-dynamic")

target_link_libraries(test_kernels
  PRIVATE
  hip::device)

add_executable(data_verification data/data_verification.cpp)

target_link_libraries(data_verification
  PRIVATE
  hip::device)

target_link_libraries(unit_tests
  PRIVATE
  GTest::GTest
  GTest::Main
  HIPIntercept
)

# Add custom target to run test_kernels with LD_PRELOAD
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test_kernels-0.trace
    COMMAND ${CMAKE_COMMAND} -E env "LD_PRELOAD=$<TARGET_FILE:HIPIntercept>" "HIP_TRACE_LOCATION=${CMAKE_CURRENT_BINARY_DIR}" $<TARGET_FILE:test_kernels>
    DEPENDS test_kernels HIPIntercept
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running test_kernels with HIPIntercept preloaded"
)

add_custom_target(run_test_kernels
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/test_kernels-0.trace
)

# Add custom command to generate data_verification trace
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/data_verification-0.trace
    COMMAND ${CMAKE_COMMAND} -E env "LD_PRELOAD=$<TARGET_FILE:HIPIntercept>" "HIP_TRACE_LOCATION=${CMAKE_CURRENT_BINARY_DIR}" $<TARGET_FILE:data_verification>
    DEPENDS data_verification HIPIntercept
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running data_verification with HIPIntercept preloaded"
)

add_custom_target(run_data_verification
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/data_verification-0.trace
)

add_dependencies(unit_tests run_test_kernels run_data_verification)
gtest_discover_tests(unit_tests)