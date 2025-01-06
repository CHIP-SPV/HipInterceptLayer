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

target_link_libraries(unit_tests
  PRIVATE
  GTest::GTest
  GTest::Main
  HIPIntercept
)

function(add_hip_trace_target TARGET_NAME SOURCE_FILE)
    # Create the executable
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    
    # Add compiler flags to preserve symbols
    target_compile_options(${TARGET_NAME} PRIVATE -g -O0 -w)
    
    # Add linker flags to preserve symbols
    set_target_properties(${TARGET_NAME} PROPERTIES 
        LINK_FLAGS "-Wl,--export-dynamic")
    
    # Link against HIP
    target_link_libraries(${TARGET_NAME}
        PRIVATE
        hip::device)

    # Set up tracing with all dependencies that affect trace generation
    add_custom_command(
        OUTPUT 
            ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-0.trace
            ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-0.log
            ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-0.trace.replay
        COMMAND ${CMAKE_COMMAND} -E env "LD_PRELOAD=$<TARGET_FILE:HIPIntercept>" "HIP_TRACE_LOCATION=${CMAKE_CURRENT_BINARY_DIR}" 
            $<TARGET_FILE:${TARGET_NAME}> > ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-0.log 2>&1
        COMMAND $<TARGET_FILE:HIPInterceptCompare> ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-0.trace > ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-0.trace.replay 2>&1
        DEPENDS 
            ${TARGET_NAME}
            HIPIntercept
            ${SOURCE_FILE}
            $<TARGET_FILE:HIPIntercept>
            $<TARGET_FILE:${TARGET_NAME}>
            $<TARGET_FILE:HIPInterceptCompare>
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running ${TARGET_NAME} with HIPIntercept preloaded and replaying trace"
    )

    add_custom_target(run_${TARGET_NAME}
        DEPENDS 
            ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-0.trace
            ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-0.log
            ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-0.trace.replay
    )
endfunction()

add_hip_trace_target(test_kernels data/test_kernels.cpp)
add_hip_trace_target(data_verification data/data_verification.cpp)
add_hip_trace_target(computeNonbondedRepro data/computeNonbondedRepro.cpp)

add_dependencies(unit_tests run_test_kernels run_data_verification)
gtest_discover_tests(unit_tests)