add_custom_target(tests)

function(add_test_binary TARGET_NAME SOURCE_FILE)
    # Create the test executable with main source file and any additional sources
    add_executable(${TARGET_NAME} ${SOURCE_FILE} ${ARGN})
    
    # Link against GTest and HIP
    target_link_libraries(${TARGET_NAME}
        PRIVATE
        GTest::GTest
        GTest::Main
        hip::device
        HIPIntercept
    )

    # Add compile definition for build directory path
    target_compile_definitions(${TARGET_NAME} PRIVATE
        CMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}"
    )

    # Register the tests
    gtest_discover_tests(${TARGET_NAME})

    # Add this test to the general tests target
    add_dependencies(tests ${TARGET_NAME})
endfunction()

add_test_binary(testInterceptor testInterceptor.cpp ${CMAKE_SOURCE_DIR}/Tracer.cc)
add_test_binary(testKernelManager testKernelManager.cpp ${CMAKE_SOURCE_DIR}/Tracer.cc)

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
        COMMAND $<TARGET_FILE:HIPInterceptCompare> ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-0.trace --print-vals > ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-0.trace.replay 2>&1
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

    add_custom_target(run_${TARGET_NAME} ALL
        DEPENDS 
            ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-0.trace
            ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-0.log
            ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-0.trace.replay
    )

    # Add the run target to the general tests target instead of unit_tests
    add_dependencies(tests run_${TARGET_NAME})
endfunction()

add_hip_trace_target(test_kernels data/test_kernels.cpp)
add_hip_trace_target(data_verification data/data_verification.cpp)
add_hip_trace_target(vectorAdd data/vectorAdd.cpp)
add_hip_trace_target(computeNonbondedRepro data/computeNonbondedRepro.cpp)
