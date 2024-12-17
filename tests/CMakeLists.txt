# Add test executable
add_executable(unit_tests
  ../Tracer.cc
  testKernelManager.cpp 
  testCodeGen.cpp
)

# Compile test kernels
add_executable(test_kernels data/Kernels.cpp)

target_link_libraries(test_kernels
  PRIVATE
  hip::device)

target_link_libraries(unit_tests
  PRIVATE
  GTest::GTest
  GTest::Main
  HIPIntercept
)



enable_testing()
gtest_discover_tests(unit_tests)