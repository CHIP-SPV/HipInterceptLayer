cmake_minimum_required(VERSION 3.10)
project(HIPInterceptLayer VERSION 1.0 LANGUAGES CXX)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Require hipcc compiler
if(NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER hipcc)
elseif(NOT CMAKE_CXX_COMPILER MATCHES ".*hipcc$")
  message(FATAL_ERROR "HIP Intercept Layer requires hipcc compiler. Current compiler: ${CMAKE_CXX_COMPILER}")
endif()

# Configure header file
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/config.hh.in
  ${CMAKE_CURRENT_BINARY_DIR}/config.hh
  @ONLY
)

add_compile_options(-g)

# Find HIP package and set up include paths
find_package(HIP CONFIG REQUIRED)
find_package(GTest REQUIRED)

message(STATUS "HIP_DIR: ${HIP_DIR}")
message(STATUS "HIP_LIBRARIES: ${HIP_LIBRARIES}")
message(STATUS "HIP_INCLUDE_DIRS: ${HIP_INCLUDE_DIRS}")

include_directories(${HIP_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})  # For config.hh

# Define BUILD_TIME for build-time path resolution
add_compile_definitions(BUILD_TIME)

# Build libraries and executable
add_library(HIPIntercept SHARED Interceptor.cc Tracer.cc)
add_executable(HIPInterceptCompare Comparator.cc Tracer.cc)
target_link_libraries(HIPIntercept INTERFACE hip::host)
target_link_libraries(HIPInterceptCompare INTERFACE hip::host)
enable_testing()
add_subdirectory(tests)
add_dependencies(tests HIPIntercept HIPInterceptCompare)

# Installation rules
include(GNUInstallDirs)

install(TARGETS HIPIntercept HIPInterceptCompare
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES 
    Interceptor.hh
    Util.hh
    Tracer.hh
    Comparator.hh
    CodeGenKernelHeaders.hh
    ${CMAKE_CURRENT_BINARY_DIR}/config.hh
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hip-intercept
)
