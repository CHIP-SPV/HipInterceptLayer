#include "Comparator.hh"
#include <gtest/gtest.h>
#include <fstream>

TEST(ComparatorTest, CompareKernelExecutions) {
    // Create test arguments
    std::vector<ArgState> pre_args1;
    std::vector<ArgState> post_args1;
    std::vector<ArgState> pre_args2;
    std::vector<ArgState> post_args2;

    // Create identical scalar arguments
    ArgState scalar_arg1(sizeof(int), 1);
    ArgState scalar_arg2(sizeof(int), 1);
    int scalar_value = 42;
    std::memcpy(scalar_arg1.data.data(), &scalar_value, sizeof(int));
    std::memcpy(scalar_arg2.data.data(), &scalar_value, sizeof(int));
    pre_args1.push_back(scalar_arg1);
    pre_args2.push_back(scalar_arg2);

    // Create identical array arguments
    ArgState array_arg1(sizeof(float), 4);
    ArgState array_arg2(sizeof(float), 4);
    float array_values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::memcpy(array_arg1.data.data(), array_values, sizeof(float) * 4);
    std::memcpy(array_arg2.data.data(), array_values, sizeof(float) * 4);
    pre_args1.push_back(array_arg1);
    pre_args2.push_back(array_arg2);

    // Create two identical kernel executions
    KernelExecution k1(
        pre_args1,
        post_args1,
        nullptr,
        "test_kernel",
        dim3(1, 1, 1),
        dim3(1, 1, 1),
        0,
        nullptr
    );

    KernelExecution k2(
        pre_args2,
        post_args2,
        nullptr,
        "test_kernel",
        dim3(1, 1, 1),
        dim3(1, 1, 1),
        0,
        nullptr
    );

    Comparator comparator;
    ASSERT_TRUE(comparator.compare(k1, k2));
}

TEST(ComparatorTest, CompareKernelExecutionsDifferentArgs) {
    // Create test arguments with different values
    std::vector<ArgState> pre_args1;
    std::vector<ArgState> post_args1;
    std::vector<ArgState> pre_args2;
    std::vector<ArgState> post_args2;

    // Create scalar arguments with different values
    ArgState scalar_arg1(sizeof(int), 1);
    ArgState scalar_arg2(sizeof(int), 1);
    int scalar_value1 = 42;
    int scalar_value2 = 43;
    std::memcpy(scalar_arg1.data.data(), &scalar_value1, sizeof(int));
    std::memcpy(scalar_arg2.data.data(), &scalar_value2, sizeof(int));
    pre_args1.push_back(scalar_arg1);
    pre_args2.push_back(scalar_arg2);

    // Create kernel executions with different arguments
    KernelExecution k1(
        pre_args1,
        post_args1,
        nullptr,
        "test_kernel",
        dim3(1, 1, 1),
        dim3(1, 1, 1),
        0,
        nullptr
    );

    KernelExecution k2(
        pre_args2,
        post_args2,
        nullptr,
        "test_kernel",
        dim3(1, 1, 1),
        dim3(1, 1, 1),
        0,
        nullptr
    );

    Comparator comparator;
    ASSERT_FALSE(comparator.compare(k1, k2));
}

TEST(ComparatorTest, CompareMemoryOperations) {
    // Create test arguments
    std::vector<ArgState> pre_args1;
    std::vector<ArgState> post_args1;
    std::vector<ArgState> pre_args2;
    std::vector<ArgState> post_args2;

    // Create identical memory arguments
    ArgState mem_arg1(sizeof(char), 1024);
    ArgState mem_arg2(sizeof(char), 1024);
    for (size_t i = 0; i < 1024; i++) {
        mem_arg1.data[i] = static_cast<char>(i & 0xFF);
        mem_arg2.data[i] = static_cast<char>(i & 0xFF);
    }
    pre_args1.push_back(mem_arg1);
    pre_args2.push_back(mem_arg2);

    // Create two identical memory operations
    MemoryOperation m1(
        pre_args1,
        post_args1,
        MemoryOpType::COPY,
        nullptr,
        nullptr,
        1024,
        0,
        hipMemcpyHostToDevice,
        nullptr
    );

    MemoryOperation m2(
        pre_args2,
        post_args2,
        MemoryOpType::COPY,
        nullptr,
        nullptr,
        1024,
        0,
        hipMemcpyHostToDevice,
        nullptr
    );

    Comparator comparator;
    ASSERT_TRUE(comparator.compare(m1, m2));
}

TEST(ComparatorTest, CompareMemoryOperationsDifferentData) {
    // Create test arguments with different data
    std::vector<ArgState> pre_args1;
    std::vector<ArgState> post_args1;
    std::vector<ArgState> pre_args2;
    std::vector<ArgState> post_args2;

    // Create memory arguments with different data
    ArgState mem_arg1(sizeof(char), 1024);
    ArgState mem_arg2(sizeof(char), 1024);
    for (size_t i = 0; i < 1024; i++) {
        mem_arg1.data[i] = static_cast<char>(i & 0xFF);
        mem_arg2.data[i] = static_cast<char>((i + 1) & 0xFF);  // Different pattern
    }
    pre_args1.push_back(mem_arg1);
    pre_args2.push_back(mem_arg2);

    // Create memory operations with different data
    MemoryOperation m1(
        pre_args1,
        post_args1,
        MemoryOpType::COPY,
        nullptr,
        nullptr,
        1024,
        0,
        hipMemcpyHostToDevice,
        nullptr
    );

    MemoryOperation m2(
        pre_args2,
        post_args2,
        MemoryOpType::COPY,
        nullptr,
        nullptr,
        1024,
        0,
        hipMemcpyHostToDevice,
        nullptr
    );

    Comparator comparator;
    ASSERT_FALSE(comparator.compare(m1, m2));
}
