cmake_minimum_required(VERSION 3.20)

# The test suite is split into five binaries by purpose so CI can run, and label,
# each category independently.
#
#   unit        -- contract tests: each kernel does what its documentation says
#   numerical   -- accuracy vs. the scalar oracle over large random inputs
#   edge_cases  -- empty / single / odd-length / non-square / invalid arguments
#   concurrency -- thread counts, repetition, reduction correctness, data races
#   regression  -- defects found during development, kept as executable proof

set(PCL_TEST_SUITES unit numerical edge_cases concurrency regression)

function(pcl_add_test_suite suite)
  set(sources ${ARGN})
  add_executable(${suite}_tests ${sources})
  target_link_libraries(${suite}_tests PRIVATE
    parallel_compute
    GTest::gtest_main
    pcl_warnings)
  target_compile_options(${suite}_tests PRIVATE ${PCL_ISA_FLAGS})
  target_include_directories(${suite}_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  include(GoogleTest)
  gtest_discover_tests(${suite}_tests
    TEST_PREFIX "${suite}/"
    PROPERTIES LABELS "${suite}"
    DISCOVERY_TIMEOUT 60)
endfunction()

pcl_add_test_suite(unit
  unit/test_vector_ops.cpp
  unit/test_reductions.cpp
  unit/test_matmul.cpp
  unit/test_memory_workloads.cpp
  unit/test_scheduler.cpp
  unit/test_simd_kernels.cpp)

pcl_add_test_suite(numerical
  numerical/test_numerical_accuracy.cpp)

pcl_add_test_suite(edge_cases
  edge_cases/test_edge_cases.cpp)

pcl_add_test_suite(concurrency
  concurrency/test_parallel_correctness.cpp)

pcl_add_test_suite(regression
  regression/test_regression.cpp)
