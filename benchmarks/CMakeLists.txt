cmake_minimum_required(VERSION 3.20)

add_executable(pcl_benchmarks
  benchmark_main.cpp
  support/bench_support.cpp
  kernels/vector_ops_bench.cpp
  kernels/reductions_bench.cpp
  kernels/matmul_bench.cpp
  workloads/memory_bench.cpp
  workloads/scheduling_bench.cpp
  baselines/blas_bench.cpp)

target_link_libraries(pcl_benchmarks PRIVATE
  parallel_compute
  benchmark::benchmark
  pcl_warnings)

target_include_directories(pcl_benchmarks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/support)
target_compile_options(pcl_benchmarks PRIVATE ${PCL_ISA_FLAGS} ${PCL_SANITIZER_FLAGS})

# Keep the benchmark binary out of the default `all` target of a Debug build:
# benchmarks are only meaningful in optimised builds.
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_custom_target(run_benchmarks_smoke
    COMMAND pcl_benchmarks --benchmark_filter=PCL/add --benchmark_min_time=1x
            --pcl_sizes=4096 --pcl_threads=2 --pcl_matrix_sizes=64
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Benchmark smoke test (single iteration each)")
endif()
