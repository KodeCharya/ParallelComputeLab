# Pinned dependency acquisition for ParallelComputeLab.
#
# Strategy:
#   1. Prefer an already installed system package (fast, offline, distro-tested).
#   2. Otherwise download a pinned release with FetchContent (requires network
#      access at *configure* time only; afterwards the test suite runs offline).
#
# Versions are pinned so that a given commit always builds the same code.
# Override with -DPCL_GTEST_VERSION=... / -DPCL_BENCHMARK_VERSION=... if you
# need to test against a different release.

set(PCL_GTEST_VERSION "1.17.0" CACHE STRING "Pinned GoogleTest version")
set(PCL_BENCHMARK_VERSION "1.9.4" CACHE STRING "Pinned Google Benchmark version")

# --- GoogleTest -------------------------------------------------------------
if(PCL_BUILD_TESTS)
  find_package(GTest 1.14 QUIET CONFIG)
  if(GTest_FOUND)
    set(PCL_GTEST_SOURCE "system" CACHE INTERNAL "")
    message(STATUS "GoogleTest: using system package")
  elseif(PCL_FETCH_DEPENDENCIES)
    message(STATUS "GoogleTest: fetching v${PCL_GTEST_VERSION}")
    include(FetchContent)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG        v${PCL_GTEST_VERSION}
      GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(googletest)
    if(TARGET gtest_main AND NOT TARGET GTest::gtest_main)
      add_library(GTest::gtest ALIAS gtest)
      add_library(GTest::gtest_main ALIAS gtest_main)
    endif()
    set(PCL_GTEST_SOURCE "fetched:v${PCL_GTEST_VERSION}" CACHE INTERNAL "")
  else()
    message(FATAL_ERROR "GoogleTest not found and PCL_FETCH_DEPENDENCIES=OFF.")
  endif()
endif()

# --- Google Benchmark -------------------------------------------------------
if(PCL_BUILD_BENCHMARKS)
  find_package(benchmark 1.8 QUIET CONFIG)
  if(benchmark_FOUND)
    set(PCL_BENCHMARK_SOURCE "system" CACHE INTERNAL "")
    message(STATUS "Google Benchmark: using system package")
  elseif(PCL_FETCH_DEPENDENCIES)
    message(STATUS "Google Benchmark: fetching v${PCL_BENCHMARK_VERSION}")
    include(FetchContent)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(benchmark
      GIT_REPOSITORY https://github.com/google/benchmark.git
      GIT_TAG        v${PCL_BENCHMARK_VERSION}
      GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(benchmark)
    set(PCL_BENCHMARK_SOURCE "fetched:v${PCL_BENCHMARK_VERSION}" CACHE INTERNAL "")
  else()
    message(FATAL_ERROR "Google Benchmark not found and PCL_FETCH_DEPENDENCIES=OFF.")
  endif()
endif()
