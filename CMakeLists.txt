cmake_minimum_required(VERSION 3.20)

project(ParallelComputeLab
        VERSION 0.1.0
        DESCRIPTION "CPU parallel computing optimization and benchmarking framework"
        LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(PCL_BUILD_TESTS      "Build the test suite"                       ON)
option(PCL_BUILD_BENCHMARKS "Build benchmark executables"                ON)
option(PCL_ENABLE_OPENMP    "Enable OpenMP parallel implementations"     ON)
option(PCL_ENABLE_SIMD      "Enable explicit SIMD kernels + runtime dispatch" ON)
option(PCL_ENABLE_BLAS      "Use an external BLAS for the GEMM baseline"  ON)
option(PCL_FETCH_DEPENDENCIES "Download pinned dependencies if not found" ON)
option(PCL_WERROR           "Treat compiler warnings as errors"          OFF)
option(PCL_NATIVE_ARCH      "Build with -march=native (non-portable!)"    OFF)
option(PCL_ENABLE_ASAN      "Build with AddressSanitizer"                OFF)
option(PCL_ENABLE_UBSAN     "Build with UndefinedBehaviorSanitizer"      OFF)
option(PCL_ENABLE_TSAN      "Build with ThreadSanitizer (separate build!)" OFF)

set(PCL_SIMD_LEVEL "portable" CACHE STRING
    "ISA level used to compile the optimized variants (portable|avx2|avx512)")
set_property(CACHE PCL_SIMD_LEVEL PROPERTY STRINGS portable avx2 avx512)

# ---------------------------------------------------------------------------
# Sanitizers are mutually exclusive; ASan and TSan cannot be combined.
# ---------------------------------------------------------------------------
if(PCL_ENABLE_ASAN AND PCL_ENABLE_TSAN)
  message(FATAL_ERROR "AddressSanitizer and ThreadSanitizer cannot be enabled together. Use separate build directories.")
endif()
if(PCL_ENABLE_UBSAN AND PCL_ENABLE_TSAN)
  message(FATAL_ERROR "ThreadSanitizer should be configured on its own (see docs/BUILDING.md).")
endif()

set(PCL_SANITIZER_FLAGS "")
if(PCL_ENABLE_ASAN)
  list(APPEND PCL_SANITIZER_FLAGS -fsanitize=address -fno-omit-frame-pointer)
endif()
if(PCL_ENABLE_UBSAN)
  list(APPEND PCL_SANITIZER_FLAGS -fsanitize=undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer)
endif()
if(PCL_ENABLE_TSAN)
  list(APPEND PCL_SANITIZER_FLAGS -fsanitize=thread -fno-omit-frame-pointer)
endif()

# ---------------------------------------------------------------------------
# Language level and warning settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Threads REQUIRED)

add_library(pcl_warnings INTERFACE)
if(MSVC)
  target_compile_options(pcl_warnings INTERFACE /W4 /permissive- /Zc:__cplusplus)
  if(PCL_WERROR)
    target_compile_options(pcl_warnings INTERFACE /WX)
  endif()
else()
  target_compile_options(pcl_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
    -Wold-style-cast -Wnon-virtual-dtor -Wdouble-promotion
    -Wcast-qual -Wunused -Wnull-dereference)
  if(PCL_WERROR)
    target_compile_options(pcl_warnings INTERFACE -Werror)
  endif()
endif()

# ---------------------------------------------------------------------------
# Target ISA selection.
#
# The portable baseline never enables instructions beyond the compiler default
# (x86-64 baseline = SSE2).  Explicit SIMD kernels use function-level target
# attributes and are only reached after a runtime capability check, so a binary
# built with PCL_SIMD_LEVEL=avx2 still runs correctly on an SSE2-only host.
# ---------------------------------------------------------------------------
set(PCL_ISA_FLAGS "")
if(PCL_NATIVE_ARCH)
  message(WARNING "PCL_NATIVE_ARCH=ON: the resulting binary is NOT portable and the compiler flags must be recorded with every result.")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i.86")
  if(PCL_SIMD_LEVEL STREQUAL "avx2")
    set(PCL_ISA_FLAGS -mavx2 -mfma)
  elseif(PCL_SIMD_LEVEL STREQUAL "avx512")
    set(PCL_ISA_FLAGS -mavx2 -mfma -mavx512f -mavx512dq -mavx512bw)
  endif()
endif()

# ---------------------------------------------------------------------------
# OpenMP
# ---------------------------------------------------------------------------
set(PCL_OPENMP_STATUS "disabled")
set(PCL_HAVE_OPENMP OFF)
set(PCL_BLAS_NAME "none")
if(PCL_ENABLE_OPENMP)
  find_package(OpenMP COMPONENTS CXX)
  if(OpenMP_CXX_FOUND)
    set(PCL_HAVE_OPENMP ON)
    set(PCL_OPENMP_STATUS "OpenMP ${OpenMP_CXX_VERSION} (spec ${OpenMP_CXX_SPEC_DATE})")
  else()
    message(WARNING "OpenMP requested but not found; parallel kernels and benchmarks are disabled.")
  endif()
endif()

# ---------------------------------------------------------------------------
# Optional BLAS baseline
# ---------------------------------------------------------------------------
set(PCL_BLAS_STATUS "not-found")
set(PCL_HAVE_BLAS OFF)
if(PCL_ENABLE_BLAS)
  find_package(BLAS QUIET)
  if(NOT BLAS_FOUND)
    # Some distributions ship only the versioned runtime (libblas.so.3) with no
    # -dev symlink, so find_package(BLAS) fails even though cblas.h may still be
    # usable.  Fall back to a direct library search before giving up.
    find_library(PCL_BLAS_FALLBACK_LIB NAMES openblas blas
                 PATHS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE} /usr/lib64 /usr/lib /lib)
    find_path(PCL_BLAS_FALLBACK_INCLUDE cblas.h)
    if(PCL_BLAS_FALLBACK_LIB AND PCL_BLAS_FALLBACK_INCLUDE)
      set(BLAS_FOUND TRUE)
      set(BLA_VENDOR "fallback")
      set(BLAS_LIBRARIES ${PCL_BLAS_FALLBACK_LIB})
      message(STATUS "BLAS: using fallback ${PCL_BLAS_FALLBACK_LIB}")
    endif()
  endif()
  if(BLAS_FOUND)
    set(PCL_HAVE_BLAS ON)
    get_filename_component(_blas_libname ${BLAS_LIBRARIES} NAME)
    if(BLA_VENDOR STREQUAL "fallback")
      string(REGEX REPLACE "lib" "" _blas_libname "${_blas_libname}")
      string(REGEX REPLACE "\\.so.*" "" _blas_libname "${_blas_libname}")
      set(PCL_BLAS_NAME "${_blas_libname} (runtime-only, no -dev package)")
    else()
      set(PCL_BLAS_NAME "${BLA_VENDOR}")
    endif()
    if(BLA_VERSION)
      string(APPEND PCL_BLAS_NAME " ${BLA_VERSION}")
    endif()
    set(PCL_BLAS_STATUS "${PCL_BLAS_NAME} (${BLAS_LIBRARIES})")
  else()
    message(STATUS "No external BLAS found; the GEMM baseline comparison is skipped.")
  endif()
endif()

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_subdirectory(src)

# ---------------------------------------------------------------------------
# Dependencies for tests/benchmarks
# ---------------------------------------------------------------------------
if(PCL_BUILD_TESTS OR PCL_BUILD_BENCHMARKS)
  include(cmake/Dependencies.cmake)
endif()

if(PCL_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(PCL_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# ---------------------------------------------------------------------------
# Generated build configuration header (records the exact compiler/flags used)
# ---------------------------------------------------------------------------
# Record the exact optimisation flags the library was compiled with, so that
# every benchmark result can be traced back to a reproducible build.
set(PCL_BUILD_TYPE_FLAGS "")
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(PCL_BUILD_TYPE_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  set(PCL_BUILD_TYPE_FLAGS "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(PCL_BUILD_TYPE_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")
elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
  set(PCL_BUILD_TYPE_FLAGS "${CMAKE_CXX_FLAGS_MINSIZEREL}")
else()
  set(PCL_BUILD_TYPE_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

string(REPLACE ";" " " PCL_EFFECTIVE_CXX_FLAGS
       "${CMAKE_CXX_FLAGS} ${PCL_BUILD_TYPE_FLAGS} ${PCL_ISA_FLAGS} ${PCL_SANITIZER_FLAGS}")
string(STRIP "${PCL_EFFECTIVE_CXX_FLAGS}" PCL_EFFECTIVE_CXX_FLAGS)
if(PCL_HAVE_OPENMP)
  string(APPEND PCL_EFFECTIVE_CXX_FLAGS " ${OpenMP_CXX_FLAGS}")
endif()
string(STRIP "${PCL_EFFECTIVE_CXX_FLAGS}" PCL_EFFECTIVE_CXX_FLAGS)

set(PCL_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
if(NOT PCL_BUILD_TYPE)
  set(PCL_BUILD_TYPE "unspecified")
endif()

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_config.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/parallel_compute/system/build_config.hpp"
  @ONLY)

message(STATUS "--------------------------------------------------")
message(STATUS "ParallelComputeLab ${PROJECT_VERSION}")
message(STATUS "  build type      : ${PCL_BUILD_TYPE}")
message(STATUS "  SIMD level      : ${PCL_SIMD_LEVEL}")
message(STATUS "  OpenMP          : ${PCL_OPENMP_STATUS}")
message(STATUS "  BLAS            : ${PCL_BLAS_STATUS}")
message(STATUS "  tests           : ${PCL_BUILD_TESTS}")
message(STATUS "  benchmarks      : ${PCL_BUILD_BENCHMARKS}")
message(STATUS "--------------------------------------------------")
