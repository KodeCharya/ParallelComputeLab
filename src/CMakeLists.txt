# Core library target.
#
# The library is deliberately small: one object library that every test and
# benchmark links against, so that all consumers see exactly the same code that
# was measured.

set(PCL_SOURCES
    common/types.cpp
    algorithms/vector_ops.cpp
    algorithms/reductions.cpp
    algorithms/matmul.cpp
    memory/workloads.cpp
    parallel/scheduler.cpp
    simd/cpu_features.cpp
    simd/simd_kernels.cpp
    system/system_info.cpp)

add_library(parallel_compute STATIC ${PCL_SOURCES})
add_library(ParallelComputeLab::parallel_compute ALIAS parallel_compute)

target_include_directories(parallel_compute
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(parallel_compute
  PUBLIC
    Threads::Threads
  PRIVATE
    pcl_warnings)

target_compile_options(parallel_compute PRIVATE ${PCL_ISA_FLAGS} ${PCL_SANITIZER_FLAGS})
target_compile_definitions(parallel_compute PRIVATE PCL_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

if(PCL_HAVE_OPENMP)
  target_link_libraries(parallel_compute PUBLIC OpenMP::OpenMP_CXX)
  target_compile_options(parallel_compute PUBLIC ${OpenMP_CXX_FLAGS})
endif()

if(PCL_HAVE_BLAS)
  target_compile_definitions(parallel_compute PRIVATE PCL_HAVE_BLAS=1)
  target_link_libraries(parallel_compute PUBLIC ${BLAS_LIBRARIES})
  if(PCL_BLAS_FALLBACK_INCLUDE)
    target_include_directories(parallel_compute PRIVATE ${PCL_BLAS_FALLBACK_INCLUDE})
  endif()
endif()

if(NOT PCL_ENABLE_SIMD)
  # Force the portable scalar path even on a machine that supports AVX2.  This
  # configuration is what CI uses to prove the baseline needs no wide ISA.
  target_compile_definitions(parallel_compute PUBLIC PCL_DISABLE_SIMD=1)
endif()

# ---------------------------------------------------------------------------
# Install rules (optional; used by downstream consumers and by the CI package
# job).
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS parallel_compute EXPORT ParallelComputeLabTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/parallel_compute
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${CMAKE_BINARY_DIR}/generated/parallel_compute/system/build_config.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/parallel_compute/system)
