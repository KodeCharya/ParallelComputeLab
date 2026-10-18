// Runtime CPU capability detection.
//
// Explicit SIMD kernels are compiled with function-level target attributes and
// are only ever called after a successful check here.  Nothing in this project
// executes an instruction that the running CPU does not advertise.
#ifndef PARALLEL_COMPUTE_SIMD_CPU_FEATURES_HPP
#define PARALLEL_COMPUTE_SIMD_CPU_FEATURES_HPP

#include <cstddef>
#include <string>

namespace parallel_compute::simd {

struct CpuFeatures {
  bool sse2 = false;
  bool sse42 = false;
  bool avx = false;
  bool avx2 = false;
  bool fma = false;
  bool avx512f = false;
  bool neon = false;
  bool asimd = false;  ///< aarch64 Advanced SIMD
};

/// Cached CPUID query (performed once, thread-safe).
const CpuFeatures& cpu_features() noexcept;

/// Name of the widest supported vector unit: "avx2", "sse2", "neon", "none".
const char* best_simd_backend() noexcept;

/// Doubles processed per vector register for `best_simd_backend()`.
std::size_t simd_lane_width() noexcept;

/// Human readable comma separated list of the supported units, for metadata.
std::string describe_supported_simd();

/// True when an AVX2/FMA kernel would actually be used on this CPU.
bool avx2_active() noexcept;

}  // namespace parallel_compute::simd

#endif  // PARALLEL_COMPUTE_SIMD_CPU_FEATURES_HPP
