#include "parallel_compute/simd/cpu_features.hpp"

#include <string>

#include "parallel_compute/simd/simd_kernels.hpp"

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define PCL_ARCH_X86 1
#else
#define PCL_ARCH_X86 0
#endif

namespace parallel_compute::simd {
namespace {

CpuFeatures detect() noexcept {
  CpuFeatures f{};
#if PCL_ARCH_X86
  // __builtin_cpu_supports performs a CPUID check and is available on both GCC
  // and Clang.  Nothing else in the project inspects the ISA directly.
  __builtin_cpu_init();
  f.sse2 = __builtin_cpu_supports("sse2");
  f.sse42 = __builtin_cpu_supports("sse4.2");
  f.avx = __builtin_cpu_supports("avx");
  f.avx2 = __builtin_cpu_supports("avx2");
  f.fma = __builtin_cpu_supports("fma");
  f.avx512f = __builtin_cpu_supports("avx512f");
#endif
#if defined(__aarch64__)
  f.neon = true;
  f.asimd = true;
  // AArch64 always has FP/ASIMD in the baseline ISA.
  f.sse2 = f.avx = f.avx2 = f.fma = false;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  f.neon = true;
#endif
  return f;
}

}  // namespace

const CpuFeatures& cpu_features() noexcept {
  static const CpuFeatures features = detect();
  return features;
}

const char* best_simd_backend() noexcept {
  const CpuFeatures& f = cpu_features();
#if PCL_ARCH_X86
  if (f.avx2 && f.fma) {
    return "avx2";
  }
  if (f.avx) {
    return "avx";
  }
  if (f.sse2) {
    return "sse2";
  }
#endif
  if (f.neon) {
    return "neon";
  }
  return "none";
}

std::size_t simd_lane_width() noexcept {
  const char* name = best_simd_backend();
  if (std::string(name) == "avx2") {
    return 4;  // 256-bit / 64-bit double
  }
  if (std::string(name) == "avx") {
    return 4;
  }
  if (std::string(name) == "sse2") {
    return 2;
  }
  if (std::string(name) == "neon") {
    return 2;
  }
  return 1;
}

std::string describe_supported_simd() {
  const CpuFeatures& f = cpu_features();
  std::string out;
  auto append = [&out](const char* name) {
    if (!out.empty()) {
      out += ",";
    }
    out += name;
  };
  if (f.sse2) append("sse2");
  if (f.sse42) append("sse4.2");
  if (f.avx) append("avx");
  if (f.avx2) append("avx2");
  if (f.fma) append("fma");
  if (f.avx512f) append("avx512f");
  if (f.neon) append("neon");
  if (out.empty()) out = "none";
  return out;
}

bool avx2_active() noexcept {
#if PCL_ARCH_X86
  return cpu_features().avx2 && cpu_features().fma;
#else
  return false;
#endif
}

}  // namespace parallel_compute::simd
