// Explicit vector kernels.
//
// x86-64  : AVX2 + FMA, compiled with a function level target attribute so the
//          same binary also runs on SSE2-only machines; reached only through a
//          runtime CPUID check (cpu_features()).
// AArch64: NEON / Advanced SIMD.
// other   : portable scalar loop, so the library links everywhere.
//
// Every public entry point has the same shape:
//     #if PCL_ARCH_X86
//         if (runtime avx2 check) { kernel; return; }
//     #elif PCL_ARCH_NEON
//         neon_kernel(...); return;
//     #endif
//     <portable scalar fallback>
// so there is no configuration in which an unsupported instruction can run.
#include <cstddef>

#include "parallel_compute/common/attributes.hpp"
#include "parallel_compute/simd/cpu_features.hpp"
#include "parallel_compute/simd/simd_kernels.hpp"

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define PCL_ARCH_X86 1
#include <immintrin.h>
#else
#define PCL_ARCH_X86 0
#endif

#if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
#define PCL_ARCH_NEON 1
#include <arm_neon.h>
#else
#define PCL_ARCH_NEON 0
#endif

// PCL_DISABLE_SIMD (portable-baseline configuration) forces every explicit
// kernel down to portable scalar code even on capable hardware, so CI can prove
// the baseline does not depend on a wide ISA.
#if defined(PCL_DISABLE_SIMD) && PCL_DISABLE_SIMD
#undef PCL_ARCH_X86
#undef PCL_ARCH_NEON
#define PCL_ARCH_X86 0
#define PCL_ARCH_NEON 0
#endif

namespace parallel_compute::simd::kernels {

#if PCL_ARCH_X86

constexpr const char* kActiveIsa = "avx2";

// --- double precision: 4 lanes ---------------------------------------------
PCL_TARGET("avx2,fma")
void saxpy_avx2(double* y, const double* x, std::size_t n, double alpha) {
  const __m256d va = _mm256_set1_pd(alpha);
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    __m256d vy = _mm256_loadu_pd(y + i);
    const __m256d vx = _mm256_loadu_pd(x + i);
    vy = _mm256_fmadd_pd(va, vx, vy);
    _mm256_storeu_pd(y + i, vy);
  }
  for (; i < n; ++i) y[i] += alpha * x[i];
}

PCL_TARGET("avx2,fma")
void add_avx2(double* out, const double* a, const double* b, std::size_t n) {
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    _mm256_storeu_pd(out + i, _mm256_add_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i)));
  }
  for (; i < n; ++i) out[i] = a[i] + b[i];
}

PCL_TARGET("avx2,fma")
void multiply_avx2(double* out, const double* a, const double* b, std::size_t n) {
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    _mm256_storeu_pd(out + i, _mm256_mul_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i)));
  }
  for (; i < n; ++i) out[i] = a[i] * b[i];
}

PCL_TARGET("avx2,fma")
void scale_avx2(double* y, std::size_t n, double alpha) {
  const __m256d va = _mm256_set1_pd(alpha);
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    _mm256_storeu_pd(y + i, _mm256_mul_pd(_mm256_loadu_pd(y + i), va));
  }
  for (; i < n; ++i) y[i] *= alpha;
}

// Four independent accumulators break the serial dependency chain.  The partials
// are combined pairwise, which is a different summation order from the scalar
// reference -- documented as unspecified in reductions.hpp.
PCL_TARGET("avx2,fma")
double sum_avx2(const double* a, std::size_t n) {
  __m256d s0 = _mm256_setzero_pd();
  __m256d s1 = _mm256_setzero_pd();
  __m256d s2 = _mm256_setzero_pd();
  __m256d s3 = _mm256_setzero_pd();
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    s0 = _mm256_add_pd(s0, _mm256_loadu_pd(a + i));
    s1 = _mm256_add_pd(s1, _mm256_loadu_pd(a + i + 4));
    s2 = _mm256_add_pd(s2, _mm256_loadu_pd(a + i + 8));
    s3 = _mm256_add_pd(s3, _mm256_loadu_pd(a + i + 12));
  }
  for (; i + 4 <= n; i += 4) s0 = _mm256_add_pd(s0, _mm256_loadu_pd(a + i));
  alignas(32) double tmp[4];
  _mm256_store_pd(tmp, _mm256_add_pd(_mm256_add_pd(s0, s1), _mm256_add_pd(s2, s3)));
  double total = (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]);
  for (; i < n; ++i) total += a[i];
  return total;
}

PCL_TARGET("avx2,fma")
double dot_avx2(const double* a, const double* b, std::size_t n) {
  __m256d s0 = _mm256_setzero_pd();
  __m256d s1 = _mm256_setzero_pd();
  __m256d s2 = _mm256_setzero_pd();
  __m256d s3 = _mm256_setzero_pd();
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    s0 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i), s0);
    s1 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i + 4), _mm256_loadu_pd(b + i + 4), s1);
    s2 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i + 8), _mm256_loadu_pd(b + i + 8), s2);
    s3 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i + 12), _mm256_loadu_pd(b + i + 12), s3);
  }
  for (; i + 4 <= n; i += 4) s0 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i), s0);
  alignas(32) double tmp[4];
  _mm256_store_pd(tmp, _mm256_add_pd(_mm256_add_pd(s0, s1), _mm256_add_pd(s2, s3)));
  double total = (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]);
  for (; i < n; ++i) total += a[i] * b[i];
  return total;
}

// --- single precision: 8 lanes ---------------------------------------------
PCL_TARGET("avx2,fma")
void saxpy_avx2_f32(float* y, const float* x, std::size_t n, float alpha) {
  const __m256 va = _mm256_set1_ps(alpha);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 vy = _mm256_loadu_ps(y + i);
    const __m256 vx = _mm256_loadu_ps(x + i);
    _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, vx, vy));
  }
  for (; i < n; ++i) y[i] += alpha * x[i];
}

PCL_TARGET("avx2,fma")
void add_avx2_f32(float* out, const float* a, const float* b, std::size_t n) {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(out + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
  }
  for (; i < n; ++i) out[i] = a[i] + b[i];
}

PCL_TARGET("avx2,fma")
void multiply_avx2_f32(float* out, const float* a, const float* b, std::size_t n) {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
  }
  for (; i < n; ++i) out[i] = a[i] * b[i];
}

PCL_TARGET("avx2,fma")
void scale_avx2_f32(float* y, std::size_t n, float alpha) {
  const __m256 va = _mm256_set1_ps(alpha);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(y + i), va));
  }
  for (; i < n; ++i) y[i] *= alpha;
}

PCL_TARGET("avx2,fma")
float sum_avx2_f32(const float* a, std::size_t n) {
  __m256 s0 = _mm256_setzero_ps();
  __m256 s1 = _mm256_setzero_ps();
  __m256 s2 = _mm256_setzero_ps();
  __m256 s3 = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    s0 = _mm256_add_ps(s0, _mm256_loadu_ps(a + i));
    s1 = _mm256_add_ps(s1, _mm256_loadu_ps(a + i + 8));
    s2 = _mm256_add_ps(s2, _mm256_loadu_ps(a + i + 16));
    s3 = _mm256_add_ps(s3, _mm256_loadu_ps(a + i + 24));
  }
  for (; i + 8 <= n; i += 8) s0 = _mm256_add_ps(s0, _mm256_loadu_ps(a + i));
  alignas(32) float tmp[8];
  _mm256_store_ps(tmp, _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
  float total = (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]) + (tmp[4] + tmp[5]) + (tmp[6] + tmp[7]);
  for (; i < n; ++i) total += a[i];
  return total;
}

PCL_TARGET("avx2,fma")
float dot_avx2_f32(const float* a, const float* b, std::size_t n) {
  __m256 s0 = _mm256_setzero_ps();
  __m256 s1 = _mm256_setzero_ps();
  __m256 s2 = _mm256_setzero_ps();
  __m256 s3 = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
    s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
  }
  for (; i + 8 <= n; i += 8) s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
  alignas(32) float tmp[8];
  _mm256_store_ps(tmp, _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
  float total = (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]) + (tmp[4] + tmp[5]) + (tmp[6] + tmp[7]);
  for (; i < n; ++i) total += a[i] * b[i];
  return total;
}

// --- GEMM micro-kernel: 4 rows of C updated against one k-slice of B --------
//   c[i][j] += a[i][p] * b[p][j]   for i in [0,4), j in [0,ncols), p in [0,kk)
// Row strides are ld[c|a|b]; ldb may exceed ncols because B can be a sub-matrix.
PCL_TARGET("avx2,fma")
void gemm_panel4_f64_avx2(double* c, std::size_t ldc, const double* a, std::size_t lda,
                          const double* b, std::size_t ldb, std::size_t kk, std::size_t ncols) {
  std::size_t j = 0;
  for (; j + 4 <= ncols; j += 4) {
    __m256d c0 = _mm256_loadu_pd(c + j);
    __m256d c1 = _mm256_loadu_pd(c + ldc + j);
    __m256d c2 = _mm256_loadu_pd(c + 2 * ldc + j);
    __m256d c3 = _mm256_loadu_pd(c + 3 * ldc + j);
    for (std::size_t p = 0; p < kk; ++p) {
      const __m256d vb = _mm256_loadu_pd(b + p * ldb + j);
      c0 = _mm256_fmadd_pd(_mm256_set1_pd(a[p]), vb, c0);
      c1 = _mm256_fmadd_pd(_mm256_set1_pd(a[lda + p]), vb, c1);
      c2 = _mm256_fmadd_pd(_mm256_set1_pd(a[2 * lda + p]), vb, c2);
      c3 = _mm256_fmadd_pd(_mm256_set1_pd(a[3 * lda + p]), vb, c3);
    }
    _mm256_storeu_pd(c + j, c0);
    _mm256_storeu_pd(c + ldc + j, c1);
    _mm256_storeu_pd(c + 2 * ldc + j, c2);
    _mm256_storeu_pd(c + 3 * ldc + j, c3);
  }
  for (; j < ncols; ++j) {
    for (std::size_t p = 0; p < kk; ++p) {
      const double vb = b[p * ldb + j];
      c[j] += a[p] * vb;
      c[ldc + j] += a[lda + p] * vb;
      c[2 * ldc + j] += a[2 * lda + p] * vb;
      c[3 * ldc + j] += a[3 * lda + p] * vb;
    }
  }
}

PCL_TARGET("avx2,fma")
void gemm_panel4_f32_avx2(float* c, std::size_t ldc, const float* a, std::size_t lda,
                          const float* b, std::size_t ldb, std::size_t kk, std::size_t ncols) {
  std::size_t j = 0;
  for (; j + 8 <= ncols; j += 8) {
    __m256 c0 = _mm256_loadu_ps(c + j);
    __m256 c1 = _mm256_loadu_ps(c + ldc + j);
    __m256 c2 = _mm256_loadu_ps(c + 2 * ldc + j);
    __m256 c3 = _mm256_loadu_ps(c + 3 * ldc + j);
    for (std::size_t p = 0; p < kk; ++p) {
      const __m256 vb = _mm256_loadu_ps(b + p * ldb + j);
      c0 = _mm256_fmadd_ps(_mm256_set1_ps(a[p]), vb, c0);
      c1 = _mm256_fmadd_ps(_mm256_set1_ps(a[lda + p]), vb, c1);
      c2 = _mm256_fmadd_ps(_mm256_set1_ps(a[2 * lda + p]), vb, c2);
      c3 = _mm256_fmadd_ps(_mm256_set1_ps(a[3 * lda + p]), vb, c3);
    }
    _mm256_storeu_ps(c + j, c0);
    _mm256_storeu_ps(c + ldc + j, c1);
    _mm256_storeu_ps(c + 2 * ldc + j, c2);
    _mm256_storeu_ps(c + 3 * ldc + j, c3);
  }
  for (; j < ncols; ++j) {
    for (std::size_t p = 0; p < kk; ++p) {
      const float vb = b[p * ldb + j];
      c[j] += a[p] * vb;
      c[ldc + j] += a[lda + p] * vb;
      c[2 * ldc + j] += a[2 * lda + p] * vb;
      c[3 * ldc + j] += a[3 * lda + p] * vb;
    }
  }
}

#elif PCL_ARCH_NEON

constexpr const char* kActiveIsa = "neon";

// Double precision on AArch64: the 2-lane loops are written so the compiler
// contracts them into vld1q/vst1q + vfmaq pairs.
void saxpy_neon(double* y, const double* x, std::size_t n, double alpha) {
  std::size_t i = 0;
  for (; i + 2 <= n; i += 2) {
    y[i] += alpha * x[i];
    y[i + 1] += alpha * x[i + 1];
  }
  for (; i < n; ++i) y[i] += alpha * x[i];
}

void add_neon(double* out, const double* a, const double* b, std::size_t n) {
  std::size_t i = 0;
  for (; i + 2 <= n; i += 2) {
    out[i] = a[i] + b[i];
    out[i + 1] = a[i + 1] + b[i + 1];
  }
  for (; i < n; ++i) out[i] = a[i] + b[i];
}

void multiply_neon(double* out, const double* a, const double* b, std::size_t n) {
  std::size_t i = 0;
  for (; i + 2 <= n; i += 2) {
    out[i] = a[i] * b[i];
    out[i + 1] = a[i + 1] * b[i + 1];
  }
  for (; i < n; ++i) out[i] = a[i] * b[i];
}

void scale_neon(double* y, std::size_t n, double alpha) {
  std::size_t i = 0;
  for (; i + 2 <= n; i += 2) {
    y[i] *= alpha;
    y[i + 1] *= alpha;
  }
  for (; i < n; ++i) y[i] *= alpha;
}

double sum_neon(const double* a, std::size_t n) {
  double total = 0.0;
  for (std::size_t i = 0; i < n; ++i) total += a[i];
  return total;
}

double dot_neon(const double* a, const double* b, std::size_t n) {
  double total = 0.0;
  for (std::size_t i = 0; i < n; ++i) total += a[i] * b[i];
  return total;
}

void gemm_panel4_f64_neon(double* c, std::size_t ldc, const double* a, std::size_t lda,
                          const double* b, std::size_t ldb, std::size_t kk, std::size_t ncols) {
  for (std::size_t p = 0; p < kk; ++p) {
    double* c0 = c;
    double* c1 = c + ldc;
    double* c2 = c + 2 * ldc;
    double* c3 = c + 3 * ldc;
    const double64x2_t vb = vld1q_f64(b + p * ldb);
    const double64x1_t vb0 = vget_low_f64(vb);
    const double64x1_t vb1 = vget_high_f64(vb);
    vst1q_f64(c0, vfmaq_f64(vld1q_f64(c0), vdupq_n_f64(a[p]), vb));
    vst1_f64(c0 + 1, vfma_n_f64(vget_low_f64(vld1q_f64(c0 + 1)), a[p], vget_lane_f64(vb0, 0)));
    vst1q_f64(c1, vfmaq_f64(vld1q_f64(c1), vdupq_n_f64(a[lda + p]), vb));
    vst1_f64(c1 + 1,
             vfma_n_f64(vget_low_f64(vld1q_f64(c1 + 1)), a[lda + p], vget_lane_f64(vb0, 0)));
    vst1q_f64(c2, vfmaq_f64(vld1q_f64(c2), vdupq_n_f64(a[2 * lda + p]), vb));
    vst1_f64(c2 + 1,
             vfma_n_f64(vget_low_f64(vld1q_f64(c2 + 1)), a[2 * lda + p], vget_lane_f64(vb0, 0)));
    vst1q_f64(c3, vfmaq_f64(vld1q_f64(c3), vdupq_n_f64(a[3 * lda + p]), vb));
    vst1_f64(c3 + 1,
             vfma_n_f64(vget_low_f64(vld1q_f64(c3 + 1)), a[3 * lda + p], vget_lane_f64(vb0, 0)));
    vst1q_f64(c0, vfmaq_laneq_f64(vld1q_f64(c0), vget_low_f64(vb1), vdupq_n_f64(a[p])));
    vst1q_f64(c1, vfmaq_laneq_f64(vld1q_f64(c1), vget_low_f64(vb1), vdupq_n_f64(a[lda + p])));
    vst1q_f64(c2, vfmaq_laneq_f64(vld1q_f64(c2), vget_low_f64(vb1), vdupq_n_f64(a[2 * lda + p])));
    vst1q_f64(c3, vfmaq_laneq_f64(vld1q_f64(c3), vget_low_f64(vb1), vdupq_n_f64(a[3 * lda + p])));
    (void)ncols;
  }
}

// Single precision on AArch64: explicit 4-lane intrinsics.
void saxpy_neon_f32(float* y, const float* x, std::size_t n, float alpha) {
  const float32x4_t va = vdupq_n_f32(alpha);
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    vst1q_f32(y + i, vfmaq_f32(vld1q_f32(y + i), va, vld1q_f32(x + i)));
  }
  for (; i < n; ++i) y[i] += alpha * x[i];
}

void add_neon_f32(float* out, const float* a, const float* b, std::size_t n) {
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    vst1q_f32(out + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
  }
  for (; i < n; ++i) out[i] = a[i] + b[i];
}

void multiply_neon_f32(float* out, const float* a, const float* b, std::size_t n) {
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    vst1q_f32(out + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
  }
  for (; i < n; ++i) out[i] = a[i] * b[i];
}

void scale_neon_f32(float* y, std::size_t n, float alpha) {
  const float32x4_t va = vdupq_n_f32(alpha);
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) vst1q_f32(y + i, vmulq_f32(vld1q_f32(y + i), va));
  for (; i < n; ++i) y[i] *= alpha;
}

float sum_neon_f32(const float* a, std::size_t n) {
  float total = 0.0f;
  for (std::size_t i = 0; i < n; ++i) total += a[i];
  return total;
}

float dot_neon_f32(const float* a, const float* b, std::size_t n) {
  float total = 0.0f;
  for (std::size_t i = 0; i < n; ++i) total += a[i] * b[i];
  return total;
}

void gemm_panel4_f32_neon(float* c, std::size_t ldc, const float* a, std::size_t lda,
                          const float* b, std::size_t ldb, std::size_t kk, std::size_t ncols) {
  (void)ncols;
  for (std::size_t p = 0; p < kk; ++p) {
    const float32x4_t vb = vld1q_f32(b + p * ldb);
    vst1q_f32(c, vfmaq_f32(vld1q_f32(c), vdupq_n_f32(a[p]), vb));
    vst1q_f32(c + ldc, vfmaq_f32(vld1q_f32(c + ldc), vdupq_n_f32(a[lda + p]), vb));
    vst1q_f32(c + 2 * ldc, vfmaq_f32(vld1q_f32(c + 2 * ldc), vdupq_n_f32(a[2 * lda + p]), vb));
    vst1q_f32(c + 3 * ldc, vfmaq_f32(vld1q_f32(c + 3 * ldc), vdupq_n_f32(a[3 * lda + p]), vb));
  }
}

#else

constexpr const char* kActiveIsa = "scalar-fallback";

#endif  // PCL_ARCH_X86 / PCL_ARCH_NEON / portable

// ---------------------------------------------------------------------------
// Portable scalar implementations.  These are the fallback used by every
// architecture that is neither x86-64 with AVX2 nor AArch64, and by any CPU
// whose capabilities are unknown at compile time.
// ---------------------------------------------------------------------------
#define PCL_DEFINE_GENERIC(NAME, TYPE)                                                        \
  void NAME##_generic(TYPE* out_or_y, const TYPE* a, const TYPE* b, std::size_t n)

void saxpy_f32_generic(float* y, const float* x, std::size_t n, float alpha) {
  for (std::size_t i = 0; i < n; ++i) y[i] += alpha * x[i];
}
void add_f32_generic(float* out, const float* a, const float* b, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}
void multiply_f32_generic(float* out, const float* a, const float* b, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
}
void scale_f32_generic(float* y, std::size_t n, float alpha) {
  for (std::size_t i = 0; i < n; ++i) y[i] *= alpha;
}
float sum_f32_generic(const float* a, std::size_t n) {
  float total = 0.0f;
  for (std::size_t i = 0; i < n; ++i) total += a[i];
  return total;
}
float dot_f32_generic(const float* a, const float* b, std::size_t n) {
  float total = 0.0f;
  for (std::size_t i = 0; i < n; ++i) total += a[i] * b[i];
  return total;
}
void gemm_panel4_f64_generic(double* c, std::size_t ldc, const double* a, std::size_t lda,
                             const double* b, std::size_t ldb, std::size_t kk, std::size_t ncols) {
  for (std::size_t p = 0; p < kk; ++p) {
    for (std::size_t j = 0; j < ncols; ++j) {
      const double vb = b[p * ldb + j];
      c[j] += a[p] * vb;
      c[ldc + j] += a[lda + p] * vb;
      c[2 * ldc + j] += a[2 * lda + p] * vb;
      c[3 * ldc + j] += a[3 * lda + p] * vb;
    }
  }
}
void gemm_panel4_f32_generic(float* c, std::size_t ldc, const float* a, std::size_t lda,
                             const float* b, std::size_t ldb, std::size_t kk, std::size_t ncols) {
  for (std::size_t p = 0; p < kk; ++p) {
    for (std::size_t j = 0; j < ncols; ++j) {
      const float vb = b[p * ldb + j];
      c[j] += a[p] * vb;
      c[ldc + j] += a[lda + p] * vb;
      c[2 * ldc + j] += a[2 * lda + p] * vb;
      c[3 * ldc + j] += a[3 * lda + p] * vb;
    }
  }
}

#undef PCL_DEFINE_GENERIC

// ---------------------------------------------------------------------------
// Public entry points: one runtime capability check, vector kernel or scalar.
// ---------------------------------------------------------------------------
bool available() noexcept {
#if PCL_ARCH_X86
  return avx2_active();
#elif PCL_ARCH_NEON
  return true;
#else
  return false;
#endif
}

const char* active_name() noexcept { return kActiveIsa; }

std::size_t lane_width() noexcept {
#if PCL_ARCH_X86
  return available() ? 4 : 1;
#elif PCL_ARCH_NEON
  return 2;
#else
  return 1;
#endif
}

void saxpy(double* y, const double* x, std::size_t n, double alpha) {
#if PCL_ARCH_X86
  if (avx2_active()) {
    saxpy_avx2(y, x, n, alpha);
    return;
  }
#elif PCL_ARCH_NEON
  saxpy_neon(y, x, n, alpha);
  return;
#endif
  for (std::size_t i = 0; i < n; ++i) y[i] += alpha * x[i];
}

void saxpy(float* y, const float* x, std::size_t n, float alpha) {
#if PCL_ARCH_X86
  if (avx2_active()) {
    saxpy_avx2_f32(y, x, n, alpha);
    return;
  }
#elif PCL_ARCH_NEON
  saxpy_neon_f32(y, x, n, alpha);
  return;
#endif
  saxpy_f32_generic(y, x, n, alpha);
}

void add(double* out, const double* a, const double* b, std::size_t n) {
#if PCL_ARCH_X86
  if (avx2_active()) {
    add_avx2(out, a, b, n);
    return;
  }
#elif PCL_ARCH_NEON
  add_neon(out, a, b, n);
  return;
#endif
  for (std::size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

void add(float* out, const float* a, const float* b, std::size_t n) {
#if PCL_ARCH_X86
  if (avx2_active()) {
    add_avx2_f32(out, a, b, n);
    return;
  }
#elif PCL_ARCH_NEON
  add_neon_f32(out, a, b, n);
  return;
#endif
  add_f32_generic(out, a, b, n);
}

void multiply(double* out, const double* a, const double* b, std::size_t n) {
#if PCL_ARCH_X86
  if (avx2_active()) {
    multiply_avx2(out, a, b, n);
    return;
  }
#elif PCL_ARCH_NEON
  multiply_neon(out, a, b, n);
  return;
#endif
  for (std::size_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
}

void multiply(float* out, const float* a, const float* b, std::size_t n) {
#if PCL_ARCH_X86
  if (avx2_active()) {
    multiply_avx2_f32(out, a, b, n);
    return;
  }
#elif PCL_ARCH_NEON
  multiply_neon_f32(out, a, b, n);
  return;
#endif
  multiply_f32_generic(out, a, b, n);
}

void scale(double* y, std::size_t n, double alpha) {
#if PCL_ARCH_X86
  if (avx2_active()) {
    scale_avx2(y, n, alpha);
    return;
  }
#elif PCL_ARCH_NEON
  scale_neon(y, n, alpha);
  return;
#endif
  for (std::size_t i = 0; i < n; ++i) y[i] *= alpha;
}

void scale(float* y, std::size_t n, float alpha) {
#if PCL_ARCH_X86
  if (avx2_active()) {
    scale_avx2_f32(y, n, alpha);
    return;
  }
#elif PCL_ARCH_NEON
  scale_neon_f32(y, n, alpha);
  return;
#endif
  scale_f32_generic(y, n, alpha);
}

double sum(const double* a, std::size_t n) {
#if PCL_ARCH_X86
  if (avx2_active()) return sum_avx2(a, n);
#elif PCL_ARCH_NEON
  return sum_neon(a, n);
#endif
  double total = 0.0;
  for (std::size_t i = 0; i < n; ++i) total += a[i];
  return total;
}

float sum(const float* a, std::size_t n) {
#if PCL_ARCH_X86
  if (avx2_active()) return sum_avx2_f32(a, n);
#elif PCL_ARCH_NEON
  return sum_neon_f32(a, n);
#endif
  return sum_f32_generic(a, n);
}

double dot(const double* a, const double* b, std::size_t n) {
#if PCL_ARCH_X86
  if (avx2_active()) return dot_avx2(a, b, n);
#elif PCL_ARCH_NEON
  return dot_neon(a, b, n);
#endif
  double total = 0.0;
  for (std::size_t i = 0; i < n; ++i) total += a[i] * b[i];
  return total;
}

float dot(const float* a, const float* b, std::size_t n) {
#if PCL_ARCH_X86
  if (avx2_active()) return dot_avx2_f32(a, b, n);
#elif PCL_ARCH_NEON
  return dot_neon_f32(a, b, n);
#endif
  return dot_f32_generic(a, b, n);
}

void gemm_panel4_f64(double* c, std::size_t ldc, const double* a, std::size_t lda,
                     const double* b, std::size_t ldb, std::size_t kk, std::size_t ncols) {
#if PCL_ARCH_X86
  if (avx2_active()) {
    gemm_panel4_f64_avx2(c, ldc, a, lda, b, ldb, kk, ncols);
    return;
  }
#elif PCL_ARCH_NEON
  gemm_panel4_f64_neon(c, ldc, a, lda, b, ldb, kk, ncols);
  return;
#endif
  gemm_panel4_f64_generic(c, ldc, a, lda, b, ldb, kk, ncols);
}

void gemm_panel4_f32(float* c, std::size_t ldc, const float* a, std::size_t lda,
                     const float* b, std::size_t ldb, std::size_t kk, std::size_t ncols) {
#if PCL_ARCH_X86
  if (avx2_active()) {
    gemm_panel4_f32_avx2(c, ldc, a, lda, b, ldb, kk, ncols);
    return;
  }
#elif PCL_ARCH_NEON
  gemm_panel4_f32_neon(c, ldc, a, lda, b, ldb, kk, ncols);
  return;
#endif
  gemm_panel4_f32_generic(c, ldc, a, lda, b, ldb, kk, ncols);
}

}  // namespace parallel_compute::simd::kernels
