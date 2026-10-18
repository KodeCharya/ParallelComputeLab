// Low-level vector primitives shared by the algorithm layer.
//
// Every function here is written once per architecture:
//   * x86-64  : AVX2 + FMA, selected at runtime through function-level target
//               attributes (see src/simd/simd_kernels.cpp);
//   * AArch64 : NEON / Advanced SIMD;
//   * other   : a portable scalar implementation, so the same code links and
//               produces correct results everywhere.
//
// `available()` reports whether a real vector implementation (not the scalar
// fallback) will be used, so benchmarks can label their results honestly.
#ifndef PARALLEL_COMPUTE_SIMD_SIMD_KERNELS_HPP
#define PARALLEL_COMPUTE_SIMD_SIMD_KERNELS_HPP

#include <cstddef>

namespace parallel_compute::simd::kernels {

/// True when a genuine vector kernel is available for the running CPU.
bool available() noexcept;

/// Name of the implementation that will actually execute ("avx2", "neon",
/// "scalar-fallback").
const char* active_name() noexcept;

/// Number of elements processed per iteration of the vector loop.
std::size_t lane_width() noexcept;

void saxpy(double* y, const double* x, std::size_t n, double alpha);
void saxpy(float* y, const float* x, std::size_t n, float alpha);
void add(double* out, const double* a, const double* b, std::size_t n);
void add(float* out, const float* a, const float* b, std::size_t n);
void multiply(double* out, const double* a, const double* b, std::size_t n);
void multiply(float* out, const float* a, const float* b, std::size_t n);
void scale(double* y, std::size_t n, double alpha);
void scale(float* y, std::size_t n, float alpha);

double sum(const double* a, std::size_t n);
float sum(const float* a, std::size_t n);
double dot(const double* a, const double* b, std::size_t n);
float dot(const float* a, const float* b, std::size_t n);

/// 4-row GEMM micro-kernel over a contiguous row panel:
///     c[i][j] += a[i][p] * b[p][j]   for i in [0,4), j in [0,ncols), p in [0,kk)
/// with row strides `ldc`, `lda`, `ldb`.  `ldb` may exceed `ncols`.
/// Requirement: ncols >= 2 for f64 and ncols >= 4 for f32; narrower column
/// strips must be handled by the caller with a scalar loop.
void gemm_panel4_f64(double* c, std::size_t ldc, const double* a, std::size_t lda,
                     const double* b, std::size_t ldb, std::size_t kk, std::size_t ncols);
void gemm_panel4_f32(float* c, std::size_t ldc, const float* a, std::size_t lda,
                     const float* b, std::size_t ldb, std::size_t kk, std::size_t ncols);

}  // namespace parallel_compute::simd::kernels

#endif  // PARALLEL_COMPUTE_SIMD_SIMD_KERNELS_HPP
