// General matrix multiplication, C = A * B, row-major storage.
//
//   A: m x k     B: k x n     C: m x n
//
// All variants compute the same mathematical result; they differ only in
// accumulation order, so for floating point `T` results may differ in the last
// ULPs.  Rectangular shapes are fully supported; dimensions of zero produce an
// empty result.  The strictest contract is `reference` (plain left-to-right
// accumulation over k, row-major i-j-k loop).
#ifndef PARALLEL_COMPUTE_ALGORITHMS_MATMUL_HPP
#define PARALLEL_COMPUTE_ALGORITHMS_MATMUL_HPP

#include <cstddef>

#include "parallel_compute/common/types.hpp"

namespace parallel_compute::gemm {

struct GemmOptions {
  std::size_t block_size = 64;  ///< cache blocking tile (must be >= 1)
  int threads = 0;              ///< 0 -> use the OpenMP default team size
};

/// m*n*k*2 floating point operations.
[[nodiscard]] std::size_t flop_count(std::size_t m, std::size_t n, std::size_t k) noexcept;

/// Simple i-j-k reference kernel.  Present as the correctness oracle.
template <Real T>
void matmul_reference(MatrixView<const T> a, MatrixView<const T> b, MutableMatrixView<T> c,
                      const GemmOptions& options = {});

/// i-k-j loop order: the innermost loop walks contiguous memory in both B and
/// C, so this is the first algorithmic step above the reference.
template <Real T>
void matmul_auto_vectorized(MatrixView<const T> a, MatrixView<const T> b, MutableMatrixView<T> c,
                            const GemmOptions& options = {});

/// Cache-blocked (tiled i-k-j).  Block size is tunable and recorded in results.
template <Real T>
void matmul_blocked(MatrixView<const T> a, MatrixView<const T> b, MutableMatrixView<T> c,
                    const GemmOptions& options = {});

/// Row-panel parallel version of the blocked kernel.
template <Real T>
void matmul_parallel(MatrixView<const T> a, MatrixView<const T> b, MutableMatrixView<T> c,
                     const GemmOptions& options = {});

/// Explicit SIMD micro-kernel over a 4-row panel; falls back to
/// matmul_auto_vectorized on CPUs without AVX2/FMA (NEON on ARM).
template <Real T>
void matmul_simd(MatrixView<const T> a, MatrixView<const T> b, MutableMatrixView<T> c,
                 const GemmOptions& options = {});

/// Optional external BLAS (OpenBLAS reference).  Returns false and leaves `c`
/// untouched when no BLAS is linked in.
template <Real T>
bool matmul_blas(MatrixView<const T> a, MatrixView<const T> b, MutableMatrixView<T> c);

/// True when matmul_blas() can run in this build.
[[nodiscard]] bool blas_available() noexcept;

/// Name of the linked BLAS implementation, or "none".
[[nodiscard]] const char* blas_name() noexcept;

}  // namespace parallel_compute::gemm

#endif  // PARALLEL_COMPUTE_ALGORITHMS_MATMUL_HPP
