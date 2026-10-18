// General matrix multiplication C = A * B (row-major, m x k * k x n -> m x n).
//
// Implementation ladder (each level is measured against the one below it):
//   reference        i-j-k, strict left-to-right accumulation over k.
//                    Element order matches a textbook triple loop; used as the
//                    numerical oracle.  B is accessed with stride `ldb`.
//   auto_vectorized  i-k-j.  B and C are walked contiguously in the innermost
//                    loop, which is what allows the compiler to vectorise it.
//   blocked          i-k-j tiled into mc x kc x nc blocks so the working set of
//                    the innermost loop fits in cache.
//   parallel         blocked kernel with the row-tile loop parallelised.
//   simd             ikj with an explicit 4-row x vector-width micro-kernel.
//   blas             optional external cblas_dgemm / cblas_sgemm baseline.
#include <algorithm>
#include <cstddef>
#include <string>

#include "parallel_compute/algorithms/matmul.hpp"
#include "parallel_compute/common/attributes.hpp"
#include "parallel_compute/common/types.hpp"
#include "parallel_compute/parallel/scheduler.hpp"
#include "parallel_compute/simd/simd_kernels.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef PCL_HAVE_BLAS
#include <cblas.h>
#endif

namespace parallel_compute::gemm {
namespace {

std::size_t min_of(std::size_t a, std::size_t b) noexcept { return a < b ? a : b; }

// --- reference -------------------------------------------------------------
template <class T>
PCL_NO_VECTORIZE_ATTR void reference_impl(MatrixView<const T> a, MatrixView<const T> b,
                                          MutableMatrixView<T> c) {
  const std::size_t m = a.rows, k = a.cols, n = b.cols;
  for (std::size_t i = 0; i < m; ++i) {
    const T* arow = a.row(i);
    T* crow = c.row(i);
    for (std::size_t j = 0; j < n; ++j) {
      T acc = T{0};
      PCL_NO_VECTORIZE_LOOP
      for (std::size_t p = 0; p < k; ++p) {
        acc += arow[p] * b.data[p * b.cols + j];
      }
      crow[j] = acc;
    }
  }
}

// --- i-k-j (contiguous inner loop) -----------------------------------------
template <class T>
void ikj_impl(MatrixView<const T> a, MatrixView<const T> b, MutableMatrixView<T> c) {
  const std::size_t m = a.rows, k = a.cols, n = b.cols;
  for (std::size_t i = 0; i < m; ++i) {
    T* crow = c.row(i);
    std::fill(crow, crow + n, T{0});
    const T* arow = a.row(i);
    for (std::size_t p = 0; p < k; ++p) {
      const T av = arow[p];
      const T* brow = b.row(p);
      for (std::size_t j = 0; j < n; ++j) {
        crow[j] += av * brow[j];
      }
    }
  }
}

// --- cache blocked i-k-j ---------------------------------------------------
template <class T>
void blocked_impl(MatrixView<const T> a, MatrixView<const T> b, MutableMatrixView<T> c,
                  std::size_t block) {
  const std::size_t m = a.rows, k = a.cols, n = b.cols;
  if (block == 0) block = 1;
  for (std::size_t i = 0; i < m; ++i) {
    std::fill(c.row(i), c.row(i) + n, T{0});
  }
  // ii -> pp -> jj keeps the mc x nc tile of C resident while the whole kc
  // strip of B streams past it.
  for (std::size_t ii = 0; ii < m; ii += block) {
    const std::size_t i_end = min_of(ii + block, m);
    for (std::size_t pp = 0; pp < k; pp += block) {
      const std::size_t p_end = min_of(pp + block, k);
      for (std::size_t jj = 0; jj < n; jj += block) {
        const std::size_t j_end = min_of(jj + block, n);
        for (std::size_t i = ii; i < i_end; ++i) {
          const T* arow = a.row(i);
          T* crow = c.row(i);
          for (std::size_t p = pp; p < p_end; ++p) {
            const T av = arow[p];
            const T* brow = b.row(p);
            for (std::size_t j = jj; j < j_end; ++j) {
              crow[j] += av * brow[j];
            }
          }
        }
      }
    }
  }
}

template <class T>
void blocked_row_range(MatrixView<const T> a, MatrixView<const T> b, MutableMatrixView<T> c,
                       std::size_t i_begin, std::size_t i_end, std::size_t block) {
  const std::size_t k = a.cols, n = b.cols;
  for (std::size_t i = i_begin; i < i_end; ++i) {
    std::fill(c.row(i), c.row(i) + n, T{0});
  }
  for (std::size_t pp = 0; pp < k; pp += block) {
    const std::size_t p_end = min_of(pp + block, k);
    for (std::size_t jj = 0; jj < n; jj += block) {
      const std::size_t j_end = min_of(jj + block, n);
      for (std::size_t i = i_begin; i < i_end; ++i) {
        const T* arow = a.row(i);
        T* crow = c.row(i);
        for (std::size_t p = pp; p < p_end; ++p) {
          const T av = arow[p];
          const T* brow = b.row(p);
          for (std::size_t j = jj; j < j_end; ++j) {
            crow[j] += av * brow[j];
          }
        }
      }
    }
  }
}

// --- explicit SIMD: 4-row panel micro-kernel --------------------------------
template <class T>
void simd_impl(MatrixView<const T> a, MatrixView<const T> b, MutableMatrixView<T> c,
               std::size_t block) {
  const std::size_t m = a.rows, k = a.cols, n = b.cols;
  if (block == 0) block = k == 0 ? 1 : k;
  for (std::size_t i = 0; i < m; ++i) {
    std::fill(c.row(i), c.row(i) + n, T{0});
  }
  const std::size_t min_cols = sizeof(T) == sizeof(double) ? 2 : 4;
  std::size_t i = 0;
  for (; i + 4 <= m; i += 4) {
    T* panel = c.row(i);
    for (std::size_t pp = 0; pp < k; pp += block) {
      const std::size_t p_end = min_of(pp + block, k);
      // Both A and B have to be offset to the first row of this k block: the
      // micro-kernel walks p in [0, kk) from the base pointers it is given.
      const T* apanel = a.row(i) + pp;
      if (n >= min_cols) {
        const T* bblock = b.data + pp * b.cols;
        if constexpr (sizeof(T) == sizeof(double)) {
          simd::kernels::gemm_panel4_f64(panel, c.cols, apanel, a.cols, bblock, b.cols, p_end - pp, n);
        } else {
          simd::kernels::gemm_panel4_f32(panel, c.cols, apanel, a.cols, bblock, b.cols, p_end - pp, n);
        }
      } else {
        // Fewer columns than one vector: fall back to the blocked scalar loop
        // restricted to this row panel.
        for (std::size_t p = pp; p < p_end; ++p) {
          const T* brow = b.row(p);
          for (std::size_t r = 0; r < 4; ++r) {
            const T av = a.row(i + r)[p];
            T* crow = c.row(i + r);
            for (std::size_t j = 0; j < n; ++j) crow[j] += av * brow[j];
          }
        }
      }
    }
  }
  // Remaining rows (0..3) use the plain i-k-j kernel.
  for (; i < m; ++i) {
    const T* arow = a.row(i);
    T* crow = c.row(i);
    for (std::size_t p = 0; p < k; ++p) {
      const T av = arow[p];
      const T* brow = b.row(p);
      for (std::size_t j = 0; j < n; ++j) crow[j] += av * brow[j];
    }
  }
}

}  // namespace

std::size_t flop_count(std::size_t m, std::size_t n, std::size_t k) noexcept { return 2 * m * n * k; }

template <Real T>
void matmul_reference(MatrixView<const T> a, MatrixView<const T> b, MutableMatrixView<T> c,
                      const GemmOptions& options) {
  (void)options;
  if (a.cols != b.rows) {
    throw_invalid_argument("gemm::matmul_reference: A.cols must equal B.rows");
  }
  if (c.rows != a.rows || c.cols != b.cols) {
    throw_invalid_argument("gemm::matmul_reference: C shape must equal A.rows x B.cols");
  }
  if (c.size() == 0) return;
  reference_impl(a, b, c);
}

template <Real T>
void matmul_auto_vectorized(MatrixView<const T> a, MatrixView<const T> b, MutableMatrixView<T> c,
                            const GemmOptions& options) {
  (void)options;
  if (a.cols != b.rows || c.rows != a.rows || c.cols != b.cols) {
    throw_invalid_argument("gemm::matmul_auto_vectorized: incompatible matrix shapes");
  }
  if (c.size() == 0) return;
  ikj_impl(a, b, c);
}

template <Real T>
void matmul_blocked(MatrixView<const T> a, MatrixView<const T> b, MutableMatrixView<T> c,
                    const GemmOptions& options) {
  if (options.block_size == 0) {
    throw_invalid_argument("gemm::matmul_blocked: block_size must be >= 1");
  }
  if (a.cols != b.rows || c.rows != a.rows || c.cols != b.cols) {
    throw_invalid_argument("gemm::matmul_blocked: incompatible matrix shapes");
  }
  if (c.size() == 0) return;
  blocked_impl(a, b, c, options.block_size);
}

template <Real T>
void matmul_parallel(MatrixView<const T> a, MatrixView<const T> b, MutableMatrixView<T> c,
                     const GemmOptions& options) {
  if (options.block_size == 0) {
    throw_invalid_argument("gemm::matmul_parallel: block_size must be >= 1");
  }
  if (a.cols != b.rows || c.rows != a.rows || c.cols != b.cols) {
    throw_invalid_argument("gemm::matmul_parallel: incompatible matrix shapes");
  }
  if (c.size() == 0) return;
  const std::size_t m = a.rows;
  const int nt = parallel::max_threads();
  const std::size_t teams = static_cast<std::size_t>(nt > 0 ? nt : 1);
  const std::size_t rows_per = (m + teams - 1) / (teams == 0 ? 1 : teams);
  if (teams <= 1 || rows_per == 0) {
    blocked_row_range(a, b, c, 0, m, options.block_size);
    return;
  }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nt)
#endif
  for (std::size_t t = 0; t < teams; ++t) {
    const std::size_t begin = std::min(t * rows_per, m);
    const std::size_t end = std::min(begin + rows_per, m);
    blocked_row_range(a, b, c, begin, end, options.block_size);
  }
}

template <Real T>
void matmul_simd(MatrixView<const T> a, MatrixView<const T> b, MutableMatrixView<T> c,
                 const GemmOptions& options) {
  if (a.cols != b.rows || c.rows != a.rows || c.cols != b.cols) {
    throw_invalid_argument("gemm::matmul_simd: incompatible matrix shapes");
  }
  if (c.size() == 0) return;
  simd_impl(a, b, c, options.block_size);
}

template <Real T>
bool matmul_blas(MatrixView<const T> a, MatrixView<const T> b, MutableMatrixView<T> c) {
#ifdef PCL_HAVE_BLAS
  if (a.cols != b.rows || c.rows != a.rows || c.cols != b.cols) {
    throw_invalid_argument("gemm::matmul_blas: incompatible matrix shapes");
  }
  if (c.size() == 0) return true;
  if constexpr (std::is_same_v<T, double>) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(a.rows),
                static_cast<int>(b.cols), static_cast<int>(a.cols), 1.0, a.data,
                static_cast<int>(a.cols), b.data, static_cast<int>(b.cols), 0.0, c.data,
                static_cast<int>(c.cols));
    return true;
  } else {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(a.rows),
                static_cast<int>(b.cols), static_cast<int>(a.cols), 1.0f, a.data,
                static_cast<int>(a.cols), b.data, static_cast<int>(b.cols), 0.0f, c.data,
                static_cast<int>(c.cols));
    return true;
  }
#else
  (void)a;
  (void)b;
  (void)c;
  return false;
#endif
}

bool blas_available() noexcept {
#ifdef PCL_HAVE_BLAS
  return true;
#else
  return false;
#endif
}

const char* blas_name() noexcept {
#ifdef PCL_HAVE_BLAS
  return PCL_BLAS_NAME;
#else
  return "none";
#endif
}

template void matmul_reference<double>(MatrixView<const double>, MatrixView<const double>,
                                      MutableMatrixView<double>, const GemmOptions&);
template void matmul_reference<float>(MatrixView<const float>, MatrixView<const float>,
                                     MutableMatrixView<float>, const GemmOptions&);
template void matmul_auto_vectorized<double>(MatrixView<const double>, MatrixView<const double>,
                                             MutableMatrixView<double>, const GemmOptions&);
template void matmul_auto_vectorized<float>(MatrixView<const float>, MatrixView<const float>,
                                            MutableMatrixView<float>, const GemmOptions&);
template void matmul_blocked<double>(MatrixView<const double>, MatrixView<const double>,
                                     MutableMatrixView<double>, const GemmOptions&);
template void matmul_blocked<float>(MatrixView<const float>, MatrixView<const float>,
                                    MutableMatrixView<float>, const GemmOptions&);
template void matmul_parallel<double>(MatrixView<const double>, MatrixView<const double>,
                                      MutableMatrixView<double>, const GemmOptions&);
template void matmul_parallel<float>(MatrixView<const float>, MatrixView<const float>,
                                     MutableMatrixView<float>, const GemmOptions&);
template void matmul_simd<double>(MatrixView<const double>, MatrixView<const double>,
                                  MutableMatrixView<double>, const GemmOptions&);
template void matmul_simd<float>(MatrixView<const float>, MatrixView<const float>,
                                 MutableMatrixView<float>, const GemmOptions&);
template bool matmul_blas<double>(MatrixView<const double>, MatrixView<const double>,
                                  MutableMatrixView<double>);
template bool matmul_blas<float>(MatrixView<const float>, MatrixView<const float>,
                                 MutableMatrixView<float>);

}  // namespace parallel_compute::gemm
