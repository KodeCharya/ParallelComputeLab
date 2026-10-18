#ifndef PCL_TEST_SUPPORT_HPP
#define PCL_TEST_SUPPORT_HPP

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "parallel_compute/common/random.hpp"
#include "parallel_compute/common/tolerance.hpp"
#include "parallel_compute/common/types.hpp"

namespace pcl_test {

using parallel_compute::Tolerance;

/// Lengths chosen to cover: empty, single, odd, non-vector-width-aligned
/// (AVX2 f64 = 4 lanes, f32 = 8 lanes), and both sides of those boundaries.
inline const std::vector<std::size_t>& interesting_lengths() {
  static const std::vector<std::size_t> lengths = {0,  1,  2,  3,  4,  5,  7,  8,
                                                   9,  15, 16, 17, 31, 32, 33, 63,
                                                   64, 65, 127, 1000, 4097};
  return lengths;
}

template <class T>
std::vector<T> make_vector(std::size_t n, std::uint64_t seed = 12345, T lo = T{-1}, T hi = T{1}) {
  std::vector<T> v(n);
  parallel_compute::random::fill_uniform<T>(v, seed, lo, hi);
  return v;
}

/// Independent oracle for C = A * B, written as plainly as possible so it shares
/// no code with the library.
template <class T>
std::vector<T> oracle_matmul(const std::vector<T>& a, std::size_t m, std::size_t k,
                              const std::vector<T>& b, std::size_t n) {
  std::vector<T> c(m * n, T{0});
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      T acc = T{0};
      for (std::size_t p = 0; p < k; ++p) {
        acc += a[i * k + p] * b[p * n + j];
      }
      c[i * n + j] = acc;
    }
  }
  return c;
}

/// True when two spans agree element-wise under `tol`.  Returns the index of
/// the first mismatch on failure so the assertion message is useful.
/// True when two contiguous containers agree element-wise under `tol`.
/// Written generically (rather than as std::span overloads) because
/// std::span<const T> cannot deduce T from a std::vector argument.
template <class A, class B>
[[nodiscard]] bool all_equal(const A& a, const B& b, Tolerance tol = {}) {
  using T = std::remove_cv_t<typename A::value_type>;
  return parallel_compute::first_mismatch(std::span<const T>(a.data(), a.size()),
                                          std::span<const T>(b.data(), b.size()), tol) == a.size() &&
         a.size() == b.size();
}

inline const std::vector<parallel_compute::Backend>& all_backends() {
  static const std::vector<parallel_compute::Backend> backends = {
      parallel_compute::Backend::reference, parallel_compute::Backend::auto_vectorized,
      parallel_compute::Backend::openmp, parallel_compute::Backend::simd,
      parallel_compute::Backend::best};
  return backends;
}

}  // namespace pcl_test

#endif  // PCL_TEST_SUPPORT_HPP
