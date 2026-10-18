// Reductions: sum, min, max, dot, mean.
//
// Numerical contract (see include/parallel_compute/algorithms/reductions.hpp):
//   * empty input is legal; sum/dot/mean return 0, min/max return the identity
//     element and report `found = false`;
//   * NaN propagates for every variant;
//   * the order of accumulation differs between variants, so results agree only
//     to within a tolerance, never bit-for-bit.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "parallel_compute/algorithms/reductions.hpp"
#include "parallel_compute/common/attributes.hpp"
#include "parallel_compute/common/types.hpp"
#include "parallel_compute/parallel/scheduler.hpp"
#include "parallel_compute/simd/simd_kernels.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace parallel_compute::reductions {
namespace {

void check_lengths(std::size_t a, std::size_t b, const char* op) {
  if (a != b) {
    throw_invalid_argument(std::string("reductions::") + op + ": span lengths must be equal (got " +
                           std::to_string(a) + " and " + std::to_string(b) + ")");
  }
}

/// NaN-propagating min/max folds.
///
/// The three-way structure matters: a plain `v < best ? v : best` keeps the
/// *last* element on a NaN comparison and therefore silently drops the NaN,
/// which is the defect recorded in tests/regression/test_regression.cpp.
/// When neither operand compares greater than the other the two are either
/// equal or at least one of them is NaN; returning the NaN in that case
/// satisfies the documented "NaN propagates" contract.
template <class T>
[[nodiscard]] constexpr T fold_min(T best, T v) noexcept {
  if (v < best) return v;
  if (best < v) return best;
  return std::isnan(static_cast<double>(v)) ? v : best;
}

template <class T>
[[nodiscard]] constexpr T fold_max(T best, T v) noexcept {
  if (v > best) return v;
  if (best > v) return best;
  return std::isnan(static_cast<double>(v)) ? v : best;
}

// --- sum -------------------------------------------------------------------
template <class T>
PCL_NO_VECTORIZE_ATTR T sum_reference(std::span<const T> a) {
  T total = T{0};
  PCL_NO_VECTORIZE_LOOP
  for (std::size_t i = 0; i < a.size(); ++i) {
    total += a[i];
  }
  return total;
}

template <class T>
T sum_auto(std::span<const T> a) {
  T total = T{0};
  for (std::size_t i = 0; i < a.size(); ++i) {
    total += a[i];
  }
  return total;
}

template <class T>
T sum_simd(std::span<const T> a) {
  return simd::kernels::sum(a.data(), a.size());
}

// --- min / max -------------------------------------------------------------
template <class T>
PCL_NO_VECTORIZE_ATTR T min_reference(std::span<const T> a, bool* found) {
  if (a.empty()) {
    if (found != nullptr) *found = false;
    return std::numeric_limits<T>::max();
  }
  // Fold from the identity element, not from a[0]: if a[0] is NaN the
  // `!(v >= best)` test would otherwise immediately replace it with a[1].
  T best = std::numeric_limits<T>::max();
  PCL_NO_VECTORIZE_LOOP
  for (std::size_t i = 0; i < a.size(); ++i) {
    best = fold_min(best, a[i]);
  }
  if (found != nullptr) *found = true;
  return best;
}

template <class T>
PCL_NO_VECTORIZE_ATTR T max_reference(std::span<const T> a, bool* found) {
  if (a.empty()) {
    if (found != nullptr) *found = false;
    return std::numeric_limits<T>::lowest();
  }
  T best = std::numeric_limits<T>::lowest();
  PCL_NO_VECTORIZE_LOOP
  for (std::size_t i = 0; i < a.size(); ++i) {
    best = fold_max(best, a[i]);
  }
  if (found != nullptr) *found = true;
  return best;
}

template <class T>
T min_auto(std::span<const T> a, bool* found) {
  return min_reference(a, found);
}

template <class T>
T max_auto(std::span<const T> a, bool* found) {
  return max_reference(a, found);
}

/// OpenMP variants use explicit per-thread partials so the NaN-propagating fold
/// is preserved (the `reduction(min:x)` clause has unspecified NaN semantics).
template <class T>
T min_parallel(std::span<const T> a, bool* found) {
  if (a.empty()) {
    if (found != nullptr) *found = false;
    return std::numeric_limits<T>::max();
  }
  const int nt = std::max(1, parallel::max_threads());
  std::vector<T> partial(static_cast<std::size_t>(nt), std::numeric_limits<T>::max());
  std::vector<char> seen(static_cast<std::size_t>(nt), 0);
#ifdef _OPENMP
#pragma omp parallel num_threads(nt)
  {
    const int tid = omp_get_thread_num();
    T local = std::numeric_limits<T>::max();
    bool any = false;
#pragma omp for schedule(static)
    for (std::size_t i = 0; i < a.size(); ++i) {
      local = fold_min(local, a[i]);
      any = true;
    }
    partial[static_cast<std::size_t>(tid)] = local;
    seen[static_cast<std::size_t>(tid)] = any ? 1 : 0;
  }
#else
  T local = std::numeric_limits<T>::max();
  for (std::size_t i = 0; i < a.size(); ++i) {
    local = fold_min(local, a[i]);
  }
  partial[0] = local;
  seen[0] = 1;
#endif
  T best = partial[0];
  for (std::size_t t = 1; t < partial.size(); ++t) {
    if (seen[t] != 0) best = fold_min(best, partial[t]);
  }
  if (found != nullptr) *found = true;
  return best;
}

template <class T>
T max_parallel(std::span<const T> a, bool* found) {
  if (a.empty()) {
    if (found != nullptr) *found = false;
    return std::numeric_limits<T>::lowest();
  }
  const int nt = std::max(1, parallel::max_threads());
  std::vector<T> partial(static_cast<std::size_t>(nt), std::numeric_limits<T>::lowest());
  std::vector<char> seen(static_cast<std::size_t>(nt), 0);
#ifdef _OPENMP
#pragma omp parallel num_threads(nt)
  {
    const int tid = omp_get_thread_num();
    T local = std::numeric_limits<T>::lowest();
    bool any = false;
#pragma omp for schedule(static)
    for (std::size_t i = 0; i < a.size(); ++i) {
      local = fold_max(local, a[i]);
      any = true;
    }
    partial[static_cast<std::size_t>(tid)] = local;
    seen[static_cast<std::size_t>(tid)] = any ? 1 : 0;
  }
#else
  T local = std::numeric_limits<T>::lowest();
  for (std::size_t i = 0; i < a.size(); ++i) {
    local = fold_max(local, a[i]);
  }
  partial[0] = local;
  seen[0] = 1;
#endif
  T best = partial[0];
  for (std::size_t t = 1; t < partial.size(); ++t) {
    if (seen[t] != 0) best = fold_max(best, partial[t]);
  }
  if (found != nullptr) *found = true;
  return best;
}

// --- dot -------------------------------------------------------------------
template <class T>
PCL_NO_VECTORIZE_ATTR T dot_reference(std::span<const T> a, std::span<const T> b) {
  T total = T{0};
  PCL_NO_VECTORIZE_LOOP
  for (std::size_t i = 0; i < a.size(); ++i) {
    total += a[i] * b[i];
  }
  return total;
}

template <class T>
T dot_auto(std::span<const T> a, std::span<const T> b) {
  T total = T{0};
  for (std::size_t i = 0; i < a.size(); ++i) {
    total += a[i] * b[i];
  }
  return total;
}

template <class T>
T dot_simd(std::span<const T> a, std::span<const T> b) {
  return simd::kernels::dot(a.data(), b.data(), a.size());
}

/// Generic parallel reduction over `body(i)` combined with `+`.
template <class T, class F>
T parallel_plus(std::size_t n, F&& body) {
  const int nt = std::max(1, parallel::max_threads());
#ifdef _OPENMP
  T total = T{0};
#pragma omp parallel for schedule(static) num_threads(nt) reduction(+ : total)
  for (std::size_t i = 0; i < n; ++i) {
    total += body(i);
  }
  return total;
#else
  (void)nt;
  T total = T{0};
  for (std::size_t i = 0; i < n; ++i) {
    total += body(i);
  }
  return total;
#endif
}

}  // namespace

template <Real T>
T sum(std::span<const T> a, Backend backend) {
  switch (backend) {
    case Backend::reference:
      return sum_reference(a);
    case Backend::auto_vectorized:
      return sum_auto(a);
    case Backend::openmp:
      if (parallel::max_threads() <= 1) return sum_auto(a);
      return parallel_plus<T>(a.size(), [&](std::size_t i) { return a[i]; });
    case Backend::simd:
    case Backend::best:
      return sum_simd(a);
  }
  return sum_auto(a);
}

template <Real T>
T min_value(std::span<const T> a, Backend backend, bool* found) {
  switch (backend) {
    case Backend::reference:
      return min_reference(a, found);
    case Backend::auto_vectorized:
    case Backend::simd:
    case Backend::best:
      // min/max are branch bound; no hand written vector kernel is justified
      // (documented in docs/LIMITATIONS.md).  Backend::simd therefore maps to
      // the plain serial fold.
      return min_auto(a, found);
    case Backend::openmp:
      if (parallel::max_threads() <= 1) return min_auto(a, found);
      return min_parallel(a, found);
  }
  return min_auto(a, found);
}

template <Real T>
T max_value(std::span<const T> a, Backend backend, bool* found) {
  switch (backend) {
    case Backend::reference:
      return max_reference(a, found);
    case Backend::auto_vectorized:
    case Backend::simd:
    case Backend::best:
      return max_auto(a, found);
    case Backend::openmp:
      if (parallel::max_threads() <= 1) return max_auto(a, found);
      return max_parallel(a, found);
  }
  return max_auto(a, found);
}

template <Real T>
T dot(std::span<const T> a, std::span<const T> b, Backend backend) {
  check_lengths(a.size(), b.size(), "dot");
  switch (backend) {
    case Backend::reference:
      return dot_reference(a, b);
    case Backend::auto_vectorized:
      return dot_auto(a, b);
    case Backend::openmp:
      if (parallel::max_threads() <= 1) return dot_auto(a, b);
      return parallel_plus<T>(a.size(), [&](std::size_t i) { return a[i] * b[i]; });
    case Backend::simd:
    case Backend::best:
      return dot_simd(a, b);
  }
  return dot_auto(a, b);
}

template <Real T>
T mean(std::span<const T> a, Backend backend) {
  if (a.empty()) return T{0};
  const T total = sum(a, backend);
  return total / static_cast<T>(a.size());
}

template double sum<double>(std::span<const double>, Backend);
template float sum<float>(std::span<const float>, Backend);
template double min_value<double>(std::span<const double>, Backend, bool*);
template float min_value<float>(std::span<const float>, Backend, bool*);
template double max_value<double>(std::span<const double>, Backend, bool*);
template float max_value<float>(std::span<const float>, Backend, bool*);
template double dot<double>(std::span<const double>, std::span<const double>, Backend);
template float dot<float>(std::span<const float>, std::span<const float>, Backend);
template double mean<double>(std::span<const double>, Backend);
template float mean<float>(std::span<const float>, Backend);

}  // namespace parallel_compute::reductions
