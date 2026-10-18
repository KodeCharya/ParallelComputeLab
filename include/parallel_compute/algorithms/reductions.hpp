// Reductions: sum, min, max, dot product, mean.
//
// API contract:
//   * empty input is legal;
//   * sum/dot/mean return T{0} on empty input;
//   * min_value/max_value return std::numeric_limits<T>::max()/lowest() on
//     empty input (documented identity) and set *found = false when the caller
//     passed a non-null `found` pointer;
//   * NaN propagates: if any inspected element is NaN the result is NaN for
//     sum/dot/min/max.  mean of an empty range returns T{0}.
//   * reduction order is unspecified for the parallel and SIMD variants, so
//     results may differ from the reference in the last few ULPs; tests use
//     Tolerance, never exact equality, for these.
#ifndef PARALLEL_COMPUTE_ALGORITHMS_REDUCTIONS_HPP
#define PARALLEL_COMPUTE_ALGORITHMS_REDUCTIONS_HPP

#include <limits>
#include <span>

#include "parallel_compute/common/types.hpp"

namespace parallel_compute::reductions {

template <Real T>
[[nodiscard]] T sum(std::span<const T> a, Backend backend);

template <Real T>
[[nodiscard]] T min_value(std::span<const T> a, Backend backend, bool* found = nullptr);

template <Real T>
[[nodiscard]] T max_value(std::span<const T> a, Backend backend, bool* found = nullptr);

template <Real T>
[[nodiscard]] T dot(std::span<const T> a, std::span<const T> b, Backend backend);

/// Arithmetic mean computed as sum/n.  Provided because it is the form used in
/// the workloads; it is *not* numerically more accurate than sum().
template <Real T>
[[nodiscard]] T mean(std::span<const T> a, Backend backend);

}  // namespace parallel_compute::reductions

#endif  // PARALLEL_COMPUTE_ALGORITHMS_REDUCTIONS_HPP
