// Floating point comparison helpers with explicit, documented tolerances.
#ifndef PARALLEL_COMPUTE_COMMON_TOLERANCE_HPP
#define PARALLEL_COMPUTE_COMMON_TOLERANCE_HPP

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>

#include "parallel_compute/common/types.hpp"

namespace parallel_compute {

/// Comparison succeeds when  |a-b| <= abs_tol + rel_tol * |b|  where `b` is
/// the reference value.  Exact equality (both operands NaN) also succeeds so
/// that NaN-propagation contracts can be tested.
struct Tolerance {
  double abs_tol = 1e-12;
  double rel_tol = 1e-12;
};

/// NaN/Inf-aware tolerant comparison.  `reference` is the trusted value.
template <Real T>
[[nodiscard]] bool approx_equal(T reference, T value, Tolerance tol = {}) noexcept {
  if (std::isnan(reference) || std::isnan(value)) {
    return std::isnan(reference) && std::isnan(value);
  }
  if (std::isinf(reference) || std::isinf(value)) {
    return reference == value;
  }
  const double diff = std::fabs(static_cast<double>(reference) - static_cast<double>(value));
  const double bound =
      tol.abs_tol + tol.rel_tol * std::fabs(static_cast<double>(reference));
  return diff <= bound;
}

/// Relative error |value-reference| / |reference|, or +inf when the reference
/// is zero and the value is not.
template <Real T>
[[nodiscard]] double relative_error(T reference, T value) noexcept {
  if (std::isnan(reference) || std::isnan(value)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double r = std::fabs(static_cast<double>(reference));
  const double d = std::fabs(static_cast<double>(reference) - static_cast<double>(value));
  if (r == 0.0) {
    return d == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
  }
  return d / r;
}

/// True when two spans agree element-wise under `tol`.
template <Real T>
[[nodiscard]] bool spans_equal(std::span<const T> a, std::span<const T> b,
                              Tolerance tol = {}) noexcept {
  if (a.size() != b.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!approx_equal<T>(a[i], b[i], tol)) {
      return false;
    }
  }
  return true;
}

/// Index of the first mismatch, or `a.size()` when the spans are equal.
template <Real T>
[[nodiscard]] std::size_t first_mismatch(std::span<const T> a, std::span<const T> b,
                                         Tolerance tol = {}) noexcept {
  const std::size_t n = a.size() < b.size() ? a.size() : b.size();
  for (std::size_t i = 0; i < n; ++i) {
    if (!approx_equal<T>(a[i], b[i], tol)) {
      return i;
    }
  }
  return a.size() == b.size() ? a.size() : n;
}

}  // namespace parallel_compute

#endif  // PARALLEL_COMPUTE_COMMON_TOLERANCE_HPP
