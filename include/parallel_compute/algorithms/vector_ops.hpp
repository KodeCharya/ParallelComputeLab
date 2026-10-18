// Element-wise vector operations.
//
// API contract (applies to every backend):
//   * all three spans must have the same length;
//   * a length of zero is legal and is a no-op;
//   * `out` may alias exactly one of the input spans (e.g. saxpy(y, y, x)),
//     but must not otherwise overlap them;
//   * violating the length contract throws std::invalid_argument.
#ifndef PARALLEL_COMPUTE_ALGORITHMS_VECTOR_OPS_HPP
#define PARALLEL_COMPUTE_ALGORITHMS_VECTOR_OPS_HPP

#include <span>

#include "parallel_compute/common/types.hpp"

namespace parallel_compute::vector_ops {

/// out = a + b
template <Real T>
void add(std::span<const T> a, std::span<const T> b, std::span<T> out, Backend backend);

/// out = a * b (element-wise product)
template <Real T>
void multiply(std::span<const T> a, std::span<const T> b, std::span<T> out, Backend backend);

/// y *= alpha
template <Real T>
void scale(std::span<T> y, T alpha, Backend backend);

/// y += alpha * x   (BLAS-style AXPY)
template <Real T>
void saxpy(std::span<T> y, std::span<const T> x, T alpha, Backend backend);

/// Number of bytes touched according to the documented convention
/// (one read + one write per element for `scale`, two reads + one write for
/// the binary operations).
template <class T>
[[nodiscard]] std::size_t bytes_touched(std::size_t n) noexcept {
  return 3 * n * sizeof(T);
}

}  // namespace parallel_compute::vector_ops

#endif  // PARALLEL_COMPUTE_ALGORITHMS_VECTOR_OPS_HPP
