// Memory-bound workloads.
//
// Byte counting convention (documented in docs/BENCHMARKING.md):
//   * `copy`       : 2 * n * sizeof(T)  (one read, one write)
//   * `scale`      : 2 * n * sizeof(T)  (one read, one write)
//   * `triad`      : 3 * n * sizeof(T)  (two reads, one write)
//   * sequential   : 1 * n * sizeof(T)  (logical bytes read)
//   * strided      : 1 * n * sizeof(T)  (LOGICAL bytes read; the hardware
//                     actually transfers whole cache lines, so the reported
//                     number is a lower bound -- see docs/LIMITATIONS.md)
#ifndef PARALLEL_COMPUTE_MEMORY_WORKLOADS_HPP
#define PARALLEL_COMPUTE_MEMORY_WORKLOADS_HPP

#include <cstddef>
#include <span>

#include "parallel_compute/common/types.hpp"

namespace parallel_compute::memory {

enum class Workload { copy, scale, triad, sequential, strided };

[[nodiscard]] const char* workload_name(Workload w) noexcept;

[[nodiscard]] std::size_t bytes_touched(Workload w, std::size_t n, std::size_t element_size) noexcept;

/// dst[i] = src[i]
template <Real T>
void copy(std::span<const T> src, std::span<T> dst, Backend backend);

/// dst[i] *= alpha
template <Real T>
void scale(std::span<T> dst, T alpha, Backend backend);

/// a[i] = b[i] + alpha * c[i]
template <Real T>
void triad(std::span<T> a, std::span<const T> b, std::span<const T> c, T alpha, Backend backend);

/// Sequential read: sum(src).  A pure streaming read used to estimate the
/// achievable read bandwidth.
template <Real T>
[[nodiscard]] T sequential_sum(std::span<const T> src, Backend backend);

/// Strided read: sums `src[0], src[stride], src[2*stride], ...`.
///
/// The span is the *buffer*, not the list of visited elements: the number of
/// elements actually read is `count = (src.size() + stride - 1) / stride`
/// (0 for an empty span, and stride 0 is treated as 1).  Every index touched is
/// therefore strictly less than `src.size()`, so the call is always in bounds.
/// `bytes_touched(Workload::strided, count, sizeof(T))` reports the logical
/// bytes read, which is a lower bound on the real cache-line traffic.
template <Real T>
[[nodiscard]] T strided_sum(std::span<const T> src, std::size_t stride, Backend backend);

/// Number of elements `strided_sum` visits for a buffer of `size` elements.
[[nodiscard]] std::size_t strided_element_count(std::size_t size,
                                                 std::size_t stride) noexcept;

}  // namespace parallel_compute::memory

#endif  // PARALLEL_COMPUTE_MEMORY_WORKLOADS_HPP
