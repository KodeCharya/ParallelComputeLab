// Non-owning matrix views and shared backend vocabulary.
#ifndef PARALLEL_COMPUTE_COMMON_TYPES_HPP
#define PARALLEL_COMPUTE_COMMON_TYPES_HPP

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace parallel_compute {

/// Floating point types supported by the kernels in this library.
template <class T>
concept Real = std::is_same_v<T, float> || std::is_same_v<T, double>;

/// Read-only row-major matrix view. No bounds are stored beyond `rows`/`cols`;
/// callers are responsible for passing consistent extents (documented in the
/// README under "Matrix layout").
template <class T>
struct MatrixView {
  const T* data = nullptr;
  std::size_t rows = 0;
  std::size_t cols = 0;

  constexpr MatrixView() = default;
  constexpr MatrixView(const T* d, std::size_t r, std::size_t c) : data(d), rows(r), cols(c) {}

  [[nodiscard]] constexpr std::size_t size() const noexcept { return rows * cols; }
  [[nodiscard]] constexpr bool empty() const noexcept { return size() == 0; }
  [[nodiscard]] constexpr const T* row(std::size_t r) const noexcept { return data + r * cols; }
};

/// Writable row-major matrix view.
template <class T>
struct MutableMatrixView {
  T* data = nullptr;
  std::size_t rows = 0;
  std::size_t cols = 0;

  constexpr MutableMatrixView() = default;
  constexpr MutableMatrixView(T* d, std::size_t r, std::size_t c) : data(d), rows(r), cols(c) {}

  [[nodiscard]] constexpr std::size_t size() const noexcept { return rows * cols; }
  [[nodiscard]] constexpr bool empty() const noexcept { return size() == 0; }
  [[nodiscard]] constexpr T* row(std::size_t r) noexcept { return data + r * cols; }
};

/// Implementation variant selector shared by all kernels.
///
///  * `reference`        -- portable scalar code, auto-vectorization disabled.
///  * `auto_vectorized`  -- same algorithm, plain loops, compiler vectorizes.
///  * `openmp`           -- OpenMP parallel loop over the same algorithm.
///  * `simd`             -- hand written intrinsics; falls back to
///                          `auto_vectorized` when no supported ISA is present.
///  * `best`             -- runtime dispatch to the fastest available variant
///                          for the current CPU.
enum class Backend {
  reference = 0,
  auto_vectorized = 1,
  openmp = 2,
  simd = 3,
  best = 4,
};

/// Stable lowercase identifier used in JSON/CSV output.
[[nodiscard]] const char* backend_name(Backend b) noexcept;

/// True if the backend runs more than one thread.
[[nodiscard]] bool backend_is_parallel(Backend b) noexcept;

/// Throws std::invalid_argument with a descriptive message.
[[noreturn]] void throw_invalid_argument(const std::string& what);

}  // namespace parallel_compute

#endif  // PARALLEL_COMPUTE_COMMON_TYPES_HPP
