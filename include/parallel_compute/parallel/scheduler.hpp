// OpenMP parallel-loop plumbing and the scheduling experiment harness.
//
// No custom task scheduler is provided on purpose: the project first measures
// what OpenMP's built-in policies cost, and only introduces anything else if
// those measurements identify a concrete scheduling problem.
#ifndef PARALLEL_COMPUTE_PARALLEL_SCHEDULER_HPP
#define PARALLEL_COMPUTE_PARALLEL_SCHEDULER_HPP

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include "parallel_compute/common/types.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace parallel_compute::parallel {

enum class Schedule {
  serial = 0,
  omp_static = 1,
  omp_dynamic = 2,
  omp_guided = 3,
};

[[nodiscard]] const char* schedule_name(Schedule s) noexcept;
[[nodiscard]] bool schedule_from_name(std::string_view name, Schedule& out) noexcept;

struct ParallelConfig {
  int threads = 0;           ///< 0 -> OpenMP default (omp_get_max_threads)
  Schedule schedule = Schedule::omp_static;
  std::size_t chunk = 1;     ///< dynamic/guided chunk; 0 -> OpenMP default
  bool use_nested = false;   ///< reserved; nested parallelism is off by default
};

/// Maximum number of threads this build/runtime can use (>= 1).
[[nodiscard]] int max_threads() noexcept;

/// Set the OpenMP default team size for the current process.  Returns false
/// when OpenMP is disabled or the request was rejected.
bool set_thread_count(int threads) noexcept;

/// Current OpenMP default team size (1 when OpenMP is disabled).
[[nodiscard]] int current_thread_count() noexcept;

/// True when the library was compiled with OpenMP support.
[[nodiscard]] bool openmp_enabled() noexcept;

namespace detail {

/// Effective team size for a config (never returns < 1).
inline int effective_threads(const ParallelConfig& cfg) noexcept {
  if (cfg.threads > 0) return cfg.threads;
  return max_threads();
}

/// True when the loop must run serially: no OpenMP in this build, a single
/// requested thread, or the serial policy.  Keeping this path identical to the
/// OpenMP one for a single thread is what makes a fair T(1) reference possible.
inline bool run_serial(const ParallelConfig& cfg) noexcept {
  return cfg.schedule == Schedule::serial || !openmp_enabled() || effective_threads(cfg) <= 1;
}

}  // namespace detail

/// Run `body(i)` for i in [0, n) using the requested policy.
template <class F>
void parallel_for(std::size_t n, const ParallelConfig& cfg, F&& body) {
  if (n == 0) return;
  if (detail::run_serial(cfg) || !openmp_enabled()) {
    for (std::size_t i = 0; i < n; ++i) body(i);
    return;
  }
#ifdef _OPENMP
  const int nt = detail::effective_threads(cfg);
  switch (cfg.schedule) {
    case Schedule::serial:
      for (std::size_t i = 0; i < n; ++i) body(i);
      return;
    case Schedule::omp_static:
      if (cfg.chunk > 0) {
#pragma omp parallel for schedule(static, cfg.chunk) num_threads(nt)
        for (std::size_t i = 0; i < n; ++i) body(i);
      } else {
#pragma omp parallel for schedule(static) num_threads(nt)
        for (std::size_t i = 0; i < n; ++i) body(i);
      }
      return;
    case Schedule::omp_dynamic:
      if (cfg.chunk > 0) {
#pragma omp parallel for schedule(dynamic, cfg.chunk) num_threads(nt)
        for (std::size_t i = 0; i < n; ++i) body(i);
      } else {
        // chunk == 0 means "let OpenMP choose"; passing 0 to the clause is an
        // invalid chunk_size and must never reach the runtime.
#pragma omp parallel for schedule(dynamic) num_threads(nt)
        for (std::size_t i = 0; i < n; ++i) body(i);
      }
      return;
    case Schedule::omp_guided:
      if (cfg.chunk > 0) {
#pragma omp parallel for schedule(guided, cfg.chunk) num_threads(nt)
        for (std::size_t i = 0; i < n; ++i) body(i);
      } else {
#pragma omp parallel for schedule(guided) num_threads(nt)
        for (std::size_t i = 0; i < n; ++i) body(i);
      }
      return;
  }
#else
  for (std::size_t i = 0; i < n; ++i) body(i);
#endif
}

/// Same partition, but the callback receives a contiguous (start, count) range
/// so it can hand a `std::span` slice to the kernel and avoid index maths.
template <class F>
void parallel_for_chunks(std::size_t n, const ParallelConfig& cfg, F&& body) {
  if (n == 0) return;
  const int nt = static_cast<int>(detail::effective_threads(cfg));
  const std::size_t teams = nt > 0 ? static_cast<std::size_t>(nt) : 1;
  const std::size_t base = n / teams;
  const std::size_t rem = n % teams;
  std::size_t offset = 0;
  for (std::size_t t = 0; t < teams; ++t) {
    const std::size_t count = base + (t < rem ? 1 : 0);
    const std::size_t start = offset;
    offset += count;
    if (count == 0) continue;
    body(start, count);
  }
}

/// Partitioned reduction: `body(i)` is summed into `init` using `+`.
template <class T, class F>
[[maybe_unused]] T parallel_reduce(std::size_t n, const ParallelConfig& cfg, T init, F&& body) {
  if (n == 0) return init;
  if (detail::run_serial(cfg) || !openmp_enabled()) {
    T total = init;
    for (std::size_t i = 0; i < n; ++i) total += body(i);
    return total;
  }
#ifdef _OPENMP
  const int nt = detail::effective_threads(cfg);
  switch (cfg.schedule) {
    case Schedule::omp_static:
#pragma omp parallel for schedule(static) num_threads(nt) reduction(+ : init)
      for (std::size_t i = 0; i < n; ++i) init += body(i);
      return init;
    case Schedule::omp_dynamic:
      if (cfg.chunk > 0) {
#pragma omp parallel for schedule(dynamic, cfg.chunk) num_threads(nt) reduction(+ : init)
        for (std::size_t i = 0; i < n; ++i) init += body(i);
      } else {
#pragma omp parallel for schedule(dynamic) num_threads(nt) reduction(+ : init)
        for (std::size_t i = 0; i < n; ++i) init += body(i);
      }
      return init;
    case Schedule::omp_guided:
      if (cfg.chunk > 0) {
#pragma omp parallel for schedule(guided, cfg.chunk) num_threads(nt) reduction(+ : init)
        for (std::size_t i = 0; i < n; ++i) init += body(i);
      } else {
#pragma omp parallel for schedule(guided) num_threads(nt) reduction(+ : init)
        for (std::size_t i = 0; i < n; ++i) init += body(i);
      }
      return init;
    case Schedule::serial:
      break;
  }
#endif
  T total = init;
  for (std::size_t i = 0; i < n; ++i) total += body(i);
  return total;
}

/// Inspectable scheduling workload used by the scheduling benchmarks.
///
/// Each item performs `min_units + hash(seed, i) % (max_units - min_units + 1)`
/// iterations of a cheap integer mix, modelling irregular-cost tasks where
/// static and dynamic scheduling actually differ.  The return value is a
/// checksum, so the work cannot be optimised away.
[[nodiscard]] std::uint64_t scheduling_workload(std::size_t items, std::size_t min_units,
                                               std::size_t max_units, const ParallelConfig& cfg,
                                               std::uint64_t seed);

/// Cache-line separated per-thread counters used by the false-sharing
/// experiment.  `slot(i)` and `slot(j)` are guaranteed to sit on different
/// 64-byte lines for every i != j, which is the "no false sharing" layout the
/// experiment contrasts against its packed (8 bytes per counter) variant.
struct FalseSharingBuffer {
  static constexpr std::size_t kSlotsPerLine = 64 / sizeof(std::uint64_t);

  std::span<std::uint64_t> storage;

  [[nodiscard]] std::uint64_t* slot(std::size_t i) noexcept {
    return storage.data() + i * kSlotsPerLine;
  }
  [[nodiscard]] const std::uint64_t* slot(std::size_t i) const noexcept {
    return storage.data() + i * kSlotsPerLine;
  }
};

[[nodiscard]] FalseSharingBuffer false_sharing_slots(std::size_t count);

}  // namespace parallel_compute::parallel

#endif  // PARALLEL_COMPUTE_PARALLEL_SCHEDULER_HPP
