#include "parallel_compute/parallel/scheduler.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include "parallel_compute/common/random.hpp"

namespace parallel_compute::parallel {

const char* schedule_name(Schedule s) noexcept {
  switch (s) {
    case Schedule::serial:
      return "serial";
    case Schedule::omp_static:
      return "static";
    case Schedule::omp_dynamic:
      return "dynamic";
    case Schedule::omp_guided:
      return "guided";
  }
  return "unknown";
}

bool schedule_from_name(std::string_view name, Schedule& out) noexcept {
  if (name == "serial") {
    out = Schedule::serial;
    return true;
  }
  if (name == "static" || name == "omp_static") {
    out = Schedule::omp_static;
    return true;
  }
  if (name == "dynamic" || name == "omp_dynamic") {
    out = Schedule::omp_dynamic;
    return true;
  }
  if (name == "guided" || name == "omp_guided") {
    out = Schedule::omp_guided;
    return true;
  }
  return false;
}

int max_threads() noexcept {
#ifdef _OPENMP
  const int t = omp_get_max_threads();
  return t > 0 ? t : 1;
#else
  return 1;
#endif
}

bool set_thread_count(int threads) noexcept {
#ifdef _OPENMP
  if (threads <= 0) return false;
  omp_set_num_threads(threads);
  return omp_get_max_threads() == threads;
#else
  (void)threads;
  return false;
#endif
}

int current_thread_count() noexcept {
#ifdef _OPENMP
  int t = 1;
#pragma omp parallel
#pragma omp single
  { t = omp_get_num_threads(); }
  return t > 0 ? t : 1;
#else
  return 1;
#endif
}

bool openmp_enabled() noexcept {
#ifdef _OPENMP
  return true;
#else
  return false;
#endif
}

std::uint64_t scheduling_workload(std::size_t items, std::size_t min_units, std::size_t max_units,
                                  const ParallelConfig& cfg, std::uint64_t seed) {
  if (items == 0) return 0;
  const std::size_t lo = std::min(min_units, max_units);
  const std::size_t hi = std::max(min_units, max_units);
  const std::size_t span = hi - lo + 1;

  return parallel_reduce<std::uint64_t>(
      items, cfg, 0, [&](std::size_t i) {
        const std::uint64_t h = random::hash64(seed, i);
        const std::size_t units = lo + static_cast<std::size_t>(h % span);
        std::uint64_t v = i + 1;
        for (std::size_t u = 0; u < units; ++u) {
          v = v * 6364136223846793005ULL + 1442695040888963407ULL;
          v ^= v >> 29;
        }
        return v;
      });
}

FalseSharingBuffer false_sharing_slots(std::size_t count) {
  alignas(64) static std::array<std::uint64_t, 4096 * FalseSharingBuffer::kSlotsPerLine> storage{};
  const std::size_t safe = std::min<std::size_t>(count, 4096);
  return FalseSharingBuffer{
      std::span<std::uint64_t>(storage.data(), safe * FalseSharingBuffer::kSlotsPerLine)};
}

}  // namespace parallel_compute::parallel
