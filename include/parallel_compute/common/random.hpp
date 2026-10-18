// Deterministic test-data generation.
//
// Benchmarks and tests must be reproducible, so all data generation goes
// through these helpers with an explicit seed that is recorded in the result
// metadata.
#ifndef PARALLEL_COMPUTE_COMMON_RANDOM_HPP
#define PARALLEL_COMPUTE_COMMON_RANDOM_HPP

#include <cstdint>
#include <span>

#include "parallel_compute/common/types.hpp"

namespace parallel_compute::random {

/// Default seed used by the benchmark harness and documented in the result
/// schema so that a rerun uses identical inputs.
inline constexpr std::uint64_t kDefaultSeed = 0x9E3779B97F4A7C15ULL;

/// Fill with uniformly distributed values in [lo, hi].
template <Real T>
void fill_uniform(std::span<T> out, std::uint64_t seed, T lo, T hi) {
  std::uint64_t state = seed;
  for (std::size_t i = 0; i < out.size(); ++i) {
    // splitmix64: fast, well distributed, trivially reproducible.
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= (z >> 31);
    const double unit = static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
    out[i] = static_cast<T>(lo + static_cast<T>(unit) * (hi - lo));
  }
}

/// Fill with a cheap integer hash of (seed, index).  Used where exact values
/// are irrelevant but exact reproducibility matters (scheduling workloads).
inline std::uint64_t hash64(std::uint64_t seed, std::uint64_t index) noexcept {
  std::uint64_t z = seed ^ (index + 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 33)) * 0xFF51AFD7ED558CCDULL;
  z = (z ^ (z >> 33)) * 0xC4CEB9FE1A85EC53ULL;
  return z ^ (z >> 33);
}

}  // namespace parallel_compute::random

#endif  // PARALLEL_COMPUTE_COMMON_RANDOM_HPP
