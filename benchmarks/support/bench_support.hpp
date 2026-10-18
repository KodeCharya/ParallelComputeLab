// Shared benchmark harness: workload configuration, environment metadata,
// correctness gating and counter conventions.
//
// Rules enforced here:
//   * correctness is validated BEFORE the timing loop starts, so only verified
//     results are ever timed and validation costs nothing in the measurement;
//   * data allocation and fill happen before benchmark::State is entered;
//   * every measurement is stamped with the environment that produced it.
#ifndef PCL_BENCH_SUPPORT_HPP
#define PCL_BENCH_SUPPORT_HPP

#include <cstdint>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

#include "parallel_compute/common/random.hpp"
#include "parallel_compute/common/tolerance.hpp"
#include "parallel_compute/common/types.hpp"
#include "parallel_compute/parallel/scheduler.hpp"
#include "parallel_compute/system/system_info.hpp"

namespace pcl_bench {

/// Everything a benchmark run needs to know about itself.
struct RunContext {
  std::vector<std::size_t> vector_sizes;
  std::vector<std::size_t> matrix_sizes;
  std::vector<int> thread_counts;
  std::vector<int> block_sizes;
  std::vector<int> schedule_chunks;
  std::uint64_t seed = parallel_compute::random::kDefaultSeed;
  std::string label;         ///< free-form tag from --pcl_label
  std::string raw_output;    ///< --benchmark_out value, for provenance
  std::string command_line;  ///< argv, recorded verbatim
  std::string repetitions;   ///< --benchmark_repetitions
  std::string min_time;      ///< --benchmark_min_time
  bool random_interleaving = false;
};

/// Parses the --pcl_* flags; unknown flags are left for Google Benchmark.
RunContext parse_context(int argc, char** argv);

/// The context for this process, set once by main() before any benchmark is
/// registered.  Every registration function reads its workload sizes from here.
RunContext& context();

/// One registration function per benchmark family, called from main().
void register_vector_ops_benchmarks();
void register_reduction_benchmarks();
void register_matmul_benchmarks();
void register_memory_benchmarks();
void register_scheduling_benchmarks();
void register_blas_benchmarks();

/// Tolerance used for the pre-timing correctness gate.  Element-wise results
/// are order independent, so this is tight; GEMM and reductions use a looser
/// one (see GemmOptions::tolerance).
inline parallel_compute::Tolerance elementwise_tolerance() { return {0.0, 0.0}; }
inline parallel_compute::Tolerance reduction_tolerance() { return {1e-12, 1e-12}; }
inline parallel_compute::Tolerance gemm_tolerance() { return {1e-11, 1e-11}; }

/// Human readable list of the --pcl_* flags, printed by the harness.
const std::vector<std::string>& context_flag_names();

/// Writes the `*.meta.json` sidecar describing the environment.  Called once
/// from main() after google::ShutdownBenchmark().
void write_metadata(const RunContext& ctx, const std::string& path);

/// Default workload sizes (documented in docs/BENCHMARKING.md):
///   1 Mi doubles  = 8 MiB   -> L2/L3 resident
///  16 Mi doubles  = 128 MiB -> larger than the L3 of most laptops
std::vector<std::size_t> default_vector_sizes();
std::vector<std::size_t> default_matrix_sizes();

/// ---------------------------------------------------------------------------
/// Fixtures.  Allocation and initialisation happen in the constructor, i.e.
/// before benchmark::State is entered, so they are never timed.
/// ---------------------------------------------------------------------------
struct VectorData {
  std::vector<double> a, b;  // inputs
  std::vector<double> y;     // output / accumulator
  std::vector<double> ref;   // reference output for a + b
  explicit VectorData(std::size_t n);
};

struct MatrixData {
  std::size_t m = 0, n = 0, k = 0;
  std::vector<double> a, b, c, ref;
  MatrixData(std::size_t m_, std::size_t n_, std::size_t k_);
};

/// ---------------------------------------------------------------------------
/// Counters.  Every benchmark publishes the same counter names so the analysis
/// scripts never have to guess what a number means.
/// ---------------------------------------------------------------------------
void set_common_counters(benchmark::State& state, std::size_t elements, std::size_t bytes,
                         int threads, bool validated, const char* schedule);

/// Accumulate the bytes and floating point operations performed by `iters`
/// iterations; call once after the timing loop.  Sets `bytes_per_second` and
/// `gflops_per_second` (the latter is 0 when `flops == 0`).
void set_rate_counters(benchmark::State& state, std::size_t bytes_per_iteration,
                       std::size_t flops_per_iteration);

/// Stable, machine-readable label, e.g.
///   "backend=simd,threads=4,schedule=static,chunk=1"
std::string make_label(const char* backend, int threads, const char* schedule, std::size_t chunk);

/// Element-wise comparison used for the pre-timing correctness gate.
bool results_agree(const std::vector<double>& want, const std::vector<double>& got,
                   parallel_compute::Tolerance tol);

/// Registers `name` only when `cond` holds, so unavailable variants (no BLAS,
/// no OpenMP, no AVX2) never appear in the result set.
template <class F>
void register_bench(const std::string& name, F&& fn) {
  benchmark::RegisterBenchmark(name, std::forward<F>(fn))
      ->Unit(benchmark::kNanosecond)
      ->UseRealTime();
}

}  // namespace pcl_bench

#endif  // PCL_BENCH_SUPPORT_HPP
