// GEMM benchmarks.
//
// The optimization ladder, all computing the same C = A * B:
//   reference        i-j-k, strict accumulation (Level 0 oracle)
//   auto_vectorized  i-k-j, contiguous inner loop (Level 1: compiler)
//   blocked          i-k-j with a cache tile        (Level 2: memory)
//   simd             4-row x AVX2 micro-kernel       (Level 4: SIMD)
//   parallel         blocked kernel, row tiles over threads (Level 3)
//   blas             optional external implementation, if linked
//
// Byte counting for the bandwidth counter uses the documented GEMM traffic
// model 2*(m*n + k*n + m*k)*sizeof(T), which is a *model* of main-memory
// traffic for a cache-blocked kernel, not a measurement of it.
#include <string>
#include <vector>

#include "bench_support.hpp"
#include "parallel_compute/algorithms/matmul.hpp"

namespace gm = parallel_compute::gemm;
using parallel_compute::MutableMatrixView;
using parallel_compute::MatrixView;

namespace {

enum class Variant { reference, auto_vectorized, blocked, simd, parallel, blas };

const char* variant_name(Variant v) {
  switch (v) {
    case Variant::reference:
      return "reference";
    case Variant::auto_vectorized:
      return "auto_vectorized";
    case Variant::blocked:
      return "blocked";
    case Variant::simd:
      return "simd";
    case Variant::parallel:
      return "parallel";
    case Variant::blas:
      return "blas";
  }
  return "unknown";
}

bool variant_is_parallel(Variant v) { return v == Variant::parallel; }

void run_gemm(benchmark::State& state, std::size_t edge, Variant variant, std::size_t block,
              int threads) {
  const std::size_t m = edge, n = edge, k = edge;
  pcl_bench::MatrixData data(m, n, k);

  if (variant_is_parallel(variant)) {
    parallel_compute::parallel::set_thread_count(threads);
  }

  gm::GemmOptions opts;
  opts.block_size = block;

  const auto av = MatrixView<const double>(data.a.data(), m, k);
  const auto bv = MatrixView<const double>(data.b.data(), k, n);

  // ---- pre-timing correctness gate ------------------------------------
  std::vector<double> check(m * n, 0.0);
  bool ran = true;
  switch (variant) {
    case Variant::reference:
      gm::matmul_reference<double>(av, bv, MutableMatrixView<double>(check.data(), m, n));
      break;
    case Variant::auto_vectorized:
      gm::matmul_auto_vectorized<double>(av, bv, MutableMatrixView<double>(check.data(), m, n));
      break;
    case Variant::blocked:
      gm::matmul_blocked<double>(av, bv, MutableMatrixView<double>(check.data(), m, n), opts);
      break;
    case Variant::simd:
      gm::matmul_simd<double>(av, bv, MutableMatrixView<double>(check.data(), m, n), opts);
      break;
    case Variant::parallel:
      gm::matmul_parallel<double>(av, bv, MutableMatrixView<double>(check.data(), m, n), opts);
      break;
    case Variant::blas:
      ran = gm::matmul_blas<double>(av, bv, MutableMatrixView<double>(check.data(), m, n));
      break;
  }
  if (!ran) {
    state.SkipWithError("no BLAS available in this build");
    return;
  }
  if (!pcl_bench::results_agree(data.ref, check, pcl_bench::gemm_tolerance())) {
    state.SkipWithError(std::string(variant_name(variant)) +
                        " disagrees with the scalar reference GEMM");
    return;
  }

  const std::size_t flops = gm::flop_count(m, n, k);
  const std::size_t bytes = 2 * (m * n + k * n + m * k) * sizeof(double);
  const char* schedule = variant_is_parallel(variant) ? "static" : "none";
  state.SetLabel(pcl_bench::make_label(variant_name(variant), threads, schedule, block));
  pcl_bench::set_common_counters(state, m * n, bytes, threads, true, schedule);
  state.counters["m"] = static_cast<double>(m);
  state.counters["n"] = static_cast<double>(n);
  state.counters["k"] = static_cast<double>(k);
  state.counters["block"] = static_cast<double>(block);
  state.counters["flops_per_iter"] = static_cast<double>(flops);

  auto* c = data.c.data();
  for (auto _ : state) {
    benchmark::DoNotOptimize(data.a.data());
    switch (variant) {
      case Variant::reference:
        gm::matmul_reference<double>(av, bv, MutableMatrixView<double>(c, m, n));
        break;
      case Variant::auto_vectorized:
        gm::matmul_auto_vectorized<double>(av, bv, MutableMatrixView<double>(c, m, n));
        break;
      case Variant::blocked:
        gm::matmul_blocked<double>(av, bv, MutableMatrixView<double>(c, m, n), opts);
        break;
      case Variant::simd:
        gm::matmul_simd<double>(av, bv, MutableMatrixView<double>(c, m, n), opts);
        break;
      case Variant::parallel:
        gm::matmul_parallel<double>(av, bv, MutableMatrixView<double>(c, m, n), opts);
        break;
      case Variant::blas:
        gm::matmul_blas<double>(av, bv, MutableMatrixView<double>(c, m, n));
        break;
    }
    benchmark::DoNotOptimize(c);
  }
  pcl_bench::set_rate_counters(state, bytes, flops);
}

}  // namespace

void pcl_bench::register_matmul_benchmarks() {
  const RunContext& ctx = context();
  const std::vector<Variant> variants = {
      Variant::reference, Variant::auto_vectorized, Variant::blocked, Variant::simd};

  for (std::size_t edge : ctx.matrix_sizes) {
    for (Variant v : variants) {
      // Only the blocked / SIMD kernels have a block parameter worth sweeping.
      std::vector<std::size_t> blocks{0};
      if (v == Variant::blocked || v == Variant::simd) {
        blocks.clear();
        for (int b : ctx.block_sizes) blocks.push_back(static_cast<std::size_t>(b));
      }
      for (std::size_t block : blocks) {
        const std::string name = std::string("PCL/gemm/f64/") + variant_name(v) + "/n=" +
                                 std::to_string(edge) + "/block=" + std::to_string(block) +
                                 "/T=1";
        register_bench(name, [edge, v, block](benchmark::State& state) {
          run_gemm(state, edge, v, block, 1);
        });
      }
    }
    // Parallel GEMM: thread sweep at the default block size, plus one blocked
    // sweep so the cache-blocking effect can be separated from threading.
    if (parallel_compute::parallel::openmp_enabled()) {
      for (int threads : ctx.thread_counts) {
        for (int block_i : ctx.block_sizes) {
          const std::size_t block = static_cast<std::size_t>(block_i);
          const std::string name = std::string("PCL/gemm/f64/parallel/n=") +
                                   std::to_string(edge) + "/block=" + std::to_string(block) +
                                   "/T=" + std::to_string(threads);
          register_bench(name, [edge, block, threads](benchmark::State& state) {
            run_gemm(state, edge, Variant::parallel, block, threads);
          });
        }
      }
    }
  }
}
