// External BLAS baseline.
//
// Registered only when a BLAS was found at configure time.  The kernel is the
// same C = A * B product as every other GEMM variant; only the implementation
// is external, so any difference is attributable to that library.
//
// Note: whichever BLAS was found is recorded verbatim in the result metadata
// (`build.blas`), because a reference (unoptimised) netlib BLAS is *slower*
// than a cache-blocked kernel of our own and saying so is part of the finding.
#include <string>
#include <vector>

#include "bench_support.hpp"
#include "parallel_compute/algorithms/matmul.hpp"

namespace gm = parallel_compute::gemm;
using parallel_compute::MutableMatrixView;
using parallel_compute::MatrixView;

namespace {

void run_blas_gemm(benchmark::State& state, std::size_t edge, int threads) {
  const std::size_t m = edge, n = edge, k = edge;
  pcl_bench::MatrixData data(m, n, k);
  if (threads > 1) {
    parallel_compute::parallel::set_thread_count(threads);
  }

  const auto av = MatrixView<const double>(data.a.data(), m, k);
  const auto bv = MatrixView<const double>(data.b.data(), k, n);

  std::vector<double> check(m * n, 0.0);
  if (!gm::matmul_blas<double>(av, bv, MutableMatrixView<double>(check.data(), m, n))) {
    state.SkipWithError("no BLAS available in this build");
    return;
  }
  std::vector<double> products(k);
  for (std::size_t p = 0; p < k; ++p) {
    double acc = 0.0;
    for (std::size_t q = 0; q < k; ++q) acc += std::fabs(data.a[q] * data.b[p * n + q]);
    products[p] = acc;
  }
  const double bound =
      pcl_bench::gemm_abs_tolerance(m, n, k, pcl_bench::sum_abs(products));
  if (!pcl_bench::results_within(data.ref, check, bound)) {
    state.SkipWithError("BLAS GEMM disagrees with the scalar reference GEMM");
    return;
  }

  const std::size_t flops = gm::flop_count(m, n, k);
  const std::size_t bytes = 2 * (m * n + k * n + m * k) * sizeof(double);
  state.SetLabel(pcl_bench::make_label(gm::blas_name(), threads, "library", 0));
  pcl_bench::set_common_counters(state, m * n, bytes, threads, true, "library");
  state.counters["m"] = static_cast<double>(m);
  state.counters["flops_per_iter"] = static_cast<double>(flops);

  double* c = data.c.data();
  for (auto _ : state) {
    benchmark::DoNotOptimize(data.a.data());
    gm::matmul_blas<double>(av, bv, MutableMatrixView<double>(c, m, n));
    benchmark::DoNotOptimize(c);
  }
  pcl_bench::set_rate_counters(state, bytes, flops);
}

}  // namespace

void pcl_bench::register_blas_benchmarks() {
  if (!gm::blas_available()) return;
  const RunContext& ctx = context();
  for (std::size_t edge : ctx.matrix_sizes) {
    // A reference BLAS is single threaded; only sweep thread counts when the
    // linked library actually uses them (unknown here), so report T=1 and say
    // so in the label.
    const std::string name =
        "PCL/gemm/f64/blas/n=" + std::to_string(edge) + "/T=1";
    register_bench(name, [edge](benchmark::State& state) { run_blas_gemm(state, edge, 1); });
  }
}
