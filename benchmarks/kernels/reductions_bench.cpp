// Reduction benchmarks: sum, min, max, dot, mean.
//
// min/max have no explicit SIMD kernel (see docs/LIMITATIONS.md), so the `simd`
// entry for them intentionally reports the same serial fold as
// `auto_vectorized`; the comparison is kept so that claim is visible in the
// results rather than only in the documentation.
#include <cmath>
#include <string>
#include <vector>

#include "bench_support.hpp"
#include "parallel_compute/algorithms/reductions.hpp"

using parallel_compute::Backend;
namespace par = parallel_compute::parallel;
namespace red = parallel_compute::reductions;

namespace {

enum class Reduction { sum, min, max, dot, mean };

const char* reduction_name(Reduction r) {
  switch (r) {
    case Reduction::sum:
      return "sum";
    case Reduction::min:
      return "min";
    case Reduction::max:
      return "max";
    case Reduction::dot:
      return "dot";
    case Reduction::mean:
      return "mean";
  }
  return "unknown";
}

struct Variant {
  const char* name;
  Backend backend;
  bool parallel_capable;
};

std::vector<Variant> variants(bool allow_openmp) {
  std::vector<Variant> v = {{"reference", Backend::reference, false},
                            {"auto_vectorized", Backend::auto_vectorized, false},
                            {"simd", Backend::simd, false}};
  if (allow_openmp) {
    v.push_back({"openmp", Backend::openmp, true});
  }
  return v;
}

/// min/max are comparisons, not sums: their result is exact for every backend,
/// so an exact comparison is the right one.  sum/mean/dot reassociate, so they
/// are checked against the forward-error bound of the problem.
bool validate(Reduction r, const pcl_bench::VectorData& data, Backend backend) {
  const auto a = std::span<const double>(data.a);
  const auto b = std::span<const double>(data.b);
  switch (r) {
    case Reduction::min:
      return red::min_value<double>(a, backend) == red::min_value<double>(a, Backend::reference);
    case Reduction::max:
      return red::max_value<double>(a, backend) == red::max_value<double>(a, Backend::reference);
    case Reduction::sum:
    case Reduction::mean: {
      const double want = r == Reduction::sum
                              ? red::sum<double>(a, Backend::reference)
                              : red::mean<double>(a, Backend::reference);
      const double got = r == Reduction::sum ? red::sum<double>(a, backend)
                                             : red::mean<double>(a, backend);
      const double bound =
          pcl_bench::reassociation_abs_tolerance(a.size(), pcl_bench::sum_abs(data.a));
      return pcl_bench::results_within({want}, {got}, bound);
    }
    case Reduction::dot: {
      const double want = red::dot<double>(a, b, Backend::reference);
      const double got = red::dot<double>(a, b, backend);
      std::vector<double> products(a.size());
      for (std::size_t i = 0; i < a.size(); ++i) products[i] = std::fabs(a[i] * b[i]);
      const double bound =
          pcl_bench::reassociation_abs_tolerance(a.size(), pcl_bench::sum_abs(products));
      return pcl_bench::results_within({want}, {got}, bound);
    }
  }
  return false;
}

void run_reduction(benchmark::State& state, Reduction r, std::size_t n, const Variant& variant,
                   int threads) {
  pcl_bench::VectorData data(n);
  if (variant.parallel_capable) {
    par::set_thread_count(threads);
  }
  if (!validate(r, data, variant.backend)) {
    state.SkipWithError(std::string(reduction_name(r)) +
                        " variant disagrees with the scalar reference");
    return;
  }

  const std::size_t bytes =
      (r == Reduction::dot ? 2 : 1) * n * sizeof(double);  // documented convention
  const char* schedule = variant.parallel_capable ? "static" : "none";
  state.SetLabel(pcl_bench::make_label(variant.name, threads, schedule, 1));
  pcl_bench::set_common_counters(state, n, bytes, threads, true, schedule);

  const auto a = std::span<const double>(data.a);
  const auto b = std::span<const double>(data.b);
  double sink = 0.0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(data.a.data());
    switch (r) {
      case Reduction::sum:
        sink = red::sum<double>(a, variant.backend);
        break;
      case Reduction::min:
        sink = red::min_value<double>(a, variant.backend);
        break;
      case Reduction::max:
        sink = red::max_value<double>(a, variant.backend);
        break;
      case Reduction::dot:
        sink = red::dot<double>(a, b, variant.backend);
        break;
      case Reduction::mean:
        sink = red::mean<double>(a, variant.backend);
        break;
    }
    benchmark::DoNotOptimize(sink);
  }
  pcl_bench::set_rate_counters(state, bytes, 0);
}

}  // namespace

void pcl_bench::register_reduction_benchmarks() {
  const RunContext& ctx = context();
  const std::vector<Reduction> reductions = {Reduction::sum,  Reduction::min, Reduction::max,
                                             Reduction::dot,  Reduction::mean};
  for (Reduction r : reductions) {
    for (std::size_t n : ctx.vector_sizes) {
      for (const Variant& v : variants(par::openmp_enabled())) {
        if (v.parallel_capable) {
          for (int threads : ctx.thread_counts) {
            const std::string name = std::string("PCL/reduce_") + reduction_name(r) + "/f64/n=" +
                                     std::to_string(n) + "/" + v.name + "/T=" +
                                     std::to_string(threads);
            register_bench(name, [r, n, v, threads](benchmark::State& state) {
              run_reduction(state, r, n, v, threads);
            });
          }
        } else {
          const std::string name = std::string("PCL/reduce_") + reduction_name(r) + "/f64/n=" +
                                   std::to_string(n) + "/" + v.name + "/T=1";
          register_bench(name, [r, n, v](benchmark::State& state) {
            run_reduction(state, r, n, v, 1);
          });
        }
      }
    }
  }
}
