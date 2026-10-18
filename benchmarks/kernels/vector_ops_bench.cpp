// Element-wise kernel benchmarks.
//
// Registered combinations: PCL/{kernel}/f64/n={elements}/{backend}/T={threads}
//   backend in {reference, auto_vectorized, simd, openmp}
//
// Each entry is validated against the scalar reference *before* the timing
// loop starts; a variant that disagrees is skipped with a reason instead of
// being reported as a (wrong) measurement.
#include <string>
#include <vector>

#include "bench_support.hpp"
#include "parallel_compute/algorithms/vector_ops.hpp"

using parallel_compute::Backend;
namespace par = parallel_compute::parallel;
namespace vops = parallel_compute::vector_ops;

namespace {

enum class Kernel { add, multiply, scale, saxpy };

const char* kernel_name(Kernel k) {
  switch (k) {
    case Kernel::add:
      return "add";
    case Kernel::multiply:
      return "multiply";
    case Kernel::scale:
      return "scale";
    case Kernel::saxpy:
      return "saxpy";
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

constexpr double kAlpha = 2.5;

/// Verify the variant against the scalar reference on the fixture's data.
bool validate(Kernel kernel, const pcl_bench::VectorData& data, Backend backend) {
  const std::size_t n = data.a.size();
  std::vector<double> got(n, 0.0);
  switch (kernel) {
    case Kernel::add:
      vops::add<double>(data.a, data.b, got, backend);
      return pcl_bench::results_agree(data.ref, got, pcl_bench::elementwise_tolerance());
    case Kernel::multiply: {
      // Reference product of the fixture inputs.
      std::vector<double> want(n);
      vops::multiply<double>(data.a, data.b, want, Backend::reference);
      vops::multiply<double>(data.a, data.b, got, backend);
      return pcl_bench::results_agree(want, got, pcl_bench::elementwise_tolerance());
    }
    case Kernel::scale: {
      std::vector<double> want(data.y);
      vops::scale<double>(want, kAlpha, Backend::reference);
      vops::scale<double>(got, kAlpha, backend);
      return pcl_bench::results_agree(want, got, pcl_bench::elementwise_tolerance());
    }
    case Kernel::saxpy: {
      std::vector<double> want(data.y);
      vops::saxpy<double>(want, data.a, kAlpha, Backend::reference);
      std::vector<double> y2(data.y);
      vops::saxpy<double>(y2, data.a, kAlpha, backend);
      return pcl_bench::results_agree(want, y2, pcl_bench::elementwise_tolerance());
    }
  }
  return false;
}

void run_kernel(benchmark::State& state, Kernel kernel, std::size_t n, const Variant& variant,
                int threads) {
  pcl_bench::VectorData data(n);

  if (variant.parallel_capable) {
    par::set_thread_count(threads);
  }
  if (!validate(kernel, data, variant.backend)) {
    state.SkipWithError(std::string(kernel_name(kernel)) + " variant disagrees with the scalar reference");
    return;
  }

  const std::size_t bytes = vops::bytes_touched<double>(n);
  const char* schedule = variant.parallel_capable ? "static" : "none";
  state.SetLabel(pcl_bench::make_label(variant.name, threads, schedule, 1));
  pcl_bench::set_common_counters(state, n, bytes, threads, true, schedule);

  for (auto _ : state) {
    benchmark::DoNotOptimize(data.a.data());
    switch (kernel) {
      case Kernel::add:
        vops::add<double>(data.a, data.b, data.y, variant.backend);
        break;
      case Kernel::multiply:
        vops::multiply<double>(data.a, data.b, data.y, variant.backend);
        break;
      case Kernel::scale:
        vops::scale<double>(data.y, kAlpha, variant.backend);
        break;
      case Kernel::saxpy:
        vops::saxpy<double>(data.y, data.a, kAlpha, variant.backend);
        break;
    }
    benchmark::DoNotOptimize(data.y.data());
  }
  pcl_bench::set_rate_counters(state, bytes, 0);
}

}  // namespace

void pcl_bench::register_vector_ops_benchmarks() {
  const RunContext& ctx = context();
  const std::vector<Kernel> kernels = {Kernel::add, Kernel::multiply, Kernel::scale,
                                       Kernel::saxpy};
  for (Kernel kernel : kernels) {
    for (std::size_t n : ctx.vector_sizes) {
      for (const Variant& v : variants(par::openmp_enabled())) {
        // Only the parallel variant sweeps thread counts; the serial variants
        // are single threaded by definition.
        if (v.parallel_capable) {
          for (int threads : ctx.thread_counts) {
            const std::string name = std::string("PCL/") + kernel_name(kernel) + "/f64/n=" +
                                     std::to_string(n) + "/" + v.name + "/T=" +
                                     std::to_string(threads);
            register_bench(name, [kernel, n, v, threads](benchmark::State& state) {
              run_kernel(state, kernel, n, v, threads);
            });
          }
        } else {
          const std::string name = std::string("PCL/") + kernel_name(kernel) + "/f64/n=" +
                                   std::to_string(n) + "/" + v.name + "/T=1";
          register_bench(name, [kernel, n, v](benchmark::State& state) {
            run_kernel(state, kernel, n, v, 1);
          });
        }
      }
    }
  }
}
