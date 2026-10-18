// Memory-bound workload benchmarks: copy, scale, triad, sequential read,
// strided read.
//
// Byte counts follow the convention documented in memory/workloads.hpp and in
// docs/BENCHMARKING.md.  For the strided workload the reported byte count is a
// LOWER BOUND: the hardware actually moves whole cache lines, so the achieved
// "bandwidth" for a large stride is a small fraction of the DRAM bandwidth.
#include <string>
#include <vector>

#include "bench_support.hpp"
#include "parallel_compute/memory/workloads.hpp"

using parallel_compute::Backend;
namespace mem = parallel_compute::memory;
namespace par = parallel_compute::parallel;

namespace {

enum class Workload { copy, scale, triad, sequential, strided };

const char* workload_key(Workload w) {
  switch (w) {
    case Workload::copy:
      return "copy";
    case Workload::scale:
      return "scale";
    case Workload::triad:
      return "triad";
    case Workload::sequential:
      return "sequential";
    case Workload::strided:
      return "strided";
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

constexpr double kAlpha = 1.5;
constexpr std::size_t kStridedElements = 1u << 16;  // window covered by the strided sweep
constexpr std::size_t kStridedStride = 16;          // 128 B for f64: > one cache line

bool validate(Workload w, const pcl_bench::VectorData& data, Backend backend) {
  switch (w) {
    case Workload::copy: {
      std::vector<double> got(data.a.size());
      mem::copy<double>(data.a, got, backend);
      return pcl_bench::results_agree(data.a, got, pcl_bench::elementwise_tolerance());
    }
    case Workload::scale: {
      std::vector<double> want(data.y), got(data.y);
      mem::scale<double>(want, kAlpha, Backend::reference);
      mem::scale<double>(got, kAlpha, backend);
      return pcl_bench::results_agree(want, got, pcl_bench::elementwise_tolerance());
    }
    case Workload::triad: {
      std::vector<double> want(data.y.size()), got(data.y.size());
      mem::triad<double>(want, data.b, data.ref, kAlpha, Backend::reference);
      mem::triad<double>(got, data.b, data.ref, kAlpha, backend);
      return pcl_bench::results_agree(want, got, pcl_bench::elementwise_tolerance());
    }
    case Workload::sequential: {
      const double want = mem::sequential_sum<double>(data.a, Backend::reference);
      const double got = mem::sequential_sum<double>(data.a, backend);
      return parallel_compute::approx_equal<double>(want, got, pcl_bench::reduction_tolerance());
    }
    case Workload::strided: {
      const auto window = std::span<const double>(data.a).first(kStridedElements);
      const double want = mem::strided_sum<double>(window, kStridedStride, Backend::reference);
      const double got = mem::strided_sum<double>(window, kStridedStride, backend);
      return parallel_compute::approx_equal<double>(want, got, pcl_bench::reduction_tolerance());
    }
  }
  return false;
}

void run_workload(benchmark::State& state, Workload w, std::size_t n, const Variant& variant,
                  int threads) {
  pcl_bench::VectorData data(n);
  if (variant.parallel_capable) {
    par::set_thread_count(threads);
  }
  if (!validate(w, data, variant.backend)) {
    state.SkipWithError(std::string(workload_key(w)) +
                        " variant disagrees with the scalar reference");
    return;
  }

  const std::size_t count = (w == Workload::strided) ? kStridedElements : n;
  const std::size_t bytes = mem::bytes_touched(
      w == Workload::copy     ? mem::Workload::copy
      : w == Workload::scale  ? mem::Workload::scale
      : w == Workload::triad  ? mem::Workload::triad
      : w == Workload::sequential ? mem::Workload::sequential
                                  : mem::Workload::strided,
      count, sizeof(double));

  const char* schedule = variant.parallel_capable ? "static" : "none";
  state.SetLabel(pcl_bench::make_label(variant.name, threads, schedule, 1));
  pcl_bench::set_common_counters(state, count, bytes, threads, true, schedule);
  if (w == Workload::strided) {
    state.counters["stride"] = static_cast<double>(kStridedStride);
  }

  double sink = 0.0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(data.a.data());
    switch (w) {
      case Workload::copy:
        mem::copy<double>(data.a, data.y, variant.backend);
        break;
      case Workload::scale:
        mem::scale<double>(data.y, kAlpha, variant.backend);
        break;
      case Workload::triad:
        mem::triad<double>(data.y, data.b, data.ref, kAlpha, variant.backend);
        break;
      case Workload::sequential:
        sink = mem::sequential_sum<double>(data.a, variant.backend);
        break;
      case Workload::strided: {
        const auto window = std::span<const double>(data.a).first(kStridedElements);
        sink = mem::strided_sum<double>(window, kStridedStride, variant.backend);
        break;
      }
    }
    benchmark::DoNotOptimize(data.y.data());
    benchmark::DoNotOptimize(sink);
  }
  pcl_bench::set_rate_counters(state, bytes, 0);
}

}  // namespace

void pcl_bench::register_memory_benchmarks() {
  const RunContext& ctx = context();
  const std::vector<Workload> workloads = {Workload::copy, Workload::scale, Workload::triad,
                                           Workload::sequential, Workload::strided};
  for (Workload w : workloads) {
    for (std::size_t n : ctx.vector_sizes) {
      // The strided sweep reads a fixed 512 KiB window, so only one size is
      // meaningful for it.
      if (w == Workload::strided && n != ctx.vector_sizes.back()) continue;
      for (const Variant& v : variants(par::openmp_enabled())) {
        if (v.parallel_capable) {
          for (int threads : ctx.thread_counts) {
            const std::string name = std::string("PCL/mem_") + workload_key(w) + "/f64/n=" +
                                     std::to_string(n) + "/" + v.name + "/T=" +
                                     std::to_string(threads);
            register_bench(name, [w, n, v, threads](benchmark::State& state) {
              run_workload(state, w, n, v, threads);
            });
          }
        } else {
          const std::string name = std::string("PCL/mem_") + workload_key(w) + "/f64/n=" +
                                   std::to_string(n) + "/" + v.name + "/T=1";
          register_bench(name, [w, n, v](benchmark::State& state) {
            run_workload(state, w, n, v, 1);
          });
        }
      }
    }
  }
}
