// Benchmark entry point.
//
// Adds --pcl_* workload controls on top of Google Benchmark's own flags and
// writes a `*.meta.json` sidecar next to the raw JSON results so that every
// measurement can be traced back to the machine, compiler and flags that
// produced it.
#include <cstdio>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

#include "bench_support.hpp"

int main(int argc, char** argv) {
  const pcl_bench::RunContext parsed = pcl_bench::parse_context(argc, argv);
  pcl_bench::context() = parsed;

  std::printf("ParallelComputeLab benchmark harness\n");
  std::printf("  extra flags:\n");
  for (const std::string& flag : pcl_bench::context_flag_names()) {
    std::printf("    %s\n", flag.c_str());
  }
  std::printf("\n");

  // Google Benchmark rejects any flag it does not know about, so the --pcl_*
  // flags have to be removed from argv before it parses the command line.
  // They have already been captured into the RunContext above.
  std::vector<char*> filtered;
  filtered.push_back(argv[0]);
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg.rfind("--pcl_", 0) != 0) {
      filtered.push_back(argv[i]);
    }
  }
  int filtered_argc = static_cast<int>(filtered.size());

  benchmark::Initialize(&filtered_argc, filtered.data());
  if (benchmark::ReportUnrecognizedArguments(filtered_argc, filtered.data())) {
    std::fprintf(stderr, "[pcl] unrecognised command line arguments; aborting\n");
    return 2;
  }

  pcl_bench::register_vector_ops_benchmarks();
  pcl_bench::register_reduction_benchmarks();
  pcl_bench::register_matmul_benchmarks();
  pcl_bench::register_memory_benchmarks();
  pcl_bench::register_scheduling_benchmarks();
  pcl_bench::register_blas_benchmarks();

  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  // The sidecar is derived from --benchmark_out so the two always match.
  std::string meta = pcl_bench::context().raw_output;
  if (!meta.empty()) {
    const auto dot = meta.find_last_of('.');
    if (dot != std::string::npos && meta.substr(dot) == ".json") {
      meta = meta.substr(0, dot);
    }
    meta += ".meta.json";
  } else {
    meta = "pcl_benchmark.meta.json";
  }
  pcl_bench::write_metadata(pcl_bench::context(), meta);
  return 0;
}
