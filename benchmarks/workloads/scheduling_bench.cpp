// Parallel scheduling experiments and a false-sharing micro-benchmark.
//
// The scheduling workload gives each item a *different* cost (deterministically
// derived from its index), which is the situation where static vs dynamic vs
// guided actually differ: with uniform cost, static scheduling is optimal by
// construction and the comparison would prove nothing.
//
// The false-sharing benchmark contrasts per-thread counters packed 8 bytes
// apart against counters on separate 64-byte cache lines.
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "bench_support.hpp"
#include "parallel_compute/parallel/scheduler.hpp"

namespace par = parallel_compute::parallel;

namespace {

struct ScheduleConfig {
  const char* name;
  par::Schedule schedule;
};

const std::vector<ScheduleConfig>& schedule_configs() {
  static const std::vector<ScheduleConfig> configs = {
      {"serial", par::Schedule::serial},
      {"static", par::Schedule::omp_static},
      {"dynamic", par::Schedule::omp_dynamic},
      {"guided", par::Schedule::omp_guided},
  };
  return configs;
}

constexpr std::size_t kSchedulingItems = 20000;
constexpr std::size_t kMinUnits = 8;
constexpr std::size_t kMaxUnits = 96;

void run_scheduling(benchmark::State& state, const ScheduleConfig& cfg, std::size_t items,
                    std::size_t chunk, int threads) {
  par::ParallelConfig pc;
  pc.schedule = cfg.schedule;
  pc.threads = threads;
  pc.chunk = chunk;

  // Pre-timing gate: the workload checksum must be identical for every
  // policy, otherwise the policies are not computing the same thing.
  par::ParallelConfig serial;
  serial.schedule = par::Schedule::serial;
  const std::uint64_t reference =
      par::scheduling_workload(items, kMinUnits, kMaxUnits, serial, 0xC0FFEEULL);
  if (par::scheduling_workload(items, kMinUnits, kMaxUnits, pc, 0xC0FFEEULL) != reference) {
    state.SkipWithError("scheduling workload checksum differs between policies");
    return;
  }

  std::uint64_t sink = 0;
  for (auto _ : state) {
    sink += par::scheduling_workload(items, kMinUnits, kMaxUnits, pc, 0xC0FFEEULL);
    benchmark::DoNotOptimize(sink);
  }
  state.counters["items"] = static_cast<double>(items);
  state.counters["threads"] = threads;
  state.counters["chunk"] = static_cast<double>(chunk);
  state.counters["validated"] = 1.0;
  state.counters["mean_units"] =
      static_cast<double>((kMinUnits + kMaxUnits) / 2);
  state.SetLabel(pcl_bench::make_label(cfg.name, threads, cfg.name, chunk));
  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) *
                           static_cast<std::int64_t>(items));
}

// --- false sharing ---------------------------------------------------------
// The packed layout deliberately places the counters 8 bytes apart so that two
// threads update the same 64-byte cache line; the padded layout gives each
// thread its own line.  Everything else -- the work per thread and the number
// of threads -- is identical, so the difference is attributable to sharing.
void run_false_sharing(benchmark::State& state, bool padded, int threads) {
  par::set_thread_count(threads);
  auto buffer = par::false_sharing_slots(static_cast<std::size_t>(threads));
  // Fixed capacity: the benchmark is registered for several thread counts in
  // ascending order, so a vector sized from the first call would be too small
  // for the later ones.
  static std::vector<std::uint64_t> packed(4096, 0);
  for (auto* p = buffer.storage.data(); p != buffer.storage.data() + buffer.storage.size(); ++p) {
    *p = 0;
  }
  std::fill(packed.begin(), packed.begin() + threads, std::uint64_t{0});

  for (auto _ : state) {
    par::ParallelConfig pc;
    pc.threads = threads;
    pc.schedule = par::Schedule::omp_static;
    par::parallel_for(static_cast<std::size_t>(threads), pc, [&](std::size_t tid) {
      std::uint64_t local = 0;
      for (std::uint64_t i = 0; i < 20000; ++i) {
        local += static_cast<std::uint64_t>(tid) * 0x9E3779B97F4A7C15ULL + i;
        local ^= local >> 29;
      }
      if (padded) {
        *buffer.slot(tid) += local;
      } else {
        packed[tid] += local;
      }
    });
    benchmark::DoNotOptimize(buffer.storage.data());
    benchmark::DoNotOptimize(packed.data());
  }
  state.counters["threads"] = threads;
  state.counters["validated"] = 1.0;
  state.counters["bytes_per_iter"] =
      static_cast<double>(padded ? static_cast<std::size_t>(threads) * 64
                                   : static_cast<std::size_t>(threads) *
                                         sizeof(std::uint64_t));
  state.SetLabel(pcl_bench::make_label(padded ? "cache_line_padded" : "packed_8b", threads,
                                       "static", 1));
}

}  // namespace

void pcl_bench::register_scheduling_benchmarks() {
  const RunContext& ctx = context();
  if (!par::openmp_enabled()) return;

  for (const ScheduleConfig& cfg : schedule_configs()) {
    const bool serial = cfg.schedule == par::Schedule::serial;
    std::vector<std::size_t> chunks{0};
    if (!serial) {
      chunks.clear();
      for (int c : ctx.schedule_chunks) chunks.push_back(static_cast<std::size_t>(c));
    }
    for (std::size_t chunk : chunks) {
      for (int threads : ctx.thread_counts) {
        const std::string name = "PCL/scheduling/" + std::string(cfg.name) + "/items=" +
                                 std::to_string(kSchedulingItems) + "/chunk=" +
                                 std::to_string(chunk) + "/T=" + std::to_string(threads);
        register_bench(name, [cfg, chunk, threads](benchmark::State& state) {
          run_scheduling(state, cfg, kSchedulingItems, chunk, threads);
        });
      }
    }
  }

  for (int threads : ctx.thread_counts) {
    if (threads < 2) continue;
    register_bench("PCL/false_sharing/packed_8b/T=" + std::to_string(threads),
                   [threads](benchmark::State& state) { run_false_sharing(state, false, threads); });
    register_bench("PCL/false_sharing/cache_line_padded/T=" + std::to_string(threads),
                   [threads](benchmark::State& state) { run_false_sharing(state, true, threads); });
  }
}
