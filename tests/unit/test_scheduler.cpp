// Contract tests for the scheduling harness.
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "parallel_compute/parallel/scheduler.hpp"
#include "test_support.hpp"

namespace par = parallel_compute::parallel;

namespace {

TEST(Scheduler, ScheduleNamesRoundTrip) {
  for (auto s : {par::Schedule::serial, par::Schedule::omp_static, par::Schedule::omp_dynamic,
                 par::Schedule::omp_guided}) {
    par::Schedule parsed = par::Schedule::serial;
    ASSERT_TRUE(par::schedule_from_name(par::schedule_name(s), parsed));
    EXPECT_EQ(s, parsed);
  }
  par::Schedule out = par::Schedule::serial;
  EXPECT_FALSE(par::schedule_from_name("does_not_exist", out));
}

TEST(Scheduler, SerialPolicyVisitsEveryIndexExactlyOnce) {
  for (std::size_t n : {0U, 1U, 3U, 17U, 1000U}) {
    std::vector<int> hits(n, 0);
    par::ParallelConfig cfg;
    cfg.schedule = par::Schedule::serial;
    par::parallel_for(n, cfg, [&](std::size_t i) { hits[i] += 1; });
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(1, hits[i]) << "i=" << i << " n=" << n;
  }
}

TEST(Scheduler, EveryOpenMpPolicyCoversTheWholeRange) {
  const int threads = std::max(1, par::max_threads());
  for (auto sched : {par::Schedule::omp_static, par::Schedule::omp_dynamic,
                     par::Schedule::omp_guided}) {
    for (std::size_t n : {0U, 1U, 5U, 64U, 257U}) {
      std::vector<std::atomic<int>> hits(n);
      for (auto& h : hits) h.store(0);
      par::ParallelConfig cfg;
      cfg.threads = threads;
      cfg.schedule = sched;
      cfg.chunk = 4;
      par::parallel_for(n, cfg, [&](std::size_t i) { hits[i].fetch_add(1); });
      for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(1, hits[i].load()) << "n=" << n << " i=" << i
                                    << " schedule=" << par::schedule_name(sched);
      }
    }
  }
}

TEST(Scheduler, ChunkedRangesPartitionExactly) {
  const int threads = std::max(1, par::max_threads());
  for (std::size_t n : {0U, 1U, 7U, 100U, 1000U}) {
    std::vector<int> covered(n, 0);
    par::ParallelConfig cfg;
    cfg.threads = threads;
    par::parallel_for_chunks(n, cfg, [&](std::size_t begin, std::size_t count) {
      for (std::size_t i = begin; i < begin + count; ++i) covered[i] += 1;
    });
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(1, covered[i]) << "n=" << n << " i=" << i;
  }
}

TEST(Scheduler, ReduceSumsEveryContribution) {
  const int threads = std::max(1, par::max_threads());
  for (auto sched : {par::Schedule::serial, par::Schedule::omp_static, par::Schedule::omp_dynamic,
                     par::Schedule::omp_guided}) {
    par::ParallelConfig cfg;
    cfg.threads = threads;
    cfg.schedule = sched;
    const std::size_t n = 1000;
    const std::uint64_t got = par::parallel_reduce<std::uint64_t>(n, cfg, 0,
                                                                   [](std::size_t i) {
                                                                     return i + 1;
                                                                   });
    EXPECT_EQ(n * (n + 1) / 2, got) << "schedule=" << par::schedule_name(sched);
    EXPECT_EQ(std::uint64_t{7},
              par::parallel_reduce<std::uint64_t>(0, cfg, 7, [](std::size_t) {
                return std::uint64_t{1};
              }));
  }
}

TEST(Scheduler, WorkloadChecksumIsDeterministicAndScheduleIndependent) {
  const std::size_t items = 500;
  const std::uint64_t seed = 99;
  par::ParallelConfig serial;
  serial.schedule = par::Schedule::serial;
  const std::uint64_t reference = par::scheduling_workload(items, 1, 9, serial, seed);

  for (auto sched : {par::Schedule::omp_static, par::Schedule::omp_dynamic,
                     par::Schedule::omp_guided}) {
    for (std::size_t chunk : {1U, 3U, 32U, 0U}) {
      par::ParallelConfig cfg;
      cfg.threads = std::max(1, par::max_threads());
      cfg.schedule = sched;
      cfg.chunk = chunk;
      EXPECT_EQ(reference, par::scheduling_workload(items, 1, 9, cfg, seed))
          << "schedule=" << par::schedule_name(sched) << " chunk=" << chunk;
    }
  }
  // A different seed must produce a different checksum (otherwise the workload
  // would be a constant and the benchmark meaningless).
  EXPECT_NE(reference, par::scheduling_workload(items, 1, 9, serial, seed + 1));
}

TEST(Scheduler, WorkloadEdgeCases) {
  par::ParallelConfig cfg;
  EXPECT_EQ(0U, par::scheduling_workload(0, 1, 4, cfg, 1));
  const std::uint64_t one = par::scheduling_workload(1, 0, 0, cfg, 1);
  // With min_units == max_units == 0 the checksum is just the index-derived
  // value; it must still be deterministic.
  EXPECT_EQ(one, par::scheduling_workload(1, 0, 0, cfg, 1));
}

TEST(Scheduler, ThreadCountHelpers) {
  EXPECT_GE(par::max_threads(), 1);
  EXPECT_GE(par::current_thread_count(), 1);
  if (!par::openmp_enabled()) {
    EXPECT_EQ(1, par::max_threads());
    EXPECT_FALSE(par::set_thread_count(4));
    EXPECT_FALSE(par::set_thread_count(0));
  } else {
    const int original = par::current_thread_count();
    ASSERT_TRUE(par::set_thread_count(2));
    EXPECT_EQ(2, par::current_thread_count());
    EXPECT_FALSE(par::set_thread_count(-1));
    EXPECT_TRUE(par::set_thread_count(original));
    EXPECT_EQ(original, par::current_thread_count());
  }
}

TEST(Scheduler, FalseSharingSlotsAreCacheLineSeparated) {
  constexpr std::size_t kCacheLine = 64;
  const auto buffer = par::false_sharing_slots(8);
  ASSERT_EQ(8U * par::FalseSharingBuffer::kSlotsPerLine, buffer.storage.size());
  // The buffer itself must be cache-line aligned, otherwise "separated" is a
  // claim about the wrong lines.
  EXPECT_EQ(0U, reinterpret_cast<std::uintptr_t>(buffer.storage.data()) % kCacheLine);
  for (std::size_t i = 0; i + 1 < 8; ++i) {
    const auto a = reinterpret_cast<std::uintptr_t>(buffer.slot(i));
    const auto b = reinterpret_cast<std::uintptr_t>(buffer.slot(i + 1));
    EXPECT_EQ(kCacheLine, b - a) << "i=" << i;
  }
}

}  // namespace
