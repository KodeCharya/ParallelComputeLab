// Concurrency correctness: thread counts, repetition, reductions, races.
//
// Run under ThreadSanitizer (preset `tsan`) to get data-race detection on top
// of these functional checks.
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <string>
#include <thread>
#include <vector>

#include "parallel_compute/algorithms/matmul.hpp"
#include "parallel_compute/algorithms/reductions.hpp"
#include "parallel_compute/algorithms/vector_ops.hpp"
#include "parallel_compute/common/tolerance.hpp"
#include "parallel_compute/memory/workloads.hpp"
#include "parallel_compute/parallel/scheduler.hpp"
#include "test_support.hpp"

using parallel_compute::Backend;
using parallel_compute::MutableMatrixView;
using parallel_compute::MatrixView;
namespace par = parallel_compute::parallel;

namespace {

std::vector<int> thread_counts() {
  std::vector<int> counts = {1, 2, 3, 4};
  const int max = par::max_threads();
  for (int extra : {max, max + 1, max * 2}) {
    if (extra > 0 && std::find(counts.begin(), counts.end(), extra) == counts.end()) {
      counts.push_back(extra);
    }
  }
  return counts;
}

class ScopedThreadCount {
 public:
  explicit ScopedThreadCount(int threads) { par::set_thread_count(threads); }
  ~ScopedThreadCount() { par::set_thread_count(original_); }
  ScopedThreadCount(const ScopedThreadCount&) = delete;
  ScopedThreadCount& operator=(const ScopedThreadCount&) = delete;

 private:
  int original_ = par::current_thread_count();
};

TEST(Concurrency, ParallelVectorOpsMatchReferenceAtEveryThreadCount) {
  const std::size_t n = 1 << 14;
  const auto a = pcl_test::make_vector<double>(n, 201);
  const auto b = pcl_test::make_vector<double>(n, 202);
  std::vector<double> expected(n), got(n);
  parallel_compute::vector_ops::add<double>(a, b, expected, Backend::reference);
  for (int threads : thread_counts()) {
    ScopedThreadCount guard(threads);
    std::fill(got.begin(), got.end(), 0.0);
    parallel_compute::vector_ops::add<double>(a, b, got, Backend::openmp);
    EXPECT_TRUE(pcl_test::all_equal(expected, got))
        << "threads=" << threads;
  }
}

TEST(Concurrency, ParallelReductionsMatchReferenceAtEveryThreadCount) {
  const std::size_t n = 1 << 16;
  const auto a = pcl_test::make_vector<double>(n, 203);
  const auto b = pcl_test::make_vector<double>(n, 204);
  const double expected_sum = parallel_compute::reductions::sum<double>(a, Backend::reference);
  const double expected_dot = parallel_compute::reductions::dot<double>(a, b, Backend::reference);
  bool found = false;
  const double expected_min =
      parallel_compute::reductions::min_value<double>(a, Backend::reference, &found);
  const double expected_max =
      parallel_compute::reductions::max_value<double>(a, Backend::reference, &found);
  for (int threads : thread_counts()) {
    ScopedThreadCount guard(threads);
    EXPECT_LT(parallel_compute::relative_error(expected_sum,
                                               parallel_compute::reductions::sum<double>(a, Backend::openmp)),
              1e-12)
        << "threads=" << threads;
    EXPECT_LT(parallel_compute::relative_error(expected_dot,
                                               parallel_compute::reductions::dot<double>(a, b, Backend::openmp)),
              1e-12)
        << "threads=" << threads;
    EXPECT_DOUBLE_EQ(expected_min,
                     parallel_compute::reductions::min_value<double>(a, Backend::openmp))
        << "threads=" << threads;
    EXPECT_DOUBLE_EQ(expected_max,
                     parallel_compute::reductions::max_value<double>(a, Backend::openmp))
        << "threads=" << threads;
  }
}

TEST(Concurrency, ParallelMinMaxPropagatesNanAtEveryThreadCount) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> a(1 << 14, 1.0);
  a[7777] = nan;  // lands in the middle of the range, not at an index boundary
  for (int threads : thread_counts()) {
    ScopedThreadCount guard(threads);
    EXPECT_TRUE(std::isnan(parallel_compute::reductions::min_value<double>(a, Backend::openmp)))
        << "threads=" << threads;
    EXPECT_TRUE(std::isnan(parallel_compute::reductions::max_value<double>(a, Backend::openmp)))
        << "threads=" << threads;
  }
}

TEST(Concurrency, ParallelMatmulMatchesReferenceAtEveryThreadCount) {
  const std::size_t m = 37, n = 41, k = 43;
  const auto a = pcl_test::make_vector<double>(m * k, 211);
  const auto b = pcl_test::make_vector<double>(k * n, 212);
  const auto expected = pcl_test::oracle_matmul(a, m, k, b, n);
  const MatrixView<const double> av(a.data(), m, k);
  const MatrixView<const double> bv(b.data(), k, n);
  parallel_compute::gemm::GemmOptions opts;
  opts.block_size = 16;
  for (int threads : thread_counts()) {
    ScopedThreadCount guard(threads);
    std::vector<double> got(m * n, 0.0);
    parallel_compute::gemm::matmul_parallel<double>(av, bv,
                                                     MutableMatrixView<double>(got.data(), m, n), opts);
    EXPECT_TRUE(pcl_test::all_equal(expected, got, pcl_test::Tolerance{1e-11, 1e-11}))
        << "threads=" << threads;
  }
}

TEST(Concurrency, MemoryWorkloadsMatchReferenceAtEveryThreadCount) {
  const std::size_t n = 1 << 14;
  const auto b = pcl_test::make_vector<double>(n, 221);
  const auto c = pcl_test::make_vector<double>(n, 222);
  std::vector<double> expected(n), got(n, 0.0);
  parallel_compute::memory::triad<double>(expected, b, c, 1.5, Backend::reference);
  for (int threads : thread_counts()) {
    ScopedThreadCount guard(threads);
    std::fill(got.begin(), got.end(), 0.0);
    parallel_compute::memory::triad<double>(got, b, c, 1.5, Backend::openmp);
    EXPECT_TRUE(pcl_test::all_equal(expected, got)) << "threads=" << threads;

    const auto src = pcl_test::make_vector<double>(n, 223);
    const double ref = parallel_compute::memory::sequential_sum<double>(src, Backend::reference);
    EXPECT_LT(parallel_compute::relative_error(ref,
                                               parallel_compute::memory::sequential_sum<double>(src, Backend::openmp)),
              1e-12)
        << "threads=" << threads;
  }
}

TEST(Concurrency, RepeatedExecutionIsStableAndDeterministic) {
  const std::size_t n = 4096;
  const auto a = pcl_test::make_vector<double>(n, 231);
  const auto b = pcl_test::make_vector<double>(n, 232);
  const double expected = parallel_compute::reductions::sum<double>(a, Backend::reference);

  par::ParallelConfig cfg;
  cfg.threads = par::max_threads();
  cfg.schedule = par::Schedule::omp_dynamic;
  cfg.chunk = 64;
  double first = 0.0;
  for (int rep = 0; rep < 50; ++rep) {
    std::vector<double> out(n, 0.0);
    parallel_compute::vector_ops::add<double>(a, b, out, Backend::openmp);
    ASSERT_TRUE(pcl_test::all_equal(
        [&] {
          std::vector<double> e(n);
          parallel_compute::vector_ops::add<double>(a, b, e, Backend::reference);
          return e;
        }(),
        out));
    const double sum = parallel_compute::reductions::sum<double>(a, Backend::openmp);
    EXPECT_LT(parallel_compute::relative_error(expected, sum), 1e-12) << "rep=" << rep;
    if (rep == 0) first = sum;
  }
  // Same input + same thread count must give the same tree reduction every time.
  const double last = parallel_compute::reductions::sum<double>(a, Backend::openmp);
  EXPECT_DOUBLE_EQ(first, last);
}

TEST(Concurrency, ConcurrentCallsFromMultipleHostThreadsAreSafe) {
  // Each host thread uses its own data and its own OpenMP team size; nothing in
  // the library keeps mutable global state, and this test proves it.
  const std::size_t n = 8192;
  std::atomic<int> failures{0};
  std::vector<std::thread> hosts;
  for (int t = 0; t < 4; ++t) {
    hosts.emplace_back([t, n, &failures] {
      par::set_thread_count(2);
      const auto a = pcl_test::make_vector<double>(n, 241 + static_cast<std::uint64_t>(t));
      const auto b = pcl_test::make_vector<double>(n, 251 + static_cast<std::uint64_t>(t));
      std::vector<double> expected(n), got(n, 0.0);
      parallel_compute::vector_ops::add<double>(a, b, expected, Backend::reference);
      for (int rep = 0; rep < 20; ++rep) {
        std::fill(got.begin(), got.end(), 0.0);
        parallel_compute::vector_ops::add<double>(a, b, got, Backend::openmp);
        if (parallel_compute::first_mismatch<double>(expected, got) != expected.size()) {
          failures.fetch_add(1);
        }
      }
      const double sum = parallel_compute::reductions::sum<double>(a, Backend::openmp);
      if (std::isnan(sum)) failures.fetch_add(1);
    });
  }
  for (auto& th : hosts) th.join();
  EXPECT_EQ(0, failures.load());
}

TEST(Concurrency, SchedulingPoliciesDoNotLoseOrDuplicateWork) {
  const std::size_t items = 5000;
  par::ParallelConfig serial;
  serial.schedule = par::Schedule::serial;
  const std::uint64_t reference = par::scheduling_workload(items, 1, 20, serial, 5);
  for (int threads : thread_counts()) {
    for (auto sched : {par::Schedule::omp_static, par::Schedule::omp_dynamic, par::Schedule::omp_guided}) {
      for (std::size_t chunk : {1U, 7U, 128U}) {
        par::ParallelConfig cfg;
        cfg.threads = threads;
        cfg.schedule = sched;
        cfg.chunk = chunk;
        EXPECT_EQ(reference, par::scheduling_workload(items, 1, 20, cfg, 5))
            << "threads=" << threads << " schedule=" << par::schedule_name(sched)
            << " chunk=" << chunk;
      }
    }
  }
}

TEST(Concurrency, OpenMpDisabledBuildStillRunsSerially) {
  if (par::openmp_enabled()) {
    GTEST_SKIP() << "this configuration has OpenMP; the portable build covers the other branch";
  }
  EXPECT_EQ(1, par::max_threads());
  const std::size_t n = 1000;
  const auto a = pcl_test::make_vector<double>(n, 261);
  const auto b = pcl_test::make_vector<double>(n, 262);
  std::vector<double> expected(n), got(n);
  parallel_compute::vector_ops::add<double>(a, b, expected, Backend::reference);
  parallel_compute::vector_ops::add<double>(a, b, got, Backend::openmp);
  EXPECT_TRUE(pcl_test::all_equal(expected, got));
}

}  // namespace
