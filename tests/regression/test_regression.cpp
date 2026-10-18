// Regression tests for defects found while building this library.
//
// Each test names the defect, the mechanism that triggered it and the fix, so
// that a future change that reintroduces the bug fails here with an explanation
// rather than somewhere unrelated.
#include <cmath>
#include <memory>
#include <limits>
#include <vector>

#include "parallel_compute/algorithms/matmul.hpp"
#include "parallel_compute/algorithms/reductions.hpp"
#include "parallel_compute/common/tolerance.hpp"
#include "parallel_compute/memory/workloads.hpp"
#include "parallel_compute/parallel/scheduler.hpp"
#include "parallel_compute/simd/simd_kernels.hpp"
#include "test_support.hpp"

using parallel_compute::Backend;
using parallel_compute::MutableMatrixView;
using parallel_compute::MatrixView;

namespace {

// ---------------------------------------------------------------------------
// Defect 1 (2026-10-18): min_value / max_value seeded their fold with a[0].
//
// `best = a[0]` followed by `if (v < best) best = v;` silently discards a NaN
// sitting at index 0, because every subsequent comparison against NaN is false.
// The documented contract is "NaN propagates", so the fold now starts from the
// identity element (numeric_limits::max() / lowest()).
// ---------------------------------------------------------------------------
TEST(Regression, NanAtIndexZeroPropagatesThroughMinAndMax) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (auto backend : pcl_test::all_backends()) {
    const std::vector<double> nan_first = {nan, 1.0, 2.0, 3.0};
    EXPECT_TRUE(std::isnan(parallel_compute::reductions::min_value<double>(nan_first, backend)))
        << "backend=" << parallel_compute::backend_name(backend);
    EXPECT_TRUE(std::isnan(parallel_compute::reductions::max_value<double>(nan_first, backend)))
        << "backend=" << parallel_compute::backend_name(backend);

    // The same input with the NaN moved one slot along, which the old code
    // handled -- this pins the difference between the two cases.
    const std::vector<double> nan_second = {1.0, nan, 2.0, 3.0};
    EXPECT_TRUE(std::isnan(parallel_compute::reductions::min_value<double>(nan_second, backend)));
    EXPECT_TRUE(std::isnan(parallel_compute::reductions::max_value<double>(nan_second, backend)));
  }
}

// NaN must also survive the OpenMP path, whose per-thread partials are combined
// with the same NaN-propagating fold rather than the `reduction(min:x)` clause
// (whose NaN behaviour the OpenMP specification leaves unspecified).
TEST(Regression, NanPropagatesThroughTheParallelMinMaxFold) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const int original = parallel_compute::parallel::current_thread_count();
  for (int threads : {1, 2, 4, 8}) {
    parallel_compute::parallel::set_thread_count(threads);
    std::vector<double> a(1 << 12, 1.0);
    a[0] = nan;  // lands in thread 0's range for every partition
    EXPECT_TRUE(std::isnan(parallel_compute::reductions::min_value<double>(a, Backend::openmp)))
        << "threads=" << threads;
    EXPECT_TRUE(std::isnan(parallel_compute::reductions::max_value<double>(a, Backend::openmp)))
        << "threads=" << threads;
  }
  parallel_compute::parallel::set_thread_count(original);
}

// ---------------------------------------------------------------------------
// Defect 2 (2026-10-18): the 4-row SIMD GEMM micro-kernel writes two (f64) or
// four (f32) columns unconditionally.
//
// Calling it with a narrower strip would write past the end of C.  The kernel
// contract now requires ncols >= 2 / >= 4 and the caller falls back to the
// scalar i-k-j loop for narrower matrices.
// ---------------------------------------------------------------------------
TEST(Regression, SimdMatmulDoesNotOverrunForNarrowColumnStrips) {
  for (std::size_t n = 1; n <= 8; ++n) {
    const std::size_t m = 9, k = 5;  // 9 rows = two full panels + a 1-row tail
    const auto a = pcl_test::make_vector<double>(m * k, 301 + n);
    const auto b = pcl_test::make_vector<double>(k * n, 311 + n);
    const auto expected = pcl_test::oracle_matmul(a, m, k, b, n);

    // Guard bytes immediately after C catch any overrun.
    const std::size_t guard = 32;
    std::vector<double> c(m * n + guard, -1234.5);
    parallel_compute::gemm::GemmOptions opts;
    opts.block_size = 2;
    parallel_compute::gemm::matmul_simd<double>(MatrixView<const double>(a.data(), m, k),
                                                 MatrixView<const double>(b.data(), k, n),
                                                 MutableMatrixView<double>(c.data(), m, n), opts);
    for (std::size_t i = m * n; i < c.size(); ++i) {
      ASSERT_DOUBLE_EQ(-1234.5, c[i]) << "write past C detected, n=" << n << " index=" << i;
    }
    EXPECT_TRUE(pcl_test::all_equal(expected, std::span<const double>(c.data(), m * n),
                                    pcl_test::Tolerance{1e-12, 1e-12}))
        << "n=" << n;
  }
}

//
// Defect 3b (2026-10-18, found by SimdMatmulDoesNotOverrunForNarrowColumnStrips
// and the matmul unit tests with a block size smaller than k): the ikj driver
// passed A's and B's *base* pointers to the micro-kernel for every k block
// instead of offsetting them to the block's first row, so every block re-added
// the contribution of k-slice 0.  This was invisible while block_size >= k
// (the default), because there is only one block.

// B is a full matrix whose leading dimension is its column count, but the
// kernel also has to work on a sub-matrix (ld > ncols) without touching the
// padding columns.
// ---------------------------------------------------------------------------
TEST(Regression, MatmulSimdStaysInsideMatrixBoundsForPaddedStorage) {
  const std::size_t m = 8, ld = 12, ncols = 5, k = 7;
  const auto a = pcl_test::make_vector<double>(m * ld, 331);
  const auto b = pcl_test::make_vector<double>(k * ld, 332);
  std::vector<double> c(m * ld, -1.0);

  // Exercise the panel kernel directly with ldb > ncols.
  for (std::size_t r = 0; r + 4 <= m; r += 4) {
    parallel_compute::simd::kernels::gemm_panel4_f64(c.data() + r * ld, ld, a.data() + r * ld, ld,
                                                     b.data(), ld, k, ncols);
  }
  for (std::size_t r = 0; r + 4 <= m; r += 4) {
    for (std::size_t j = 0; j < ncols; ++j) {
      // C starts at -1.0 and the micro-kernel accumulates into it, so the
      // oracle must start from the same value.
      double acc = -1.0;
      for (std::size_t p = 0; p < k; ++p) acc += a[(r + 0) * ld + p] * b[p * ld + j];
      EXPECT_DOUBLE_EQ(acc, c[r * ld + j]) << "r=" << r << " j=" << j;
    }
    // The padding columns beyond ncols must still be untouched.
    for (std::size_t j = ncols; j < ld; ++j) {
      EXPECT_DOUBLE_EQ(-1.0, c[r * ld + j]) << "r=" << r << " j=" << j;
    }
  }
}

// ---------------------------------------------------------------------------
// Defect 4 (2026-10-18): partitioning m rows over more teams than rows.
//
// matmul_parallel computed `rows_per = ceil(m / teams)`.  When m < teams this
// is 1 and the extra teams must be clamped to an empty range rather than
// computing begin/end from an uninitialised slice.
// ---------------------------------------------------------------------------
TEST(Regression, ParallelMatmulWithFewerRowsThanThreads) {
  for (std::size_t m = 1; m <= 3; ++m) {
    const std::size_t n = 6, k = 4;
    const auto a = pcl_test::make_vector<double>(m * k, 341);
    const auto b = pcl_test::make_vector<double>(k * n, 342);
    const auto expected = pcl_test::oracle_matmul(a, m, k, b, n);
    std::vector<double> got(m * n, 0.0);
    parallel_compute::gemm::GemmOptions opts;
    opts.block_size = 2;
    parallel_compute::gemm::matmul_parallel<double>(MatrixView<const double>(a.data(), m, k),
                                                     MatrixView<const double>(b.data(), k, n),
                                                     MutableMatrixView<double>(got.data(), m, n), opts);
    EXPECT_TRUE(pcl_test::all_equal(expected, got, pcl_test::Tolerance{1e-12, 1e-12}))
        << "m=" << m;
  }
}

TEST(Regression, ParallelForChunksWithFewerItemsThanTeams) {
  for (std::size_t n = 0; n <= 3; ++n) {
    std::vector<int> covered(n, 0);
    parallel_compute::parallel::ParallelConfig cfg;
    cfg.threads = 8;
    cfg.schedule = parallel_compute::parallel::Schedule::omp_static;
    parallel_compute::parallel::parallel_for_chunks(n, cfg, [&](std::size_t begin, std::size_t count) {
      ASSERT_LE(begin + count, n) << "n=" << n;
      for (std::size_t i = begin; i < begin + count; ++i) covered[i] += 1;
    });
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(1, covered[i]) << "n=" << n << " i=" << i;
  }
}

// ---------------------------------------------------------------------------
// Defect 4b (2026-10-18): memory::strided_sum treated its span as the list of
// elements to visit, so `strided_sum(buf_of_100, 16)` read up to index 1584 --
// past the end of the buffer.  It happened to stay in bounds in the benchmark
// only because that buffer was large; the unit test with n = 100 and stride 16
// was reading unallocated memory.  The contract is now "the span is the buffer;
// the number of elements visited is ceil(size / stride)", which is in bounds by
// construction.
// ---------------------------------------------------------------------------
TEST(Regression, StridedSumNeverReadsPastTheBuffer) {
  for (std::size_t n = 1; n <= 300; ++n) {
    // A heap buffer whose size is exactly n: AddressSanitizer turns any read
    // beyond the last element into a hard failure.
    auto buffer = std::make_unique<double[]>(n);
    parallel_compute::random::fill_uniform<double>(std::span<double>(buffer.get(), n),
                                                   0x5EEDULL, -1.0, 1.0);
    for (std::size_t stride : {1U, 2U, 3U, 7U, 16U, 64U, 128U}) {
      const double got = parallel_compute::memory::strided_sum<double>(
          std::span<const double>(buffer.get(), n), stride, Backend::reference);
      ASSERT_FALSE(std::isnan(got)) << "n=" << n << " stride=" << stride;
      EXPECT_LE(parallel_compute::memory::strided_element_count(n, stride) * stride, n + stride)
          << "n=" << n << " stride=" << stride;
    }
  }
}

}  // namespace
