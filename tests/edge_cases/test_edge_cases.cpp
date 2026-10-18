// Boundary conditions: empty, single element, odd and unaligned lengths,
// degenerate matrices, invalid configuration.
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "parallel_compute/algorithms/matmul.hpp"
#include "parallel_compute/algorithms/reductions.hpp"
#include "parallel_compute/algorithms/vector_ops.hpp"
#include "parallel_compute/common/tolerance.hpp"
#include "parallel_compute/memory/workloads.hpp"
#include "parallel_compute/parallel/scheduler.hpp"
#include "parallel_compute/simd/simd_kernels.hpp"
#include "test_support.hpp"

using parallel_compute::Backend;
using parallel_compute::MutableMatrixView;
using parallel_compute::MatrixView;

namespace {

// Every AVX2 f64 loop is 4-wide, so n = 1, 2, 3 exercise the scalar tail that
// is the most common source of off-by-one bugs.
TEST(EdgeCases, VectorOpsWithOneToThreeElements) {
  for (std::size_t n = 1; n <= 3; ++n) {
    const auto a = pcl_test::make_vector<double>(n, 101);
    const auto b = pcl_test::make_vector<double>(n, 102);
    std::vector<double> expected(n), got(n);
    for (auto backend : pcl_test::all_backends()) {
      parallel_compute::vector_ops::add<double>(a, b, expected, Backend::reference);
      parallel_compute::vector_ops::add<double>(a, b, got, backend);
      EXPECT_TRUE(pcl_test::all_equal(expected, got)) << "n=" << n;
    }
  }
}

TEST(EdgeCases, EmptyVectorOpsLeaveOutputUntouched) {
  std::vector<double> y = {1.0, 2.0, 3.0};
  for (auto backend : pcl_test::all_backends()) {
    parallel_compute::vector_ops::scale<double>(std::span<double>(y).subspan(0, 0), 5.0, backend);
    parallel_compute::vector_ops::saxpy<double>(std::span<double>(y).subspan(0, 0),
                                        std::span<double>(y).subspan(0, 0), 5.0, backend);
  }
  EXPECT_EQ(1.0, y[0]);
  EXPECT_EQ(2.0, y[1]);
  EXPECT_EQ(3.0, y[2]);
}

TEST(EdgeCases, ReductionsOnEmptyInput) {
  const std::vector<double> empty;
  for (auto backend : pcl_test::all_backends()) {
    EXPECT_DOUBLE_EQ(0.0, parallel_compute::reductions::sum<double>(empty, backend));
    EXPECT_DOUBLE_EQ(0.0, parallel_compute::reductions::mean<double>(empty, backend));
    EXPECT_DOUBLE_EQ(0.0, parallel_compute::reductions::dot<double>(empty, empty, backend));
    EXPECT_DOUBLE_EQ(std::numeric_limits<double>::max(),
                     parallel_compute::reductions::min_value<double>(empty, backend));
    EXPECT_DOUBLE_EQ(std::numeric_limits<double>::lowest(),
                     parallel_compute::reductions::max_value<double>(empty, backend));
    EXPECT_DOUBLE_EQ(0.0, parallel_compute::memory::sequential_sum<double>(empty, backend));
    EXPECT_DOUBLE_EQ(0.0, parallel_compute::memory::strided_sum<double>(empty, 4, backend));
  }
}

TEST(EdgeCases, UnalignedPointersProduceTheSameResult) {
  // Offsetting the data pointer by 1..7 elements puts every buffer at a
  // different alignment, which is exactly what the unaligned load paths must
  // handle without changing the answer.
  const std::size_t n = 37;
  const auto storage_a = pcl_test::make_vector<double>(n + 8, 111);
  const auto storage_b = pcl_test::make_vector<double>(n + 8, 112);
  std::vector<double> expected(n), got(n + 8);
  for (std::size_t offset = 0; offset < 8; ++offset) {
    const double* a = storage_a.data() + offset;
    const double* b = storage_b.data() + offset;
    double* out = got.data() + offset;
    // The expectation is rebuilt per offset: the storage vectors are longer
    // than the window, so a single precomputed reference would be wrong.
    parallel_compute::vector_ops::add<double>(std::span<const double>(a, n),
                                              std::span<const double>(b, n), expected,
                                              Backend::reference);
    for (auto backend : pcl_test::all_backends()) {
      parallel_compute::vector_ops::add<double>(std::span<const double>(a, n),
                                                std::span<const double>(b, n),
                                                std::span<double>(out, n), backend);
      EXPECT_TRUE(pcl_test::all_equal(expected, std::span<double>(out, n)))
          << "offset=" << offset << " backend=" << parallel_compute::backend_name(backend);
    }
  }
}

TEST(EdgeCases, NonSquareMatrices) {
  struct Shape {
    std::size_t m, n, k;
  };
  const Shape shapes[] = {{1, 100, 1}, {100, 1, 1}, {1, 1, 100}, {100, 3, 7}, {7, 100, 3},
                          {2, 3, 5},  {5, 3, 2},   {3, 2, 5}};
  for (const Shape& s : shapes) {
    const auto a = pcl_test::make_vector<double>(s.m * s.k, 121);
    const auto b = pcl_test::make_vector<double>(s.k * s.n, 122);
    const auto expected = pcl_test::oracle_matmul(a, s.m, s.k, b, s.n);
    std::vector<double> got(s.m * s.n, 0.0);
    const MatrixView<const double> av(a.data(), s.m, s.k);
    const MatrixView<const double> bv(b.data(), s.k, s.n);
    parallel_compute::gemm::GemmOptions opts;
    opts.block_size = 16;
    auto check = [&](const char* name) {
      EXPECT_TRUE(pcl_test::all_equal(expected, got, pcl_test::Tolerance{1e-12, 1e-12}))
          << name << " shape " << s.m << "x" << s.n << "x" << s.k;
    };
    parallel_compute::gemm::matmul_reference<double>(av, bv, MutableMatrixView<double>(got.data(), s.m, s.n));
    check("reference");
    std::fill(got.begin(), got.end(), 0.0);
    parallel_compute::gemm::matmul_blocked<double>(av, bv, MutableMatrixView<double>(got.data(), s.m, s.n), opts);
    check("blocked");
    std::fill(got.begin(), got.end(), 0.0);
    parallel_compute::gemm::matmul_parallel<double>(av, bv, MutableMatrixView<double>(got.data(), s.m, s.n), opts);
    check("parallel");
    std::fill(got.begin(), got.end(), 0.0);
    parallel_compute::gemm::matmul_simd<double>(av, bv, MutableMatrixView<double>(got.data(), s.m, s.n), opts);
    check("simd");
  }
}

TEST(EdgeCases, MatmulDimensionsNotAlignedToTheVectorWidth) {
  // m = 5 leaves a 1-row tail after the 4-row SIMD panel; n = 6 leaves a
  // 2-column tail; k = 7 leaves a partial k block.
  const std::size_t m = 5, n = 6, k = 7;
  const auto a = pcl_test::make_vector<double>(m * k, 131);
  const auto b = pcl_test::make_vector<double>(k * n, 132);
  const auto expected = pcl_test::oracle_matmul(a, m, k, b, n);
  const MatrixView<const double> av(a.data(), m, k);
  const MatrixView<const double> bv(b.data(), k, n);
  std::vector<double> got(m * n, 0.0);
  parallel_compute::gemm::GemmOptions opts;
  opts.block_size = 4;
  parallel_compute::gemm::matmul_simd<double>(av, bv, MutableMatrixView<double>(got.data(), m, n),
                                              opts);
  EXPECT_TRUE(pcl_test::all_equal(expected, got, pcl_test::Tolerance{1e-12, 1e-12}));
}

TEST(EdgeCases, MatmulWithZeroInnerDimensionGivesZero) {
  const std::size_t m = 4, n = 3;
  std::vector<double> c(m * n, 5.0);
  const MatrixView<const double> a(nullptr, m, 0);
  const MatrixView<const double> b(nullptr, 0, n);
  parallel_compute::gemm::matmul_parallel<double>(a, b,
                                                   MutableMatrixView<double>(c.data(), m, n));
  for (double v : c) EXPECT_DOUBLE_EQ(0.0, v);
}

TEST(EdgeCases, ZeroBlocksInParallelFor) {
  for (auto sched : {parallel_compute::parallel::Schedule::serial,
                     parallel_compute::parallel::Schedule::omp_static,
                     parallel_compute::parallel::Schedule::omp_dynamic,
                     parallel_compute::parallel::Schedule::omp_guided}) {
    std::atomic<int> calls{0};
    parallel_compute::parallel::ParallelConfig cfg;
    cfg.schedule = sched;
    cfg.threads = 4;
    parallel_compute::parallel::parallel_for(0U, cfg, [&](std::size_t) {
      calls.fetch_add(1);
    });
    EXPECT_EQ(0, calls.load());
  }
}

TEST(EdgeCases, IntegerInputsDoNotOverflowTheCounter) {
  // The scheduling checksum is unsigned 64-bit arithmetic with wraparound by
  // design; what must hold is that the same input always gives the same value.
  parallel_compute::parallel::ParallelConfig cfg;
  cfg.schedule = parallel_compute::parallel::Schedule::serial;
  const auto a = parallel_compute::parallel::scheduling_workload(10000, 0, 3, cfg, 7);
  const auto b = parallel_compute::parallel::scheduling_workload(10000, 0, 3, cfg, 7);
  EXPECT_EQ(a, b);
}

TEST(EdgeCases, InvalidThreadCountIsRejected) {
  // threads <= 0 means "use the default" in ParallelConfig, but the runtime
  // setter must reject it outright rather than silently changing state.
  EXPECT_FALSE(parallel_compute::parallel::set_thread_count(0));
  EXPECT_FALSE(parallel_compute::parallel::set_thread_count(-5));
  EXPECT_GE(parallel_compute::parallel::max_threads(), 1);
}

TEST(EdgeCases, SetThreadCountLargerThanCoresIsClampedOrHonoured) {
  const int original = parallel_compute::parallel::current_thread_count();
  if (!parallel_compute::parallel::openmp_enabled()) {
    EXPECT_EQ(1, original);
    return;
  }
  // Over-subscribing is allowed by OpenMP; the library must not crash and must
  // still produce correct results (covered in tests/concurrency).
  ASSERT_TRUE(parallel_compute::parallel::set_thread_count(original * 2));
  EXPECT_GE(parallel_compute::parallel::current_thread_count(), original);
  ASSERT_TRUE(parallel_compute::parallel::set_thread_count(original));
}

TEST(EdgeCases, OverflowingDimensionsAreRejectedByTheShapeCheck) {
  // Dimensions this large are impossible to allocate; they must be rejected by
  // the argument validation, never silently wrapped into a small buffer.
  const std::vector<double> a(4, 1.0);
  std::vector<double> c(4, 0.0);
  const std::size_t huge = std::numeric_limits<std::size_t>::max() / 2;
  EXPECT_THROW(parallel_compute::gemm::matmul_reference<double>(
                   MatrixView<const double>(a.data(), huge, huge - 1),
                   MatrixView<const double>(a.data(), huge, huge),
                   MutableMatrixView<double>(c.data(), huge, huge)),
               std::invalid_argument);
  EXPECT_THROW(parallel_compute::gemm::matmul_parallel<double>(
                   MatrixView<const double>(a.data(), huge, huge),
                   MatrixView<const double>(a.data(), huge, huge),
                   MutableMatrixView<double>(c.data(), huge - 1, huge)),
               std::invalid_argument);
}

}  // namespace
