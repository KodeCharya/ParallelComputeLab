// Numerical accuracy tests.
//
// These do not re-check contracts (see tests/unit); they check how far each
// optimized variant drifts from the scalar oracle over large inputs, and that
// the drift stays inside a documented budget.  A variant that is fast *and*
// wrong shows up here.
#include <cmath>
#include <cstddef>
#include <string>
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

namespace {

// Accumulated relative error budget for a length-n double summation in a
// different order is ~ n * eps / (sum of |x_i|).  For the uniformly random
// inputs used here (mean |x| = 0.5) the observed error is far below this;
// the constant below is deliberately generous so the test is not flaky.
constexpr double kReductionRelTol = 1e-12;
constexpr double kElementwiseRelTol = 0.0;  // element-wise ops are order independent

std::string backend_label(Backend b) { return parallel_compute::backend_name(b); }

TEST(NumericalElementWise, AllBackendsAreBitExactForElementWiseOps) {
  // Addition, multiplication and scaling have no reduction, so any difference
  // from the reference would be a real defect, not a rounding artefact.
  const std::size_t n = 1 << 16;
  const auto a = pcl_test::make_vector<double>(n, 51);
  const auto b = pcl_test::make_vector<double>(n, 52);
  std::vector<double> expected(n), got(n);
  for (auto backend : pcl_test::all_backends()) {
    parallel_compute::vector_ops::add<double>(a, b, expected, Backend::reference);
    parallel_compute::vector_ops::add<double>(a, b, got, backend);
    EXPECT_TRUE(pcl_test::all_equal(expected, got)) << backend_label(backend);

    parallel_compute::vector_ops::multiply<double>(a, b, expected, Backend::reference);
    parallel_compute::vector_ops::multiply<double>(a, b, got, backend);
    EXPECT_TRUE(pcl_test::all_equal(expected, got)) << backend_label(backend);

    parallel_compute::vector_ops::saxpy<double>(expected, b, 0.37, Backend::reference);
    parallel_compute::vector_ops::saxpy<double>(got, b, 0.37, backend);
    EXPECT_TRUE(pcl_test::all_equal(expected, got)) << backend_label(backend);
  }
  EXPECT_DOUBLE_EQ(0.0, kElementwiseRelTol);
}

TEST(NumericalReductions, SumErrorStaysWithinBudget) {
  const std::size_t n = 1 << 18;
  const auto a = pcl_test::make_vector<double>(n, 53);
  const double reference = parallel_compute::reductions::sum<double>(a, Backend::reference);
  ASSERT_GT(std::fabs(reference), 0.0);
  for (auto backend : pcl_test::all_backends()) {
    const double got = parallel_compute::reductions::sum<double>(a, backend);
    const double rel = parallel_compute::relative_error(reference, got);
    EXPECT_LT(rel, kReductionRelTol)
        << backend_label(backend) << " got=" << got << " ref=" << reference;
  }
}

TEST(NumericalReductions, DotErrorStaysWithinBudget) {
  const std::size_t n = 1 << 18;
  const auto a = pcl_test::make_vector<double>(n, 54);
  const auto b = pcl_test::make_vector<double>(n, 55);
  const double reference = parallel_compute::reductions::dot<double>(a, b, Backend::reference);
  ASSERT_GT(std::fabs(reference), 0.0);
  for (auto backend : pcl_test::all_backends()) {
    const double got = parallel_compute::reductions::dot<double>(a, b, backend);
    EXPECT_LT(parallel_compute::relative_error(reference, got), kReductionRelTol)
        << backend_label(backend);
  }
}

TEST(NumericalReductions, ParallelReductionIsStableAcrossThreadCounts) {
  const std::size_t n = 1 << 20;
  const auto a = pcl_test::make_vector<double>(n, 56);
  const double reference = parallel_compute::reductions::sum<double>(a, Backend::reference);
  const int original = parallel_compute::parallel::current_thread_count();
  for (int threads : {1, 2, 3, 4, 8}) {
    parallel_compute::parallel::set_thread_count(threads);
    const double got = parallel_compute::reductions::sum<double>(a, Backend::openmp);
    EXPECT_LT(parallel_compute::relative_error(reference, got), kReductionRelTol)
        << "threads=" << threads;
  }
  parallel_compute::parallel::set_thread_count(original);
}

TEST(NumericalReductions, CatastrophicCancellationIsNotHidden) {
  // The parallel sum cannot recover information the reference sum also loses.
  // This documents the honest limit of a tree reduction rather than pretending
  // reordering is free.
  std::vector<double> a(1 << 16);
  for (std::size_t i = 0; i < a.size(); ++i) a[i] = 1.0;
  a[0] = 1.0;
  a[a.size() / 2] = -1.0;
  const double reference = parallel_compute::reductions::sum<double>(a, Backend::reference);
  for (auto backend : pcl_test::all_backends()) {
    const double got = parallel_compute::reductions::sum<double>(a, backend);
    EXPECT_NEAR(reference, got, 1e-9) << backend_label(backend);
  }
}

TEST(NumericalMatmul, VariantsAgreeWithOracleWithinTolerance) {
  const std::size_t m = 96, n = 80, k = 112;
  const auto a = pcl_test::make_vector<double>(m * k, 61);
  const auto b = pcl_test::make_vector<double>(k * n, 62);
  const auto expected = pcl_test::oracle_matmul(a, m, k, b, n);

  const MatrixView<const double> av(a.data(), m, k);
  const MatrixView<const double> bv(b.data(), k, n);
  parallel_compute::gemm::GemmOptions opts;

  std::vector<double> got(m * n, 0.0);
  // Worst case is reported as |diff| / (|ref| + 1) rather than a pure relative
  // error: a GEMM entry that happens to cancel to ~0 would otherwise report an
  // enormous "relative" error that says nothing about accuracy.
  auto run = [&](const char* name, auto&& fn) {
    std::fill(got.begin(), got.end(), 0.0);
    fn(MutableMatrixView<double>(got.data(), m, n));
    double worst = 0.0;
    std::size_t worst_i = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
      const double denom = std::fabs(expected[i]) + 1.0;
      const double err = std::fabs(expected[i] - got[i]) / denom;
      if (err > worst) {
        worst = err;
        worst_i = i;
      }
    }
    // k = 112 sequential accumulations of O(1) products -> ~1e-14 drift.
    EXPECT_LT(worst, 1e-12) << "variant=" << name << " at index " << worst_i
                            << " ref=" << expected[worst_i] << " got=" << got[worst_i];
  };

  run("reference", [&](auto c) {
    parallel_compute::gemm::matmul_reference<double>(av, bv, c);
  });
  run("auto_vectorized", [&](auto c) {
    parallel_compute::gemm::matmul_auto_vectorized<double>(av, bv, c);
  });
  run("blocked", [&](auto c) { parallel_compute::gemm::matmul_blocked<double>(av, bv, c, opts); });
  run("parallel", [&](auto c) { parallel_compute::gemm::matmul_parallel<double>(av, bv, c, opts); });
  run("simd", [&](auto c) { parallel_compute::gemm::matmul_simd<double>(av, bv, c, opts); });
  if (parallel_compute::gemm::blas_available()) {
    run("blas", [&](auto c) { parallel_compute::gemm::matmul_blas<double>(av, bv, c); });
  }
}

TEST(NumericalMatmul, BlockSizeDoesNotChangeTheResultBeyondTolerance) {
  const std::size_t m = 64, n = 64, k = 64;
  const auto a = pcl_test::make_vector<double>(m * k, 71);
  const auto b = pcl_test::make_vector<double>(k * n, 72);
  const MatrixView<const double> av(a.data(), m, k);
  const MatrixView<const double> bv(b.data(), k, n);
  std::vector<double> base(m * n, 0.0), other(m * n, 0.0);
  parallel_compute::gemm::GemmOptions o1;
  o1.block_size = 8;
  parallel_compute::gemm::matmul_blocked<double>(av, bv, MutableMatrixView<double>(base.data(), m, n),
                                                 o1);
  for (std::size_t bs : {1U, 2U, 16U, 64U, 128U}) {
    std::fill(other.begin(), other.end(), 0.0);
    parallel_compute::gemm::GemmOptions o2;
    o2.block_size = bs;
    parallel_compute::gemm::matmul_blocked<double>(av, bv, MutableMatrixView<double>(other.data(), m, n),
                                                   o2);
    EXPECT_TRUE(pcl_test::all_equal(base, other, pcl_test::Tolerance{1e-12, 1e-12}))
        << "block_size=" << bs;
  }
}

TEST(NumericalMatmul, FloatMatmulWithinSinglePrecisionBudget) {
  const std::size_t m = 48, n = 40, k = 56;
  const auto a = pcl_test::make_vector<float>(m * k, 81);
  const auto b = pcl_test::make_vector<float>(k * n, 82);
  const auto expected = pcl_test::oracle_matmul(a, m, k, b, n);
  const MatrixView<const float> av(a.data(), m, k);
  const MatrixView<const float> bv(b.data(), k, n);
  parallel_compute::gemm::GemmOptions opts;
  std::vector<float> got(m * n, 0.0f);

  auto worst_rel = [&](const char* name) {
    float worst = 0.0f;
    std::size_t worst_i = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
      const float denom = std::fabs(expected[i]) + 1.0f;
      const float err = std::fabs(expected[i] - got[i]) / denom;
      if (err > worst) {
        worst = err;
        worst_i = i;
      }
    }
    // float has ~1.2e-7 eps; k = 56 sequential accumulations -> ~1e-6.
    EXPECT_LT(worst, 1e-5f) << "variant=" << name << " at index " << worst_i
                             << " ref=" << expected[worst_i] << " got=" << got[worst_i];
  };

  parallel_compute::gemm::matmul_simd<float>(av, bv, MutableMatrixView<float>(got.data(), m, n),
                                             opts);
  worst_rel("simd");
  std::fill(got.begin(), got.end(), 0.0f);
  parallel_compute::gemm::matmul_blocked<float>(av, bv, MutableMatrixView<float>(got.data(), m, n),
                                                 opts);
  worst_rel("blocked");
  std::fill(got.begin(), got.end(), 0.0f);
  parallel_compute::gemm::matmul_reference<float>(av, bv, MutableMatrixView<float>(got.data(), m, n));
  worst_rel("reference");
}

TEST(NumericalMemory, TriadAndCopyAreBitExactAcrossBackends) {
  const std::size_t n = 1 << 16;
  const auto b = pcl_test::make_vector<double>(n, 91);
  const auto c = pcl_test::make_vector<double>(n, 92);
  std::vector<double> expected(n), got(n);
  parallel_compute::memory::triad<double>(expected, b, c, 1.75, Backend::reference);
  for (auto backend : pcl_test::all_backends()) {
    std::fill(got.begin(), got.end(), 0.0);
    parallel_compute::memory::triad<double>(got, b, c, 1.75, backend);
    EXPECT_TRUE(pcl_test::all_equal(expected, got))
        << backend_label(backend);
  }
}

TEST(NumericalMemory, StridedSumMatchesSequentialSumAtStrideOne) {
  const std::size_t n = 1 << 14;
  const auto a = pcl_test::make_vector<double>(n, 93);
  const double seq = parallel_compute::memory::sequential_sum<double>(a, Backend::reference);
  const double strided = parallel_compute::memory::strided_sum<double>(a, 1, Backend::reference);
  EXPECT_LT(parallel_compute::relative_error(seq, strided), 1e-12);
}

}  // namespace
