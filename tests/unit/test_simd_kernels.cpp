// Tests for the runtime capability detection and the raw vector kernels.
#include <cstddef>
#include <string>
#include <vector>

#include "parallel_compute/common/tolerance.hpp"
#include "parallel_compute/simd/cpu_features.hpp"
#include "parallel_compute/simd/simd_kernels.hpp"
#include "test_support.hpp"

namespace simd = parallel_compute::simd;
namespace kernels = parallel_compute::simd::kernels;

namespace {

TEST(CpuFeatures, DetectionIsConsistent) {
  const auto& f = simd::cpu_features();
  // AVX2 without FMA is not enough for this project's AVX2 kernels; if the CPU
  // reports AVX2 it must also report FMA, otherwise the kernels would have to
  // be split.
  if (f.avx2) {
    EXPECT_TRUE(f.fma);
  }
  // SSE2 is part of the x86-64 baseline.
#if defined(__x86_64__) || defined(__i386__)
  EXPECT_TRUE(f.sse2);
#endif
#if defined(__aarch64__)
  EXPECT_TRUE(f.neon);
#endif
  EXPECT_EQ(f.avx2 && f.fma, simd::avx2_active());
}

TEST(CpuFeatures, BestBackendIsAmongTheSupportedSet) {
  const std::string best = simd::best_simd_backend();
  const std::string supported = simd::describe_supported_simd();
  if (best == "none") {
    EXPECT_EQ("none", supported);
  } else {
    EXPECT_NE(std::string::npos, supported.find(best)) << "best=" << best;
  }
  EXPECT_GE(simd::simd_lane_width(), 1U);
}

TEST(CpuFeatures, RepeatedQueriesAreStable) {
  EXPECT_EQ(simd::best_simd_backend(), simd::best_simd_backend());
  EXPECT_EQ(simd::describe_supported_simd(), simd::describe_supported_simd());
}

TEST(SimdKernels, ActiveNameIsReported) {
  const std::string name = kernels::active_name();
  if (kernels::available()) {
    EXPECT_NE("scalar-fallback", name);
    EXPECT_GE(kernels::lane_width(), 2U);
  } else {
    EXPECT_EQ("scalar-fallback", name);
    EXPECT_EQ(1U, kernels::lane_width());
  }
}

TEST(SimdKernels, SaxpyMatchesScalar) {
  for (std::size_t n : pcl_test::interesting_lengths()) {
    auto y = pcl_test::make_vector<double>(n, 31);
    const auto x = pcl_test::make_vector<double>(n, 32);
    auto expected = y;
    for (std::size_t i = 0; i < n; ++i) expected[i] += 2.5 * x[i];
    kernels::saxpy(y.data(), x.data(), n, 2.5);
    EXPECT_TRUE(pcl_test::all_equal(expected, y))
        << "n=" << n << " isa=" << kernels::active_name();
  }
}

TEST(SimdKernels, SaxpyFloatMatchesScalar) {
  for (std::size_t n : pcl_test::interesting_lengths()) {
    auto y = pcl_test::make_vector<float>(n, 33);
    const auto x = pcl_test::make_vector<float>(n, 34);
    auto expected = y;
    for (std::size_t i = 0; i < n; ++i) expected[i] += -0.75f * x[i];
    kernels::saxpy(y.data(), x.data(), n, -0.75f);
    // Not bit-exact: the AVX2 kernel contracts `y + alpha*x` into a single FMA
    // while the oracle here may or may not, depending on how the compiler
    // contracts the test's own loop.  A few ULP is the honest expectation.
    EXPECT_TRUE(pcl_test::all_equal(expected, y, pcl_test::Tolerance{1e-6, 1e-6}))
        << "n=" << n;
  }
}

TEST(SimdKernels, AddAndMultiplyMatchScalar) {
  for (std::size_t n : pcl_test::interesting_lengths()) {
    const auto a = pcl_test::make_vector<double>(n, 35);
    const auto b = pcl_test::make_vector<double>(n, 36);
    std::vector<double> out(n), exp_add(n), exp_mul(n);
    for (std::size_t i = 0; i < n; ++i) {
      exp_add[i] = a[i] + b[i];
      exp_mul[i] = a[i] * b[i];
    }
    kernels::add(out.data(), a.data(), b.data(), n);
    EXPECT_TRUE(pcl_test::all_equal(exp_add, out)) << "n=" << n;
    kernels::multiply(out.data(), a.data(), b.data(), n);
    EXPECT_TRUE(pcl_test::all_equal(exp_mul, out)) << "n=" << n;
  }
}

TEST(SimdKernels, ScaleMatchesScalar) {
  for (std::size_t n : pcl_test::interesting_lengths()) {
    auto y = pcl_test::make_vector<float>(n, 37);
    auto expected = y;
    for (std::size_t i = 0; i < n; ++i) expected[i] *= 3.0f;
    kernels::scale(y.data(), n, 3.0f);
    EXPECT_TRUE(pcl_test::all_equal(expected, y)) << "n=" << n;
  }
}

TEST(SimdKernels, SumAndDotMatchScalarWithinTolerance) {
  for (std::size_t n : pcl_test::interesting_lengths()) {
    const auto a = pcl_test::make_vector<double>(n, 38);
    const auto b = pcl_test::make_vector<double>(n, 39);
    double exp_sum = 0.0, exp_dot = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      exp_sum += a[i];
      exp_dot += a[i] * b[i];
    }
    EXPECT_LT(parallel_compute::relative_error(exp_sum, kernels::sum(a.data(), n)), 1e-12)
        << "n=" << n;
    EXPECT_LT(parallel_compute::relative_error(exp_dot, kernels::dot(a.data(), b.data(), n)),
              1e-12)
        << "n=" << n;
  }
}

TEST(SimdKernels, SumPropagatesNan) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> a = {1.0, nan, 2.0};
  EXPECT_TRUE(std::isnan(kernels::sum(a.data(), a.size())));
}

TEST(SimdKernels, GemmPanelMatchesScalar) {
  // ncols must be >= 2 for f64 and >= 4 for f32; check both the vector path and
  // the scalar tail columns.
  for (std::size_t ncols : {2U, 3U, 4U, 5U, 8U, 9U, 16U, 17U}) {
    for (std::size_t kk : {1U, 2U, 7U, 16U}) {
      const auto a = pcl_test::make_vector<double>(4 * kk, 41);
      const auto b = pcl_test::make_vector<double>(kk * ncols, 42);
      std::vector<double> c(4 * ncols, 0.0);
      kernels::gemm_panel4_f64(c.data(), ncols, a.data(), kk, b.data(), ncols, kk, ncols);
      for (std::size_t r = 0; r < 4; ++r) {
        for (std::size_t j = 0; j < ncols; ++j) {
          double acc = 0.0;
          for (std::size_t p = 0; p < kk; ++p) acc += a[r * kk + p] * b[p * ncols + j];
          EXPECT_TRUE(parallel_compute::approx_equal<double>(acc, c[r * ncols + j],
                                                             pcl_test::Tolerance{1e-13, 1e-13}))
              << "r=" << r << " j=" << j << " kk=" << kk << " ncols=" << ncols;
        }
      }
    }
  }
}

TEST(SimdKernels, GemmPanelRespectsLeadingDimension) {
  // A sub-matrix with a larger leading dimension than the used width.
  const std::size_t kk = 3, ld = 8;
  const auto a = pcl_test::make_vector<double>(4 * ld, 43);
  const auto b = pcl_test::make_vector<double>(kk * ld, 44);
  std::vector<double> c(4 * ld, -1.0);
  const std::size_t ncols = 5;
  kernels::gemm_panel4_f64(c.data(), ld, a.data(), ld, b.data(), ld, kk, ncols);
  for (std::size_t r = 0; r < 4; ++r) {
    for (std::size_t j = 0; j < ncols; ++j) {
      double acc = -1.0;  // C starts at -1.0 and the kernel accumulates
      for (std::size_t p = 0; p < kk; ++p) acc += a[r * ld + p] * b[p * ld + j];
      EXPECT_TRUE(parallel_compute::approx_equal<double>(acc, c[r * ld + j],
                                                         pcl_test::Tolerance{1e-13, 1e-13}))
          << "r=" << r << " j=" << j;
    }
    // Columns beyond ncols must be untouched.
    for (std::size_t j = ncols; j < ld; ++j) EXPECT_DOUBLE_EQ(-1.0, c[r * ld + j]);
  }
}

TEST(SimdKernels, GemmPanelFloatMatchesScalar) {
  for (std::size_t ncols : {4U, 5U, 8U, 11U}) {
    for (std::size_t kk : {1U, 3U, 8U}) {
      const auto a = pcl_test::make_vector<float>(4 * kk, 45);
      const auto b = pcl_test::make_vector<float>(kk * ncols, 46);
      std::vector<float> c(4 * ncols, 0.0f);
      kernels::gemm_panel4_f32(c.data(), ncols, a.data(), kk, b.data(), ncols, kk, ncols);
      for (std::size_t r = 0; r < 4; ++r) {
        for (std::size_t j = 0; j < ncols; ++j) {
          float acc = 0.0f;
          for (std::size_t p = 0; p < kk; ++p) acc += a[r * kk + p] * b[p * ncols + j];
          EXPECT_TRUE(parallel_compute::approx_equal<float>(acc, c[r * ncols + j],
                                                            pcl_test::Tolerance{1e-5, 1e-5}))
              << "r=" << r << " j=" << j << " kk=" << kk << " ncols=" << ncols;
        }
      }
    }
  }
}

}  // namespace
