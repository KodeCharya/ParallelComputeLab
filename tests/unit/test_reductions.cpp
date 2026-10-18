// Contract tests for the reductions, including the documented empty-input,
// NaN/Inf and identity semantics.
#include <cmath>
#include <limits>
#include <vector>

#include "parallel_compute/algorithms/reductions.hpp"
#include "parallel_compute/common/tolerance.hpp"
#include "test_support.hpp"

using parallel_compute::Backend;
namespace red = parallel_compute::reductions;

namespace {

class ReductionTest : public ::testing::TestWithParam<Backend> {
 protected:
  Backend backend() const { return GetParam(); }
};

TEST_P(ReductionTest, SumMatchesReference) {
  for (std::size_t n : pcl_test::interesting_lengths()) {
    const auto a = pcl_test::make_vector<double>(n, 11);
    const double expected = red::sum<double>(a, Backend::reference);
    const double got = red::sum<double>(a, backend());
    // Different summation orders: tolerance, never exact equality.
    EXPECT_LT(parallel_compute::relative_error(expected, got), 1e-12)
        << "n=" << n << " backend=" << parallel_compute::backend_name(backend());
  }
}

TEST_P(ReductionTest, DotMatchesReference) {
  for (std::size_t n : pcl_test::interesting_lengths()) {
    const auto a = pcl_test::make_vector<double>(n, 12);
    const auto b = pcl_test::make_vector<double>(n, 13);
    const double expected = red::dot<double>(a, b, Backend::reference);
    const double got = red::dot<double>(a, b, backend());
    EXPECT_LT(parallel_compute::relative_error(expected, got), 1e-12)
        << "n=" << n << " backend=" << parallel_compute::backend_name(backend());
  }
}

TEST_P(ReductionTest, MeanMatchesReference) {
  for (std::size_t n : pcl_test::interesting_lengths()) {
    if (n == 0) continue;
    const auto a = pcl_test::make_vector<double>(n, 14);
    const double expected = red::mean<double>(a, Backend::reference);
    const double got = red::mean<double>(a, backend());
    EXPECT_LT(parallel_compute::relative_error(expected, got), 1e-12)
        << "n=" << n << " backend=" << parallel_compute::backend_name(backend());
  }
}

TEST_P(ReductionTest, MinMaxMatchReference) {
  for (std::size_t n : pcl_test::interesting_lengths()) {
    const auto a = pcl_test::make_vector<double>(n, 15);
    bool ref_found = false, got_found = false;
    const double emin = red::min_value<double>(a, Backend::reference, &ref_found);
    const double gmin = red::min_value<double>(a, backend(), &got_found);
    const double emax = red::max_value<double>(a, Backend::reference, &ref_found);
    const double gmax = red::max_value<double>(a, backend(), &got_found);
    EXPECT_EQ(n > 0, ref_found);
    EXPECT_EQ(ref_found, got_found);
    EXPECT_EQ(emin, gmin) << "n=" << n;
    EXPECT_EQ(emax, gmax) << "n=" << n;
  }
}

TEST_P(ReductionTest, EmptyInputContract) {
  const std::vector<double> empty;
  EXPECT_DOUBLE_EQ(0.0, red::sum<double>(empty, backend()));
  EXPECT_DOUBLE_EQ(0.0, red::mean<double>(empty, backend()));
  EXPECT_DOUBLE_EQ(0.0, red::dot<double>(empty, empty, backend()));

  bool found = true;
  EXPECT_EQ(std::numeric_limits<double>::max(),
            red::min_value<double>(empty, backend(), &found));
  EXPECT_FALSE(found);
  found = true;
  EXPECT_EQ(std::numeric_limits<double>::lowest(),
            red::max_value<double>(empty, backend(), &found));
  EXPECT_FALSE(found);
}

TEST_P(ReductionTest, NanPropagates) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  // NaN in the middle of the range.
  std::vector<double> a = {3.0, 1.0, nan, 7.0, 2.0};
  EXPECT_TRUE(std::isnan(red::sum<double>(a, backend())));
  EXPECT_TRUE(std::isnan(red::min_value<double>(a, backend())));
  EXPECT_TRUE(std::isnan(red::max_value<double>(a, backend())));

  // NaN as the very first element must win as well (see reductions.cpp).
  std::vector<double> first = {nan, 1.0, 2.0};
  EXPECT_TRUE(std::isnan(red::sum<double>(first, backend())));
  EXPECT_TRUE(std::isnan(red::min_value<double>(first, backend())));
  EXPECT_TRUE(std::isnan(red::max_value<double>(first, backend())));

  std::vector<double> inf_a = {inf, -inf};
  std::vector<double> inf_b = {1.0, 1.0};
  EXPECT_TRUE(std::isnan(red::sum<double>(inf_a, backend()))) << "inf + (-inf)";
  EXPECT_TRUE(std::isinf(red::min_value<double>(inf_a, backend()))) << "min(inf, -inf)";
  // dot([inf, -inf], [1, 1]) = inf - inf = NaN, not +inf.
  EXPECT_TRUE(std::isnan(red::dot<double>(inf_a, inf_b, backend())));

  // A dot product with a single infinite product and a finite sum is +inf.
  std::vector<double> pos_a = {inf, 2.0};
  std::vector<double> pos_b = {1.0, 1.0};
  EXPECT_DOUBLE_EQ(std::numeric_limits<double>::infinity(),
                   red::dot<double>(pos_a, pos_b, backend()));
}

TEST(ReductionArgumentValidation, DotLengthMismatchThrows) {
  const std::vector<double> a(8, 1.0);
  const std::vector<double> b(7, 1.0);
  EXPECT_THROW(static_cast<void>(red::dot<double>(a, b, Backend::reference)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(red::dot<double>(a, b, Backend::openmp)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(red::dot<double>(a, b, Backend::simd)),
               std::invalid_argument);
}

TEST(ReductionSingleElement, SumIsIdentity) {
  for (auto backend : pcl_test::all_backends()) {
    const std::vector<float> a = {42.0f};
    EXPECT_FLOAT_EQ(42.0f, red::sum<float>(a, backend));
    bool found = false;
    EXPECT_FLOAT_EQ(42.0f, red::min_value<float>(a, backend, &found));
    EXPECT_TRUE(found);
    EXPECT_FLOAT_EQ(42.0f, red::max_value<float>(a, backend, &found));
  }
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(AllBackends, ReductionTest, ::testing::ValuesIn(pcl_test::all_backends()),
                         [](const ::testing::TestParamInfo<Backend>& p) {
                           return std::string(parallel_compute::backend_name(p.param));
                         });
