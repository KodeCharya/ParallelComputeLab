// Contract tests for the element-wise kernels.
#include <span>
#include <string>
#include <vector>

#include "parallel_compute/algorithms/vector_ops.hpp"
#include "parallel_compute/common/tolerance.hpp"
#include "test_support.hpp"

using parallel_compute::Backend;
namespace vops = parallel_compute::vector_ops;

namespace {

class VectorOpsTest : public ::testing::TestWithParam<Backend> {
 protected:
  Backend backend() const { return GetParam(); }
};

TEST_P(VectorOpsTest, AddMatchesReference) {
  for (std::size_t n : pcl_test::interesting_lengths()) {
    const auto a = pcl_test::make_vector<double>(n, 1);
    const auto b = pcl_test::make_vector<double>(n, 2);
    std::vector<double> expected(n), got(n);
    vops::add<double>(a, b, expected, Backend::reference);
    vops::add<double>(a, b, got, backend());
    EXPECT_TRUE(pcl_test::all_equal(expected, got))
        << "n=" << n << " backend=" << parallel_compute::backend_name(backend());
  }
}

TEST_P(VectorOpsTest, MultiplyMatchesReference) {
  for (std::size_t n : pcl_test::interesting_lengths()) {
    const auto a = pcl_test::make_vector<double>(n, 3);
    const auto b = pcl_test::make_vector<double>(n, 4);
    std::vector<double> expected(n), got(n);
    vops::multiply<double>(a, b, expected, Backend::reference);
    vops::multiply<double>(a, b, got, backend());
    EXPECT_TRUE(pcl_test::all_equal(expected, got))
        << "n=" << n << " backend=" << parallel_compute::backend_name(backend());
  }
}

TEST_P(VectorOpsTest, ScaleMatchesReference) {
  for (std::size_t n : pcl_test::interesting_lengths()) {
    auto a = pcl_test::make_vector<double>(n, 5);
    auto expected = a;
    auto got = a;
    vops::scale<double>(expected, 3.5, Backend::reference);
    vops::scale<double>(got, 3.5, backend());
    EXPECT_TRUE(pcl_test::all_equal(expected, got))
        << "n=" << n << " backend=" << parallel_compute::backend_name(backend());
  }
}

TEST_P(VectorOpsTest, SaxpyMatchesReference) {
  for (std::size_t n : pcl_test::interesting_lengths()) {
    auto y = pcl_test::make_vector<double>(n, 6);
    const auto x = pcl_test::make_vector<double>(n, 7);
    auto expected = y;
    auto got = y;
    vops::saxpy<double>(expected, x, -2.25, Backend::reference);
    vops::saxpy<double>(got, x, -2.25, backend());
    EXPECT_TRUE(pcl_test::all_equal(expected, got))
        << "n=" << n << " backend=" << parallel_compute::backend_name(backend());
  }
}

TEST_P(VectorOpsTest, EmptyInputIsANoOp) {
  const std::vector<double> empty;
  std::vector<double> out_empty;
  EXPECT_NO_THROW(vops::add<double>(empty, empty, out_empty, backend()));
  EXPECT_NO_THROW(vops::multiply<double>(empty, empty, out_empty, backend()));
  EXPECT_NO_THROW(vops::scale<double>(out_empty, 2.0, backend()));
  std::vector<double> y;
  EXPECT_NO_THROW(vops::saxpy<double>(y, empty, 1.0, backend()));
}

TEST_P(VectorOpsTest, SpecialValuesPropagate) {
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> a = {1.0, inf, nan, -inf};
  const std::vector<double> b = {1.0, 1.0, 1.0, 1.0};
  std::vector<double> got(4);
  vops::add<double>(a, b, got, backend());
  EXPECT_DOUBLE_EQ(2.0, got[0]);
  EXPECT_EQ(inf, got[1]);
  EXPECT_TRUE(std::isnan(got[2]));
  EXPECT_EQ(-inf, got[3]);

  // 0 * inf is NaN and must be produced, not optimised away.
  std::vector<double> zeros(4, 0.0);
  std::vector<double> scaled(4);
  vops::multiply<double>(zeros, a, scaled, backend());
  EXPECT_TRUE(std::isnan(scaled[1]));
}

TEST_P(VectorOpsTest, ByteCountingConvention) {
  // Documented as 3 * n * sizeof(T) for the binary operations.
  EXPECT_EQ(3U * 100U * sizeof(double), vops::bytes_touched<double>(100));
  EXPECT_EQ(3U * 100U * sizeof(float), vops::bytes_touched<float>(100));
}

TEST(VectorOpsArgumentValidation, MismatchedLengthsThrow) {
  const std::vector<double> a(10, 1.0);
  const std::vector<double> b(9, 1.0);
  std::vector<double> out(10, 0.0);
  EXPECT_THROW(vops::add<double>(a, b, out, Backend::reference), std::invalid_argument);
  EXPECT_THROW(vops::add<double>(a, b, out, Backend::auto_vectorized), std::invalid_argument);
  EXPECT_THROW(vops::add<double>(a, b, out, Backend::openmp), std::invalid_argument);
  EXPECT_THROW(vops::add<double>(a, b, out, Backend::simd), std::invalid_argument);
  EXPECT_THROW(vops::add<double>(a, b, out, Backend::best), std::invalid_argument);
}

TEST(VectorOpsArgumentValidation, SaxpyLengthMismatchThrows) {
  std::vector<double> y(10, 1.0);
  const std::vector<double> x(4, 1.0);
  EXPECT_THROW(vops::saxpy<double>(y, x, 1.0, Backend::reference), std::invalid_argument);
}

TEST(VectorOpsArgumentValidation, BackendNamesAreStable) {
  EXPECT_STREQ("reference", parallel_compute::backend_name(Backend::reference));
  EXPECT_STREQ("auto_vectorized", parallel_compute::backend_name(Backend::auto_vectorized));
  EXPECT_STREQ("openmp", parallel_compute::backend_name(Backend::openmp));
  EXPECT_STREQ("simd", parallel_compute::backend_name(Backend::simd));
  EXPECT_STREQ("best", parallel_compute::backend_name(Backend::best));
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(AllBackends, VectorOpsTest, ::testing::ValuesIn(pcl_test::all_backends()),
                         [](const ::testing::TestParamInfo<Backend>& p) {
                           return std::string(parallel_compute::backend_name(p.param));
                         });
