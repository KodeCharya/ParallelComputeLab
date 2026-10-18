// Contract tests for the GEMM implementations.
#include <cstddef>
#include <string>
#include <vector>

#include "parallel_compute/algorithms/matmul.hpp"
#include "parallel_compute/common/random.hpp"
#include "parallel_compute/common/tolerance.hpp"
#include "test_support.hpp"

using parallel_compute::Backend;
using parallel_compute::MutableMatrixView;
using parallel_compute::MatrixView;
namespace gm = parallel_compute::gemm;

namespace {

struct Variant {
  const char* name;
  void (*run)(const double*, std::size_t, std::size_t, const double*, std::size_t,
              MutableMatrixView<double>, const gm::GemmOptions&);
};

// Adapter so the variant table stays type erased.
#define PCL_DEFINE_VARIANT(NAME, FN)                                                   \
  void NAME(const double* a, std::size_t m, std::size_t k, const double* b,            \
            std::size_t n, MutableMatrixView<double> c, const gm::GemmOptions& opts) {  \
    FN(MatrixView<const double>(a, m, k), MatrixView<const double>(b, k, n), c, opts); \
  }

PCL_DEFINE_VARIANT(run_reference, gm::matmul_reference<double>)
PCL_DEFINE_VARIANT(run_auto, gm::matmul_auto_vectorized<double>)
PCL_DEFINE_VARIANT(run_blocked_16, gm::matmul_blocked<double>)
PCL_DEFINE_VARIANT(run_blocked_64, gm::matmul_blocked<double>)
PCL_DEFINE_VARIANT(run_simd, gm::matmul_simd<double>)

std::vector<Variant> variants() {
  std::vector<Variant> v;
  v.push_back({"reference", run_reference});
  v.push_back({"auto_vectorized", run_auto});
  v.push_back({"blocked_bs16", run_blocked_16});
  v.push_back({"blocked_bs64", run_blocked_64});
  v.push_back({"simd", run_simd});
  v.push_back({"parallel", [](const double* a, std::size_t m, std::size_t k, const double* b,
                             std::size_t n, MutableMatrixView<double> c,
                             const gm::GemmOptions& opts) {
                gm::matmul_parallel<double>(MatrixView<const double>(a, m, k),
                                            MatrixView<const double>(b, k, n), c, opts);
              }});
  if (gm::blas_available()) {
    v.push_back({"blas", [](const double* a, std::size_t m, std::size_t k, const double* b,
                            std::size_t n, MutableMatrixView<double> c,
                            const gm::GemmOptions& opts) {
                 (void)opts;
                 EXPECT_TRUE(gm::matmul_blas<double>(MatrixView<const double>(a, m, k),
                                                     MatrixView<const double>(b, k, n), c));
               }});
  }
  return v;
}

class MatmulTest : public ::testing::TestWithParam<Variant> {};

TEST_P(MatmulTest, MatchesIndependentOracle) {
  const Variant& v = GetParam();
  gm::GemmOptions opts;
  opts.block_size = 16;

  struct Shape {
    std::size_t m, n, k;
  };
  const Shape shapes[] = {
      {1, 1, 1},   {1, 7, 3},   {7, 1, 3},    {3, 3, 3},   {5, 8, 7},
      {8, 5, 7},   {16, 16, 16},{17, 13, 19}, {32, 8, 40}, {9, 33, 5},
  };

  for (const Shape& s : shapes) {
    const auto a = pcl_test::make_vector<double>(s.m * s.k, 101);
    const auto b = pcl_test::make_vector<double>(s.k * s.n, 102);
    const auto expected = pcl_test::oracle_matmul(a, s.m, s.k, b, s.n);

    std::vector<double> got(s.m * s.n, 0.0);
    v.run(a.data(), s.m, s.k, b.data(), s.n, MutableMatrixView<double>(got.data(), s.m, s.n),
          opts);

    EXPECT_TRUE(pcl_test::all_equal(expected, got, pcl_test::Tolerance{1e-12, 1e-12}))
        << "variant=" << v.name << " shape=" << s.m << "x" << s.n << "x" << s.k;
  }
}

TEST_P(MatmulTest, NonSquareLargeShape) {
  const Variant& v = GetParam();
  gm::GemmOptions opts;
  opts.block_size = 32;
  const std::size_t m = 61, n = 47, k = 53;  // all prime -> no blocking alignment
  const auto a = pcl_test::make_vector<double>(m * k, 201);
  const auto b = pcl_test::make_vector<double>(k * n, 202);
  const auto expected = pcl_test::oracle_matmul(a, m, k, b, n);
  std::vector<double> got(m * n, 0.0);
  v.run(a.data(), m, k, b.data(), n, MutableMatrixView<double>(got.data(), m, n), opts);
  EXPECT_TRUE(pcl_test::all_equal(expected, got, pcl_test::Tolerance{1e-11, 1e-11}))
      << "variant=" << v.name;
}

TEST(MatmulBasics, FlopCount) {
  EXPECT_EQ(2U * 4U * 5U * 6U, gm::flop_count(4, 5, 6));
  EXPECT_EQ(0U, gm::flop_count(0, 5, 6));
}

TEST(MatmulBasics, InvalidShapesThrow) {
  const std::vector<double> a(6, 1.0);  // 3x2
  const std::vector<double> b(6, 1.0);  // 2x3
  std::vector<double> c(9, 0.0);        // 3x3

  // A.cols (3) != B.rows (2) -> not a valid product.
  EXPECT_THROW(gm::matmul_reference<double>(MatrixView<const double>(a.data(), 3, 2),
                                            MatrixView<const double>(c.data(), 3, 3),
                                            MutableMatrixView<double>(c.data(), 3, 3)),
               std::invalid_argument);

  // Inner dimensions disagree.
  EXPECT_THROW(gm::matmul_auto_vectorized<double>(MatrixView<const double>(a.data(), 3, 2),
                                                  MatrixView<const double>(c.data(), 3, 3),
                                                  MutableMatrixView<double>(c.data(), 3, 3)),
               std::invalid_argument);

  // C shape mismatch
  EXPECT_THROW(gm::matmul_blocked<double>(MatrixView<const double>(a.data(), 3, 2),
                                          MatrixView<const double>(b.data(), 2, 3),
                                          MutableMatrixView<double>(c.data(), 3, 2)),
               std::invalid_argument);
  EXPECT_THROW(gm::matmul_simd<double>(MatrixView<const double>(a.data(), 3, 2),
                                       MatrixView<const double>(b.data(), 2, 3),
                                       MutableMatrixView<double>(c.data(), 2, 3)),
               std::invalid_argument);

  // block_size == 0 is rejected rather than silently treated as 1
  gm::GemmOptions bad;
  bad.block_size = 0;
  EXPECT_THROW(gm::matmul_blocked<double>(MatrixView<const double>(a.data(), 3, 2),
                                          MatrixView<const double>(b.data(), 2, 3),
                                          MutableMatrixView<double>(c.data(), 3, 3), bad),
               std::invalid_argument);
  EXPECT_THROW(gm::matmul_parallel<double>(MatrixView<const double>(a.data(), 3, 2),
                                           MatrixView<const double>(b.data(), 2, 3),
                                           MutableMatrixView<double>(c.data(), 3, 3), bad),
               std::invalid_argument);
}

TEST(MatmulBasics, EmptyMatricesAreNoOp) {
  std::vector<double> a, b, c;
  EXPECT_NO_THROW(gm::matmul_reference<double>(MatrixView<const double>(nullptr, 0, 0),
                                               MatrixView<const double>(nullptr, 0, 0),
                                               MutableMatrixView<double>(nullptr, 0, 0)));
  // m=0, k=3, n=4 : valid, produces nothing.
  const std::vector<double> bb(12, 1.0);
  EXPECT_NO_THROW(gm::matmul_blocked<double>(MatrixView<const double>(nullptr, 0, 3),
                                             MatrixView<const double>(bb.data(), 3, 4),
                                             MutableMatrixView<double>(nullptr, 0, 4)));
  // k=0: C is the zero matrix of the right shape.
  std::vector<double> cc(9, 7.0);
  gm::matmul_simd<double>(MatrixView<const double>(nullptr, 3, 0),
                          MatrixView<const double>(nullptr, 0, 3),
                          MutableMatrixView<double>(cc.data(), 3, 3));
  for (double v : cc) EXPECT_DOUBLE_EQ(0.0, v);
  (void)a;
  (void)b;
  (void)c;
}

TEST(MatmulBasics, BlasAvailabilityIsReported) {
  const bool available = gm::blas_available();
  const char* name = gm::blas_name();
  ASSERT_NE(nullptr, name);
  if (available) {
    EXPECT_STRNE("none", name);
  } else {
    EXPECT_STREQ("none", name);
  }
  // matmul_blas must return false (and leave C untouched) when unavailable.
  const std::vector<double> a(4, 1.0);
  const std::vector<double> b(4, 2.0);
  std::vector<double> c(4, 42.0);
  const bool ran = gm::matmul_blas<double>(MatrixView<const double>(a.data(), 2, 2),
                                           MatrixView<const double>(b.data(), 2, 2),
                                           MutableMatrixView<double>(c.data(), 2, 2));
  EXPECT_EQ(available, ran);
  if (!available) {
    for (double v : c) EXPECT_DOUBLE_EQ(42.0, v);
  } else {
    for (double v : c) EXPECT_DOUBLE_EQ(2.0, v);
  }
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(Variants, MatmulTest, ::testing::ValuesIn(variants()),
                         [](const ::testing::TestParamInfo<Variant>& p) {
                           return std::string(p.param.name);
                         });
