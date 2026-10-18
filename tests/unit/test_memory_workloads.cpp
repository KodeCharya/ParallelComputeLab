// Contract tests for the memory-bound workloads.
#include <string>
#include <vector>

#include "parallel_compute/common/tolerance.hpp"
#include "parallel_compute/memory/workloads.hpp"
#include "test_support.hpp"

using parallel_compute::Backend;
namespace mem = parallel_compute::memory;

namespace {

class MemoryWorkloadTest : public ::testing::TestWithParam<Backend> {
 protected:
  Backend backend() const { return GetParam(); }
};

TEST_P(MemoryWorkloadTest, CopyMatchesReference) {
  for (std::size_t n : pcl_test::interesting_lengths()) {
    const auto src = pcl_test::make_vector<double>(n, 21);
    std::vector<double> expected(n), got(n, -7.0);
    mem::copy<double>(src, expected, Backend::reference);
    mem::copy<double>(src, got, backend());
    EXPECT_TRUE(pcl_test::all_equal(expected, got))
        << "n=" << n << " backend=" << parallel_compute::backend_name(backend());
  }
}

TEST_P(MemoryWorkloadTest, ScaleMatchesReference) {
  for (std::size_t n : pcl_test::interesting_lengths()) {
    auto data = pcl_test::make_vector<double>(n, 22);
    auto expected = data;
    auto got = data;
    mem::scale<double>(expected, 0.125, Backend::reference);
    mem::scale<double>(got, 0.125, backend());
    EXPECT_TRUE(pcl_test::all_equal(expected, got))
        << "n=" << n << " backend=" << parallel_compute::backend_name(backend());
  }
}

TEST_P(MemoryWorkloadTest, TriadMatchesReference) {
  for (std::size_t n : pcl_test::interesting_lengths()) {
    auto a = pcl_test::make_vector<double>(n, 23);
    const auto b = pcl_test::make_vector<double>(n, 24);
    const auto c = pcl_test::make_vector<double>(n, 25);
    std::vector<double> expected(n), got(n, 0.0);
    mem::triad<double>(expected, b, c, 2.5, Backend::reference);
    mem::triad<double>(got, b, c, 2.5, backend());
    EXPECT_TRUE(pcl_test::all_equal(expected, got))
        << "n=" << n << " backend=" << parallel_compute::backend_name(backend());
  }
}

TEST_P(MemoryWorkloadTest, SequentialSumMatchesReference) {
  for (std::size_t n : pcl_test::interesting_lengths()) {
    const auto src = pcl_test::make_vector<double>(n, 26);
    const double expected = mem::sequential_sum<double>(src, Backend::reference);
    const double got = mem::sequential_sum<double>(src, backend());
    EXPECT_LT(parallel_compute::relative_error(expected, got), 1e-12)
        << "n=" << n << " backend=" << parallel_compute::backend_name(backend());
  }
}

TEST_P(MemoryWorkloadTest, StridedSumMatchesReference) {
  for (std::size_t n : {0U, 1U, 2U, 5U, 16U, 17U, 64U, 100U, 4096U}) {
    const auto src = pcl_test::make_vector<double>(n, 27);
    for (std::size_t stride : {1U, 2U, 3U, 8U, 16U, 64U}) {
      const double expected = mem::strided_sum<double>(src, stride, Backend::reference);
      const double got = mem::strided_sum<double>(src, stride, backend());
      EXPECT_LT(parallel_compute::relative_error(expected, got), 1e-12)
          << "n=" << n << " stride=" << stride
          << " backend=" << parallel_compute::backend_name(backend());
    }
  }
}

// The strided contract is "span = buffer, count = ceil(size/stride)", which is
// what keeps every access in bounds for stride > 1.  An earlier version treated
// the span as the list of visited elements and read past the end of the buffer.
TEST(MemoryStridedContract, VisitsCeilSizeOverStrideElements) {
  EXPECT_EQ(0U, mem::strided_element_count(0, 4));
  EXPECT_EQ(1U, mem::strided_element_count(1, 4));
  EXPECT_EQ(1U, mem::strided_element_count(4, 4));
  EXPECT_EQ(2U, mem::strided_element_count(5, 4));
  EXPECT_EQ(100U, mem::strided_element_count(100, 1));
  EXPECT_EQ(100U, mem::strided_element_count(100, 0));  // stride 0 == stride 1
  EXPECT_EQ(7U, mem::strided_element_count(100, 16));
}

TEST(MemoryStridedContract, TouchesOnlyIndicesInsideTheBuffer) {
  for (std::size_t n = 1; n <= 200; ++n) {
    for (std::size_t stride = 1; stride <= 64; ++stride) {
      const std::size_t count = mem::strided_element_count(n, stride);
      ASSERT_LE((count - 1) * stride, n - 1) << "n=" << n << " stride=" << stride;
    }
  }
}

TEST_P(MemoryWorkloadTest, StrideZeroIsTreatedAsOne) {
  const auto src = pcl_test::make_vector<double>(32, 28);
  const double one = mem::strided_sum<double>(src, 1, backend());
  const double zero = mem::strided_sum<double>(src, 0, backend());
  EXPECT_DOUBLE_EQ(one, zero);
}

TEST(MemoryByteCounting, MatchesDocumentedConvention) {
  EXPECT_EQ(2U * 100U * sizeof(double), mem::bytes_touched(mem::Workload::copy, 100, sizeof(double)));
  EXPECT_EQ(2U * 100U * sizeof(double), mem::bytes_touched(mem::Workload::scale, 100, sizeof(double)));
  EXPECT_EQ(3U * 100U * sizeof(double),
            mem::bytes_touched(mem::Workload::triad, 100, sizeof(double)));
  EXPECT_EQ(100U * sizeof(double),
            mem::bytes_touched(mem::Workload::sequential, 100, sizeof(double)));
  EXPECT_EQ(100U * sizeof(double),
            mem::bytes_touched(mem::Workload::strided, 100, sizeof(double)));
  EXPECT_EQ(0U, mem::bytes_touched(mem::Workload::copy, 0, sizeof(double)));
}

TEST(MemoryArgumentValidation, MismatchedLengthsThrow) {
  const std::vector<double> a(10, 1.0);
  std::vector<double> b(9, 0.0);
  EXPECT_THROW(mem::copy<double>(a, b, Backend::reference), std::invalid_argument);
  EXPECT_THROW(mem::copy<double>(a, b, Backend::openmp), std::invalid_argument);
  EXPECT_THROW(mem::triad<double>(b, a, a, 1.0, Backend::reference), std::invalid_argument);
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(AllBackends, MemoryWorkloadTest,
                         ::testing::ValuesIn(pcl_test::all_backends()),
                         [](const ::testing::TestParamInfo<Backend>& p) {
                           return std::string(parallel_compute::backend_name(p.param));
                         });
