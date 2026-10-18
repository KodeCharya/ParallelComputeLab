// Memory-bound workloads: copy, scale, triad, sequential read, strided read.
//
// These kernels are dominated by data movement, so they are the reference
// point for any "optimization" that claims to improve memory behaviour.  Byte
// counts follow the convention documented in workloads.hpp and in
// docs/BENCHMARKING.md.
#include <algorithm>
#include <cstddef>
#include <span>
#include <string>

#include "parallel_compute/common/attributes.hpp"
#include "parallel_compute/common/types.hpp"
#include "parallel_compute/memory/workloads.hpp"
#include "parallel_compute/parallel/scheduler.hpp"
#include "parallel_compute/simd/simd_kernels.hpp"

namespace parallel_compute::memory {
namespace {

bool omp_worthwhile() noexcept { return parallel::max_threads() > 1; }

parallel::ParallelConfig default_cfg() {
  parallel::ParallelConfig cfg;
  cfg.threads = parallel::max_threads();
  return cfg;
}

// --- reference -------------------------------------------------------------
template <class T>
PCL_NO_VECTORIZE_ATTR void copy_reference(std::span<const T> src, std::span<T> dst) {
  PCL_NO_VECTORIZE_LOOP
  for (std::size_t i = 0; i < src.size(); ++i) {
    dst[i] = src[i];
  }
}

template <class T>
PCL_NO_VECTORIZE_ATTR void scale_reference(std::span<T> dst, T alpha) {
  PCL_NO_VECTORIZE_LOOP
  for (std::size_t i = 0; i < dst.size(); ++i) {
    dst[i] *= alpha;
  }
}

template <class T>
PCL_NO_VECTORIZE_ATTR void triad_reference(std::span<T> a, std::span<const T> b,
                                           std::span<const T> c, T alpha) {
  PCL_NO_VECTORIZE_LOOP
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = b[i] + alpha * c[i];
  }
}

template <class T>
PCL_NO_VECTORIZE_ATTR T sequential_reference(std::span<const T> src) {
  T total = T{0};
  PCL_NO_VECTORIZE_LOOP
  for (std::size_t i = 0; i < src.size(); ++i) {
    total += src[i];
  }
  return total;
}

template <class T>
PCL_NO_VECTORIZE_ATTR T strided_reference(std::span<const T> src, std::size_t stride) {
  if (stride == 0) stride = 1;
  T total = T{0};
  PCL_NO_VECTORIZE_LOOP
  for (std::size_t i = 0; i < src.size(); ++i) {
    total += src[i * stride];
  }
  return total;
}

// --- auto vectorized -------------------------------------------------------
template <class T>
void copy_auto(std::span<const T> src, std::span<T> dst) {
  for (std::size_t i = 0; i < src.size(); ++i) dst[i] = src[i];
}

template <class T>
void scale_auto(std::span<T> dst, T alpha) {
  for (std::size_t i = 0; i < dst.size(); ++i) dst[i] *= alpha;
}

template <class T>
void triad_auto(std::span<T> a, std::span<const T> b, std::span<const T> c, T alpha) {
  for (std::size_t i = 0; i < a.size(); ++i) a[i] = b[i] + alpha * c[i];
}

template <class T>
T sequential_auto(std::span<const T> src) {
  T total = T{0};
  for (std::size_t i = 0; i < src.size(); ++i) total += src[i];
  return total;
}

template <class T>
T strided_auto(std::span<const T> src, std::size_t stride) {
  if (stride == 0) stride = 1;
  T total = T{0};
  for (std::size_t i = 0; i < src.size(); ++i) total += src[i * stride];
  return total;
}

}  // namespace

const char* workload_name(Workload w) noexcept {
  switch (w) {
    case Workload::copy:
      return "copy";
    case Workload::scale:
      return "scale";
    case Workload::triad:
      return "triad";
    case Workload::sequential:
      return "sequential";
    case Workload::strided:
      return "strided";
  }
  return "unknown";
}

std::size_t bytes_touched(Workload w, std::size_t n, std::size_t element_size) noexcept {
  switch (w) {
    case Workload::copy:
    case Workload::scale:
      return 2 * n * element_size;
    case Workload::triad:
      return 3 * n * element_size;
    case Workload::sequential:
    case Workload::strided:
      return n * element_size;
  }
  return 0;
}

template <Real T>
void copy(std::span<const T> src, std::span<T> dst, Backend backend) {
  if (src.size() != dst.size()) {
    throw_invalid_argument("memory::copy: src and dst must have the same length (got " +
                           std::to_string(src.size()) + " and " + std::to_string(dst.size()) + ")");
  }
  switch (backend) {
    case Backend::reference:
      copy_reference(src, dst);
      return;
    case Backend::auto_vectorized:
    case Backend::simd:
    case Backend::best:
      copy_auto(src, dst);
      return;
    case Backend::openmp:
      if (!omp_worthwhile()) {
        copy_auto(src, dst);
        return;
      }
      parallel::parallel_for_chunks(src.size(), default_cfg(), [&](std::size_t begin,
                                                                  std::size_t count) {
        copy_auto(src.subspan(begin, count), dst.subspan(begin, count));
      });
      return;
  }
}

template <Real T>
void scale(std::span<T> dst, T alpha, Backend backend) {
  switch (backend) {
    case Backend::reference:
      scale_reference(dst, alpha);
      return;
    case Backend::auto_vectorized:
    case Backend::simd:
    case Backend::best:
      scale_auto(dst, alpha);
      return;
    case Backend::openmp:
      if (!omp_worthwhile()) {
        scale_auto(dst, alpha);
        return;
      }
      parallel::parallel_for_chunks(dst.size(), default_cfg(), [&](std::size_t begin,
                                                                  std::size_t count) {
        scale_auto(dst.subspan(begin, count), alpha);
      });
      return;
  }
}

template <Real T>
void triad(std::span<T> a, std::span<const T> b, std::span<const T> c, T alpha, Backend backend) {
  if (a.size() != b.size() || a.size() != c.size()) {
    throw_invalid_argument("memory::triad: a, b and c must have the same length");
  }
  switch (backend) {
    case Backend::reference:
      triad_reference(a, b, c, alpha);
      return;
    case Backend::auto_vectorized:
    case Backend::simd:
    case Backend::best:
      triad_auto(a, b, c, alpha);
      return;
    case Backend::openmp:
      if (!omp_worthwhile()) {
        triad_auto(a, b, c, alpha);
        return;
      }
      parallel::parallel_for_chunks(a.size(), default_cfg(), [&](std::size_t begin,
                                                                std::size_t count) {
        triad_auto(a.subspan(begin, count), b.subspan(begin, count), c.subspan(begin, count),
                   alpha);
      });
      return;
  }
}

template <Real T>
T sequential_sum(std::span<const T> src, Backend backend) {
  switch (backend) {
    case Backend::reference:
      return sequential_reference(src);
    case Backend::auto_vectorized:
      return sequential_auto(src);
    case Backend::openmp:
      if (!omp_worthwhile()) return sequential_auto(src);
      return parallel::parallel_reduce<T>(src.size(), default_cfg(), T{0},
                                          [&](std::size_t i) { return src[i]; });
    case Backend::simd:
    case Backend::best:
      return simd::kernels::sum(src.data(), src.size());
  }
  return sequential_auto(src);
}

template <Real T>
T strided_sum(std::span<const T> src, std::size_t stride, Backend backend) {
  switch (backend) {
    case Backend::reference:
      return strided_reference(src, stride);
    case Backend::auto_vectorized:
    case Backend::simd:
    case Backend::best:
      // A strided pattern is never a win for a hand written vector kernel; the
      // hardware prefetcher handles it or it does not.
      return strided_auto(src, stride);
    case Backend::openmp:
      if (!omp_worthwhile()) return strided_auto(src, stride);
      return parallel::parallel_reduce<T>(src.size(), default_cfg(), T{0},
                                          [&](std::size_t i) { return src[i * (stride == 0 ? 1 : stride)]; });
  }
  return strided_auto(src, stride);
}

template void copy<double>(std::span<const double>, std::span<double>, Backend);
template void copy<float>(std::span<const float>, std::span<float>, Backend);
template void scale<double>(std::span<double>, double, Backend);
template void scale<float>(std::span<float>, float, Backend);
template void triad<double>(std::span<double>, std::span<const double>, std::span<const double>,
                            double, Backend);
template void triad<float>(std::span<float>, std::span<const float>, std::span<const float>, float,
                           Backend);
template double sequential_sum<double>(std::span<const double>, Backend);
template float sequential_sum<float>(std::span<const float>, Backend);
template double strided_sum<double>(std::span<const double>, std::size_t, Backend);
template float strided_sum<float>(std::span<const float>, std::size_t, Backend);

}  // namespace parallel_compute::memory
