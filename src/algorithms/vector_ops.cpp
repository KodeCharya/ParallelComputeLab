// Element-wise kernels: add, multiply, scale, saxpy.
//
// Four variants per kernel:
//   reference       scalar loop, auto-vectorisation disabled (correctness oracle)
//   auto_vectorized same algorithm, left to the compiler (Level 1)
//   openmp          OpenMP static partition of the same loop (Level 3)
//   simd            explicit AVX2/NEON intrinsics (Level 4)
//
// `best` deliberately selects the fastest *single-threaded* variant: mixing
// threading into an autotuning-style "best" would make the comparison against
// a fixed thread count meaningless.  Thread scaling is measured separately.
#include <cstddef>
#include <span>
#include <string>

#include "parallel_compute/algorithms/vector_ops.hpp"
#include "parallel_compute/common/attributes.hpp"
#include "parallel_compute/common/types.hpp"
#include "parallel_compute/parallel/scheduler.hpp"
#include "parallel_compute/simd/simd_kernels.hpp"

namespace parallel_compute::vector_ops {
namespace {

void check_lengths(std::size_t a, std::size_t b, std::size_t out, const char* op) {
  if (a != b || a != out) {
    throw_invalid_argument(std::string("vector_ops::") + op +
                           ": span lengths must be equal (got " + std::to_string(a) + ", " +
                           std::to_string(b) + ", " + std::to_string(out) + ")");
  }
}

/// True when the OpenMP backend would actually use more than one thread.
bool omp_worthwhile() noexcept { return parallel::max_threads() > 1; }

// --- reference -------------------------------------------------------------
template <class T>
PCL_NO_VECTORIZE_ATTR void add_reference(std::span<const T> a, std::span<const T> b,
                                         std::span<T> out) {
  PCL_NO_VECTORIZE_LOOP
  for (std::size_t i = 0; i < a.size(); ++i) {
    out[i] = a[i] + b[i];
  }
}

template <class T>
PCL_NO_VECTORIZE_ATTR void multiply_reference(std::span<const T> a, std::span<const T> b,
                                             std::span<T> out) {
  PCL_NO_VECTORIZE_LOOP
  for (std::size_t i = 0; i < a.size(); ++i) {
    out[i] = a[i] * b[i];
  }
}

template <class T>
PCL_NO_VECTORIZE_ATTR void scale_reference(std::span<T> y, T alpha) {
  PCL_NO_VECTORIZE_LOOP
  for (std::size_t i = 0; i < y.size(); ++i) {
    y[i] *= alpha;
  }
}

template <class T>
PCL_NO_VECTORIZE_ATTR void saxpy_reference(std::span<T> y, std::span<const T> x, T alpha) {
  PCL_NO_VECTORIZE_LOOP
  for (std::size_t i = 0; i < y.size(); ++i) {
    y[i] += alpha * x[i];
  }
}

// --- auto vectorized -------------------------------------------------------
template <class T>
void add_auto(std::span<const T> a, std::span<const T> b, std::span<T> out) {
  for (std::size_t i = 0; i < a.size(); ++i) {
    out[i] = a[i] + b[i];
  }
}

template <class T>
void multiply_auto(std::span<const T> a, std::span<const T> b, std::span<T> out) {
  for (std::size_t i = 0; i < a.size(); ++i) {
    out[i] = a[i] * b[i];
  }
}

template <class T>
void scale_auto(std::span<T> y, T alpha) {
  for (std::size_t i = 0; i < y.size(); ++i) {
    y[i] *= alpha;
  }
}

template <class T>
void saxpy_auto(std::span<T> y, std::span<const T> x, T alpha) {
  for (std::size_t i = 0; i < y.size(); ++i) {
    y[i] += alpha * x[i];
  }
}

// --- explicit SIMD (falls back to portable scalar code when unavailable) ---
template <class T>
void add_simd(std::span<const T> a, std::span<const T> b, std::span<T> out) {
  simd::kernels::add(out.data(), a.data(), b.data(), a.size());
}

template <class T>
void multiply_simd(std::span<const T> a, std::span<const T> b, std::span<T> out) {
  simd::kernels::multiply(out.data(), a.data(), b.data(), a.size());
}

template <class T>
void scale_simd(std::span<T> y, T alpha) {
  simd::kernels::scale(y.data(), y.size(), alpha);
}

template <class T>
void saxpy_simd(std::span<T> y, std::span<const T> x, T alpha) {
  simd::kernels::saxpy(y.data(), x.data(), y.size(), alpha);
}

}  // namespace

template <Real T>
void add(std::span<const T> a, std::span<const T> b, std::span<T> out, Backend backend) {
  check_lengths(a.size(), b.size(), out.size(), "add");
  switch (backend) {
    case Backend::reference:
      add_reference(a, b, out);
      return;
    case Backend::auto_vectorized:
      add_auto(a, b, out);
      return;
    case Backend::openmp:
      if (!omp_worthwhile()) {
        add_auto(a, b, out);
        return;
      }
      {
        parallel::ParallelConfig cfg;
        cfg.threads = parallel::max_threads();
        T* op = out.data();
        parallel::parallel_for(a.size(), cfg, [&](std::size_t i) { op[i] = a[i] + b[i]; });
      }
      return;
    case Backend::simd:
    case Backend::best:
      add_simd(a, b, out);
      return;
  }
}

template <Real T>
void multiply(std::span<const T> a, std::span<const T> b, std::span<T> out, Backend backend) {
  check_lengths(a.size(), b.size(), out.size(), "multiply");
  switch (backend) {
    case Backend::reference:
      multiply_reference(a, b, out);
      return;
    case Backend::auto_vectorized:
      multiply_auto(a, b, out);
      return;
    case Backend::openmp:
      if (!omp_worthwhile()) {
        multiply_auto(a, b, out);
        return;
      }
      {
        parallel::ParallelConfig cfg;
        cfg.threads = parallel::max_threads();
        T* op = out.data();
        parallel::parallel_for(a.size(), cfg, [&](std::size_t i) { op[i] = a[i] * b[i]; });
      }
      return;
    case Backend::simd:
    case Backend::best:
      multiply_simd(a, b, out);
      return;
  }
}

template <Real T>
void scale(std::span<T> y, T alpha, Backend backend) {
  switch (backend) {
    case Backend::reference:
      scale_reference(y, alpha);
      return;
    case Backend::auto_vectorized:
      scale_auto(y, alpha);
      return;
    case Backend::openmp:
      if (!omp_worthwhile()) {
        scale_auto(y, alpha);
        return;
      }
      {
        parallel::ParallelConfig cfg;
        cfg.threads = parallel::max_threads();
        parallel::parallel_for(y.size(), cfg, [&](std::size_t i) { y[i] *= alpha; });
      }
      return;
    case Backend::simd:
    case Backend::best:
      scale_simd(y, alpha);
      return;
  }
}

template <Real T>
void saxpy(std::span<T> y, std::span<const T> x, T alpha, Backend backend) {
  check_lengths(y.size(), x.size(), y.size(), "saxpy");
  switch (backend) {
    case Backend::reference:
      saxpy_reference(y, x, alpha);
      return;
    case Backend::auto_vectorized:
      saxpy_auto(y, x, alpha);
      return;
    case Backend::openmp:
      if (!omp_worthwhile()) {
        saxpy_auto(y, x, alpha);
        return;
      }
      {
        parallel::ParallelConfig cfg;
        cfg.threads = parallel::max_threads();
        T* yp = y.data();
        const T* xp = x.data();
        parallel::parallel_for(y.size(), cfg, [&](std::size_t i) { yp[i] += alpha * xp[i]; });
      }
      return;
    case Backend::simd:
    case Backend::best:
      saxpy_simd(y, x, alpha);
      return;
  }
}

template void add<double>(std::span<const double>, std::span<const double>, std::span<double>,
                         Backend);
template void add<float>(std::span<const float>, std::span<const float>, std::span<float>, Backend);
template void multiply<double>(std::span<const double>, std::span<const double>, std::span<double>,
                               Backend);
template void multiply<float>(std::span<const float>, std::span<const float>, std::span<float>,
                              Backend);
template void scale<double>(std::span<double>, double, Backend);
template void scale<float>(std::span<float>, float, Backend);
template void saxpy<double>(std::span<double>, std::span<const double>, double, Backend);
template void saxpy<float>(std::span<float>, std::span<const float>, float, Backend);

}  // namespace parallel_compute::vector_ops
