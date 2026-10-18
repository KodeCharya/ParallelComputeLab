#include "parallel_compute/common/types.hpp"

namespace parallel_compute {

const char* backend_name(Backend b) noexcept {
  switch (b) {
    case Backend::reference:
      return "reference";
    case Backend::auto_vectorized:
      return "auto_vectorized";
    case Backend::openmp:
      return "openmp";
    case Backend::simd:
      return "simd";
    case Backend::best:
      return "best";
  }
  return "unknown";
}

bool backend_is_parallel(Backend b) noexcept { return b == Backend::openmp; }

void throw_invalid_argument(const std::string& what) { throw std::invalid_argument(what); }

}  // namespace parallel_compute
