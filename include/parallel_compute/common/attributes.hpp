// Compiler attribute helpers used throughout ParallelComputeLab.
#ifndef PARALLEL_COMPUTE_COMMON_ATTRIBUTES_HPP
#define PARALLEL_COMPUTE_COMMON_ATTRIBUTES_HPP

// ---------------------------------------------------------------------------
// Vectorization control
// ---------------------------------------------------------------------------
// The scalar reference kernels must not be auto-vectorized, otherwise they would
// stop being an independent oracle for the optimized kernels.
//
//   PCL_NO_VECTORIZE_LOOP  -- placed immediately *before* the loop statement.
//   PCL_NO_VECTORIZE_ATTR  -- applied to the enclosing function.
//
// GCC has no per-loop switch, so the function attribute is used there.  Clang
// honours the loop pragma and ignores the function attribute.
#if defined(__clang__)
#define PCL_NO_VECTORIZE_LOOP _Pragma("clang loop vectorize(disable) interleaved(vectorize)(disable)")
#define PCL_NO_VECTORIZE_ATTR
#elif defined(__GNUC__)
#define PCL_NO_VECTORIZE_LOOP
#define PCL_NO_VECTORIZE_ATTR __attribute__((optimize("no-tree-vectorize")))
#else
#define PCL_NO_VECTORIZE_LOOP
#define PCL_NO_VECTORIZE_ATTR
#endif

// ---------------------------------------------------------------------------
// Function-level target attributes (used for explicit SIMD kernels)
// ---------------------------------------------------------------------------
// This is what makes explicit SIMD safe on a portable build: the kernel is
// compiled for a wider ISA than the translation unit default, but it is only
// called after a runtime CPUID check, so an unsupported instruction can never
// be executed.
#if defined(__clang__) || (defined(__GNUC__) && !defined(__INTEL_COMPILER))
#define PCL_TARGET(isa) __attribute__((target(isa)))
#else
#define PCL_TARGET(isa)
#endif

#if defined(__GNUC__) || defined(__clang__)
#define PCL_ALWAYS_INLINE inline __attribute__((always_inline))
#define PCL_NOINLINE __attribute__((noinline))
#define PCL_HOT __attribute__((hot))
#define PCL_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define PCL_LIKELY(x) __builtin_expect(!!(x), 1)
#else
#define PCL_ALWAYS_INLINE inline
#define PCL_NOINLINE
#define PCL_HOT
#define PCL_UNLIKELY(x) (x)
#define PCL_LIKELY(x) (x)
#endif

#if defined(__GNUC__) || defined(__clang__)
#define PCL_RESTRICT __restrict__
#else
#define PCL_RESTRICT
#endif

#endif  // PARALLEL_COMPUTE_COMMON_ATTRIBUTES_HPP
