# Testing

The suite is five separate binaries, each with its own CTest label, so a failure
tells you immediately which kind of contract broke.

| Binary | Directory | What it establishes |
|---|---|---|
| `unit_tests` | `tests/unit` | each kernel does what its documented API contract says |
| `numerical_tests` | `tests/numerical` | how far each optimized variant drifts from the scalar oracle, against a stated budget |
| `edge_cases_tests` | `tests/edge_cases` | empty / single / odd / unaligned / degenerate shapes and invalid arguments |
| `concurrency_tests` | `tests/concurrency` | correctness at every thread count, repeated execution, reduction determinism, host-thread safety |
| `regression_tests` | `tests/regression` | one executable test per defect found while building this library |

## Running

```bash
cmake --preset gcc
cmake --build --preset gcc
ctest --preset gcc                 # all suites, 4 jobs, output on failure
ctest --test-dir build/gcc -L unit  # one suite
ctest --test-dir build/gcc -R "Regression" --output-on-failure
```

Or run a binary directly, which is often more useful while debugging:

```bash
./build/gcc/tests/unit_tests --gtest_filter='*Matmul*'
./build/gcc/tests/regression_tests --gtest_list_tests
```

## How correctness is decided

* **Exact comparison** is used only where it is the right criterion: element-wise
  operations (`add`, `multiply`, `scale`, `saxpy`, `copy`, `triad`) have no
  reduction, so any difference from the reference would be a defect.  This
  comparison is *exact*, and the tests assert it is.
* **Tolerant comparison** is used for anything whose accumulation order differs
  between variants: reductions and GEMM.  The helper is
  `parallel_compute::approx_equal`, which accepts
  `|a-b| <= abs_tol + rel_tol*|b|` and treats NaN==NaN as equal so that the
  NaN-propagation contract can be tested.
* Bitwise equality is **not** demanded of the AVX2/NEON kernels even for
  element-wise operations, because the compiler is free to contract
  `y + alpha*x` into an FMA in the library but not in the test.  Those tests use
  a small ULP tolerance and say so in a comment.

Typical budgets used in `tests/numerical`:

| Kernel | Budget | Why |
|---|---|---|
| reductions (f64) | relative error `< 1e-12` | `n = 2^18` reassociated summations |
| GEMM (f64) | `|d|/(|ref|+1) < 1e-12` | `k = 112` sequential accumulations |
| GEMM (f32) | `|d|/(|ref|+1) < 1e-5` | float eps ≈ 1.2e-7, `k = 56` |

A pure relative error is deliberately *not* used for GEMM: an entry that
cancels to ~0 would report an enormous relative error that says nothing about
accuracy.

## What is covered

Empty inputs, single elements, odd lengths, lengths on both sides of the AVX2
vector width (4 f64 / 8 f32), unaligned buffers at every offset 0..7,
non-square and degenerate matrices, dimensions that are not multiples of the
block size, invalid dimensions and shapes, large inputs, floating-point
tolerances, NaN and infinity behaviour, thread counts from 1 to twice the
logical core count, repeated execution, parallel reduction correctness, and
invalid configuration handling.

## Sanitizers

```bash
cmake --preset asan && cmake --build --preset asan && ctest --preset asan
cmake --preset tsan && cmake --build --preset tsan && ctest --preset tsan
```

* **AddressSanitizer + UndefinedBehaviorSanitizer** catch out-of-bounds access
  and undefined behaviour.
* **ThreadSanitizer** must be configured on its own and detects data races in
  the OpenMP paths and in the multi-host-thread test.

Sanitizer findings are not suppressed anywhere.  `-fno-sanitize-recover` is set
for UBSan so a finding fails the run instead of printing and continuing.

## Continuous integration

The workflow runs: five build configurations (GCC, Clang, Debug, no-OpenMP,
AArch64), both sanitizer configurations, the benchmark smoke test with the
optional BLAS installed, and a system-dependency build with
`PCL_FETCH_DEPENDENCIES=OFF`.

There are **no timing assertions** in CI.  Shared runners are not a measurement
environment (CPU frequency scaling, noisy neighbours and virtualisation all
dominate), so a timing threshold there would fail for reasons that have nothing
to do with the code.  The benchmark job runs one iteration per benchmark and
asserts only that results are produced, validated and serialisable.

## Adding a test

1. Put it in the suite that matches its purpose.
2. If it covers a defect you just fixed, put it in `tests/regression` and write
   a comment naming the defect, the mechanism and the date — as the existing
   tests there do.
3. Every optimized variant must be checked against the scalar reference, not
   against another optimized variant.
