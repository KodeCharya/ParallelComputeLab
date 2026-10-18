# Building ParallelComputeLab

ParallelComputeLab targets **C++20 on Linux x86-64**, with AArch64 as a
supported secondary platform.  It builds with CMake ≥ 3.20 and either GCC or
Clang.

## Dependencies

| Dependency | Version | Required | How it is obtained |
|---|---|---|---|
| A C++20 compiler | GCC ≥ 11, Clang ≥ 15 | yes | system |
| CMake | ≥ 3.20 | yes | system |
| GoogleTest | 1.17.0 (pinned) | for tests | system package, else `FetchContent` |
| Google Benchmark | 1.9.4 (pinned) | for benchmarks | system package, else `FetchContent` |
| OpenMP | 4.5+ | no | system; the library builds without it |
| A CBLAS implementation | any | no | system; enables the GEMM baseline |

`cmake/Dependencies.cmake` prefers an installed system package and falls back to
downloading a **pinned** release with `FetchContent`.  Network access is needed
only at configure time in that case; afterwards the test suite and the
benchmark harness run entirely offline.

To force the system-package path (this is what the `portability` CI job checks):

```bash
sudo apt-get install cmake ninja-build libgtest-dev libbenchmark-dev libopenblas-dev
cmake -S . -B build/sysdeps -DPCL_FETCH_DEPENDENCIES=OFF
```

To pin different upstream versions:

```bash
cmake -S . -B build -DPCL_GTEST_VERSION=1.18.0 -DPCL_BENCHMARK_VERSION=1.9.5
```

## Presets

```bash
cmake --preset gcc      && cmake --build --preset gcc      && ctest --preset gcc
```

| Preset | What it is |
|---|---|
| `gcc` | GCC, `Release` (`-O3 -DNDEBUG`), OpenMP, portable ISA baseline |
| `clang` | the same sources with Clang, for compiler comparison |
| `gcc-avx2-native` | `-march=native`: **not portable**, only for inspecting generated code |
| `debug` | `Debug` with assertions and sanitizers off |
| `asan` | `RelWithDebInfo` + AddressSanitizer + UndefinedBehaviorSanitizer |
| `tsan` | `RelWithDebInfo` + ThreadSanitizer (must be its own build) |
| `portable-noopenmp` | no OpenMP, no explicit SIMD: the minimal baseline |

## Options

| Option | Default | Meaning |
|---|---|---|
| `PCL_BUILD_TESTS` | `ON` | build the five test binaries |
| `PCL_BUILD_BENCHMARKS` | `ON` | build `pcl_benchmarks` |
| `PCL_ENABLE_OPENMP` | `ON` | compile the OpenMP kernels; falls back with a warning |
| `PCL_ENABLE_SIMD` | `ON` | compile the explicit SIMD kernels |
| `PCL_ENABLE_BLAS` | `ON` | link a CBLAS if one is found |
| `PCL_FETCH_DEPENDENCIES` | `ON` | allow `FetchContent` when a package is missing |
| `PCL_SIMD_LEVEL` | `portable` | `portable`, `avx2` or `avx512` (see below) |
| `PCL_NATIVE_ARCH` | `OFF` | `-march=native`; the result is **not portable** |
| `PCL_WERROR` | `OFF` | warnings are errors |
| `PCL_ENABLE_ASAN` / `UBSAN` / `TSAN` | `OFF` | sanitizers (mutually exclusive configurations) |

### Portability and `-march=native`

The library never *requires* a wide instruction set to be correct:

* the portable baseline compiles with the compiler default, which on x86-64 is
  SSE2 and on AArch64 is FP/ASIMD;
* the AVX2 kernels are compiled with a **function-level target attribute**
  (`__attribute__((target("avx2,fma")))`) and are only entered after a runtime
  CPUID check, so a binary built on an AVX2 machine still runs on an SSE2-only
  one;
* `PCL_DISABLE_SIMD` (set by `PCL_ENABLE_SIMD=OFF`) forces every explicit
  kernel down to portable scalar code.

`PCL_SIMD_LEVEL` / `PCL_NATIVE_ARCH` change the flags the *translation unit* is
compiled with, which is a different thing from the runtime dispatch.  Any build
that uses them must have its flags recorded with its results — the harness does
this automatically in `<label>.meta.json` under `build.cxx_flags`.

## Sanitizers

Sanitizers need separate build directories; ASan and TSan in particular cannot
share a configuration (the build fails fast if you try).

```bash
cmake --preset asan && cmake --build --preset asan && ctest --preset asan
cmake --preset tsan && cmake --build --preset tsan && ctest --preset tsan
```

## Building without optional dependencies

```bash
cmake -S . -B build/minimal \
  -DPCL_ENABLE_OPENMP=OFF -DPCL_ENABLE_SIMD=OFF -DPCL_ENABLE_BLAS=OFF \
  -DPCL_BUILD_TESTS=OFF -DPCL_BUILD_BENCHMARKS=OFF
cmake --build build/minimal -j
```

This is the configuration that proves the library needs nothing beyond a C++20
compiler and pthreads.

## Install

```bash
cmake --install build/gcc --prefix /usr/local
```

Exports the CMake target `ParallelComputeLab::parallel_compute` and installs the
public headers, including the generated `build_config.hpp` that records the
compiler settings.
