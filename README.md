# ParallelComputeLab

A CPU parallel-computing **optimization and benchmarking framework**: a small,
deliberately focused C++20 library of numerical kernels, an automated correctness
suite for every implementation variant of those kernels, and a reproducible
measurement pipeline with offline analysis.

The point of the project is not the number of algorithms in it.  It is to be able
to say, with evidence, *which implementation of a kernel is faster on this
machine, by how much, and how confident we are in that number* — and to be able
to re-run the measurement and get the same answer.

---

## Quick start

```bash
# 1. Configure, build and test (downloads pinned GoogleTest/Google Benchmark on
#    first run; needs network only at configure time)
cmake --preset gcc
cmake --build --preset gcc
ctest --preset gcc

# 2. Measure
python3 scripts/run_benchmarks.py --build build/gcc --preset quick --label gcc-quick

# 3. Validate and analyse
python3 scripts/validate_results.py results/raw/gcc-quick.json \
    --meta results/raw/gcc-quick.meta.json
python3 scripts/analyze_results.py results/raw/gcc-quick.json \
    --meta results/raw/gcc-quick.meta.json --outdir results/analysis
```

`analyze_results.py` prints a CSV, a markdown report and plots.  Every number in
them comes from the raw JSON; none is synthesised.

---

## What is in the library

| Area | Kernels | Variants |
|---|---|---|
| Element-wise | `add`, `multiply`, `scale`, `saxpy` | scalar reference, compiler auto-vectorised, OpenMP, explicit AVX2/NEON |
| Reductions | `sum`, `min`, `max`, `dot`, `mean` | same four variants |
| Matrix multiply | `C = A·B`, rectangular, row-major | reference (i-j-k), auto-vectorised (i-k-j), cache-blocked, OpenMP row panels, explicit 4-row AVX2/NEON micro-kernel, optional CBLAS |
| Memory | `copy`, `scale`, `triad`, streaming read, strided read | same four variants |
| Parallelism | `parallel_for`, `parallel_reduce`, scheduling workload | serial / `static` / `dynamic` / `guided`, configurable chunk size, false-sharing buffer |

Every core kernel has a **portable scalar reference** that is the correctness
oracle, and every optimized variant is tested against it.

### The optimization ladder

The GEMM family is the clearest illustration; the same structure applies
elsewhere.

| Level | Implementation | What it changes |
|---|---|---|
| 0 | `matmul_reference` | strict left-to-right accumulation, `B` accessed with a stride |
| 1 | `matmul_auto_vectorized` | `i-k-j` order: the inner loop walks `B` and `C` contiguously so the compiler can vectorise it |
| 2 | `matmul_blocked` | cache tiling over `m`/`k`/`n` so the inner working set fits in L1/L2 |
| 3 | `matmul_parallel` | the blocked kernel with the row-tile loop parallelised |
| 4 | `matmul_simd` | 4-row × AVX2 micro-kernel holding accumulators in registers |
| — | `matmul_blas` | external CBLAS, when one is linked |

Each step is measured against the one below it, not against a remembered number.

---

## Correctness before performance

A timed kernel that computes the wrong answer is worse than no benchmark at all,
so the harness validates **before** the timing loop starts:

```cpp
// benchmarks/kernels/matmul_bench.cpp, in outline
if (!pcl_bench::results_agree(data.ref, check, pcl_bench::gemm_tolerance())) {
  state.SkipWithError("... disagrees with the scalar reference GEMM");
  return;
}
for (auto _ : state) { /* only the kernel is timed */ }
```

A variant that fails is published as `validated=0` and skipped, never timed.

Five test binaries, each with its own purpose and CTest label:

```
unit_tests        contract: does the kernel do what the header promises?
numerical_tests   how far does each variant drift from the oracle?
edge_cases_tests  empty / single / odd / unaligned / non-square / invalid
concurrency_tests thread counts, repetition, reduction determinism, races
regression_tests  one executable test per defect found during development
```

Sanitizer configurations are provided (`asan` = ASan+UBSan, `tsan` = ThreadSanitizer
in its own build directory) and are run in CI.  No sanitizer diagnostic is
suppressed.

---

## Reproducible measurement

Each run writes four files, and validation fails without the metadata:

| File | Contents |
|---|---|
| `<label>.json` | raw Google Benchmark output, **every repetition preserved** |
| `<label>.meta.json` | CPU model and caches, OS, compiler id/version, **exact flags**, build type, dependency versions and provenance, git revision, seed, byte-counting convention |
| `<label>.log` | full console output |
| `<label>.system.json` | an independent host description collected before the run |

The analyzer groups by an environment fingerprint (CPU, core count, compiler,
build type, flags, SIMD level) and never averages across groups.

### Reported statistics

`median`, `min`, `mean`, **MAD** (median absolute deviation) and relative IQR.
The minimum is recorded but is not the headline: on a shared laptop the minimum
is an outlier, not a typical result.  No measurement is ever discarded.

### Claim rule

A speedup is only claimed when it exceeds twice the relative dispersion of the
baseline it is compared against.  Anything smaller is reported as *within
measurement uncertainty*.  The full rule set is in
[`docs/PERFORMANCE_METHODOLOGY.md`](docs/PERFORMANCE_METHODOLOGY.md).

---

## Result schema (version 1.0.0)

Raw files follow Google Benchmark's JSON format with our counters added;
`*.meta.json` is ours.  `scripts/validate_results.py` enforces both.

```jsonc
{
  "schema_version": "1.0.0",
  "pcl_version": "0.1.0",
  "generated_utc": "...",
  "label": "gcc-release",
  "raw_results": "results/raw/gcc-release.json",
  "command_line": "...verbatim argv...",
  "seed": 11400714819323198485,
  "config": { "vector_sizes": [...], "matrix_sizes": [...],
              "thread_counts": [...], "block_sizes": [...], "schedule_chunks": [...] },
  "build": { "compiler_id": "GNU", "compiler_version": "14.2.0",
             "build_type": "Release", "cxx_standard": "20",
             "cxx_flags": "-O3 -DNDEBUG -fopenmp",
             "simd_level": "portable", "native_arch": "OFF", "openmp": "..." },
  "cpu":   { "model": "...", "logical_cores": 8, "physical_cores": 4,
             "l1d_bytes": 32768, "l2_bytes": 262144, "l3_bytes": 6291456,
             "supported_simd": "sse2,sse4.2,avx,avx2,fma",
             "active_simd_kernel": "avx2" },
  "os":    { "system": "Linux", "release": "...", "machine": "x86_64" },
  "repo":  { "revision": "abc1234", "branch": "main", "dirty": false },
  "dependencies": ["googletest=fetched:v1.17.0", "openmp=...", "blas=..."],
  "measurement": { "repetitions": "5", "min_time": "0.05",
                   "timed_region": "kernel only; allocation and input fill are excluded",
                   "validation": "..." },
  "byte_counting": { "...": "documented per workload" }
}
```

Each raw benchmark entry adds: `elements`, `bytes_per_iter`, `bytes_per_second`,
`gflops_per_sec` (GEMM), `threads`, `validated`, `simd_lanes`, `iterations`, and
a `label` string `backend=<v>,threads=<p>,schedule=<s>,chunk=<c>`.

---

## Portability

The portable baseline never depends on anything wider than the compiler
default.  AVX2 kernels are compiled with a function-level
`__attribute__((target("avx2,fma")))` and entered only after a runtime CPUID
check, so a binary built on an AVX2 machine still runs on an SSE2-only one.
`-march=native` exists only as a preset for inspecting generated code and is
never the supported configuration.

---

## Repository layout

```
include/parallel_compute/   public headers (common, algorithms, simd, parallel, memory, system)
src/                        implementation
tests/{unit,numerical,edge_cases,concurrency,regression}/
benchmarks/{kernels,workloads,baselines,support}/
scripts/                    run_benchmarks.py, analyze_results.py,
                            validate_results.py, collect_system_info.py
configs/benchmark_defaults.json
docs/                       BUILDING, TESTING, BENCHMARKING,
                            PERFORMANCE_METHODOLOGY, LIMITATIONS
results/                    raw/ analysis/ reference/
```

---

## Profiling

```bash
# vectoriser report (GCC)
cmake -S . -B build/vec -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -fopt-info-vec-optimized -fopt-info-vec-missed"
cmake --build build/vec -j

# disassembly of one kernel
objdump -d --demangle build/gcc/src/libparallel_compute.a | less

# hardware counters, when the host permits
perf stat -e cycles,instructions,cache-misses \
  ./build/gcc/benchmarks/pcl_benchmarks --benchmark_filter='PCL/gemm/f64/simd/n=512'
```

Hardware counters are **optional**.  On a host with
`kernel.perf_event_paranoid > 1` — including the machine this project was
developed on — `perf stat` cannot read any counter, and analysis falls back to
timing plus static evidence.  That limitation is recorded in
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) rather than papered over.

---

## Documentation

| Document | Contents |
|---|---|
| [docs/BUILDING.md](docs/BUILDING.md) | dependencies, presets, options, sanitizers, portability |
| [docs/TESTING.md](docs/TESTING.md) | suites, tolerance policy, coverage, CI |
| [docs/BENCHMARKING.md](docs/BENCHMARKING.md) | commands, counters, byte-counting convention, schema |
| [docs/PERFORMANCE_METHODOLOGY.md](docs/PERFORMANCE_METHODOLOGY.md) | definitions, claim thresholds, optimisation loop, profiling commands |
| [docs/LIMITATIONS.md](docs/LIMITATIONS.md) | what could not be measured or verified here, and why |
| [docs/RESULTS.md](docs/RESULTS.md) | the measurements actually taken, with their provenance |

---

## Scope

CPU only.  No GPU, no distributed execution, no neural networks, no automatic
differentiation, no tensor framework, and no runtime or allocator of its own.
Those are not needed for the question this project exists to answer.

## License

MIT — see [LICENSE](LICENSE).
