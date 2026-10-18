# Performance methodology

This document describes how a claim is allowed to be made in this project.
It is deliberately strict: the point is not to produce impressive numbers, it is
to produce numbers someone else can check.

## 1. Definitions

For a kernel *K* with a workload *W* and a thread count *p*:

```
T(p)  = median wall-clock time of one execution of K(W) using p threads
speedup(p)   = T(1) / T(p)
efficiency(p)= speedup(p) / p
```

`T(1)` **must** be the same algorithm, on the same data, with one participating
thread.  Comparing an OpenMP implementation at `p = 4` against a serial
reference at `p = 1` is a speedup only if the algorithm is identical, which is
exactly what `scripts/analyze_results.py` enforces by keying the baseline on
(family, kernel, dtype, variant, size, block).

A *variant* comparison uses `T(reference, p=1) / T(variant, p=1)`.  Both sides
are single threaded, so the number reflects the implementation change and
nothing else.

## 2. What is inside the timed region

* Input allocation, random fill and reference computation happen **before**
  `benchmark::State` is entered.
* Only the kernel call is inside the loop.
* Reduction results are passed through `benchmark::DoNotOptimize` so they cannot
  be dead-code eliminated.
* Correctness is validated once, before the loop.  A variant that disagrees with
  the scalar reference is reported as `validated=0` and skipped, never timed and
  published as a measurement.

End-to-end timings (including allocation) are deliberately **not** mixed into
these results; if that overhead matters for a specific use case it has to be
measured separately and labelled as such.

## 2a. How the correctness gate decides "correct"

This matters more than it first appears, because the first attempt at it was
wrong and the benchmark run itself caught it.

**Element-wise operations** (`add`, `multiply`, `scale`, `saxpy`, `copy`,
`triad`) have no reduction, so every backend must agree with the reference
**exactly**.  A tolerance here would hide a real bug.

**Reductions and GEMM** reassociate the sum, so they need a real error model.
For a summation of `n` terms,

```
|approx - exact|  <=  (n - 1) * eps * sum_i |x_i|
```

holds for *every* ordering — naive and pairwise alike.  Two different orders
therefore differ by at most twice that, and the gate compares against

```
tol_abs = 2 * (n - 1) * eps * sum_i |x_i|
```

with `eps = DBL_EPSILON`.  `sum|x_i|` is computed once, outside the timed region.

**The naive alternative does not work.**  The first full measurement run used a
flat `1e-12` relative tolerance, and the correctness gate rejected `reduce_dot`
at `n = 16 777 216` with 4 and 8 threads.  That rejection was *correct
behaviour on a wrong criterion*: at that size the flat serial reference is
itself the **least** accurate ordering available (its error grows like `n`),
while the OpenMP reduction is a tree whose error grows like `log2(n)`.  The two
differ by far more than `1e-12` relative — and both are "right" to within the
problem's own conditioning.  A fixed relative epsilon is only meaningful while
`n * eps` stays comfortably below the magnitude of the answer, which is not
something a benchmark can assume.

The same forward-error bound is applied to GEMM, per output entry, with `n = k`
(the dot-product length) and `sum|x_i|` replaced by the sum of `|a * b|` along
the part of `B` that entry consumes.

The practical consequence is worth stating plainly: **the correctness gate is
part of the experiment, not a formality.**  A gate that is too tight produces
false alarms that look like defects; a gate that is too loose hides real ones.

## 3. Repetitions and dispersion

* `--benchmark_repetitions=5` by default; every repetition is written to the
  raw JSON, and none is discarded.
* Reported statistic is the **median**.  The minimum is also reported, but it is
  not the headline: on a shared or thermally throttled machine the minimum is an
  outlier, not a typical result.
* Dispersion is reported as the **median absolute deviation** (MAD) and, when
  there are at least 4 repetitions, the relative interquartile range
  `(Q3-Q1)/median`.
* `--benchmark_enable_random_interleaving=true` is used so that slow drift in
  machine state (frequency, cache warming, other load) is spread across
  repetitions instead of being confounded with the benchmark order.
* **No measurement is ever silently dropped.**  If a run produced outliers they
  are in the raw file, in the median, and in the dispersion column.

## 4. Claim thresholds

A speedup may be claimed only when

```
median_speedup - 2 * relative_MAD(T(1)) > 1.0
```

i.e. when the improvement is larger than twice the relative dispersion of the
baseline it is compared against.  Anything smaller is reported as "within
measurement uncertainty", not as an improvement.

A regression is claimed under the same rule, with the roles swapped.

## 5. Prohibited comparisons

* Different CPUs or different build configurations.  Results are grouped by an
  environment fingerprint (CPU model, core count, compiler id+version, build
  type, flags, SIMD level) and never averaged across groups.
* Different algorithms or numerical semantics without an explicit explanation.
  The GEMM variants differ only in accumulation order; `mean` is `sum/n` and is
  *not* more accurate than `sum`; `Backend::best` selects the fastest
  single-threaded variant and never implies threading.
* `-march=native` results against a portable build.  A native build is a
  different binary, not the same program.

## 6. Reported units

Bandwidth and GFLOP/s use the byte convention in `docs/BENCHMARKING.md`.  Where
that convention is a model or a lower bound (GEMM, strided reads) the report
says so rather than presenting the number as a measurement of DRAM traffic.

## 7. The optimisation loop

Every optimization recorded in this repository followed the same steps, in this
order:

1. Measure the baseline and record its dispersion.
2. State a specific hypothesis about the bottleneck.
3. Change **one** thing.
4. Run the correctness tests (`ctest --preset gcc`).
5. Re-measure under the same conditions (same host, same flags, same
   repetitions, ideally in the same run as the baseline so drift is shared).
6. Apply the threshold in §4: keep, or revert.
7. Record the finding, including the ones that did not work.

Hypotheses are not accepted because they are plausible.  `perf stat` counters
are used where the kernel is available and permissions allow; where they are not,
the missing information is stated rather than substituted with a guess.  See
`docs/LIMITATIONS.md` for what could not be measured on the development host.

## 8. Profiling commands

```bash
# Compiler vectorisation report (GCC)
cmake -S . -B build/vec -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -fopt-info-vec-optimized -fopt-info-vec-missed"
cmake --build build/vec -j
grep -rn "vectorized\|not vectorized" build/vec/src/CMakeFiles/parallel_compute.dir/

# Compiler vectorisation report (Clang)
cmake -S . -B build/vec -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS="-O3 -Rpass-analysis=loop-vectorize -Rpass-missed=loop-vectorize"

# Generated assembly for one kernel
objdump -d --demangle build/gcc/src/libparallel_compute.a | less
objdump -d --demangle build/gcc/src/libparallel_compute.a \
  | awk '/<.*gemm_panel4_f64_avx2.*>:/,/ret/' | head -60

# Hardware counters (optional; see Limitations if this fails)
perf stat -e task-clock,cycles,instructions,cache-references,cache-misses \
  ./build/gcc/benchmarks/pcl_benchmarks \
    --benchmark_filter='PCL/gemm/f64/simd/n=512' --benchmark_min_time=2 \
    --pcl_matrix_sizes=512 --pcl_threads=1

# IPC and branch behaviour
perf stat -e cycles,instructions,branches,branch-misses,L1-dcache-load-misses \
  ./build/gcc/benchmarks/pcl_benchmarks --benchmark_filter='PCL/add.*'

# Host description used for every result
python3 scripts/collect_system_info.py --out results/system_info.json
```

If `perf` reports `perf_event_paranoid > 1` or `<not supported>`, hardware
counters are simply unavailable in this environment.  That is recorded and the
measurement falls back to wall-clock timing — no bottleneck is inferred without
evidence.
