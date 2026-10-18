# Known limitations

This file records what this project **cannot** currently do, and what could not
be verified in the environment it was developed in.  Everything here is a
statement about measurements and coverage, not about intentions.

## 1. Hardware performance counters are unavailable on the development host

```
$ perf stat -e cycles,instructions true
access to performance monitoring and observability operations for processes
without CAP_PERFMON, CAP_SYS_PTRACE or CAP_SYS_ADMIN Linux capability.
perf_event_paranoid setting is 3
```

`kernel.perf_event_paranoid` is `3` on the development machine, so `perf stat`
cannot read hardware counters at all — not even for the calling process.
Consequently:

* **No** IPC, cache-miss, branch-miss or stall-cycle measurement was taken.
* Bottleneck claims in this repository rest on **timing evidence and static
  analysis only** (vectoriser reports, disassembly, size sweeps).  Where a
  bottleneck was not demonstrated, it is not claimed.
* `scripts/collect_system_info.py` records `perf.usable = false` so this is
  visible in every result set.

On a host with `perf_event_paranoid <= 1` the commands in
`docs/PERFORMANCE_METHODOLOGY.md` work unchanged.  Re-running them there would
let the "why is this faster" questions be answered with counters instead of
inference.

## 2. No external BLAS was available

The development machine has `libblas.so.3` at runtime but **no `cblas.h`**, so
`PCL_HAVE_BLAS` is off and `gemm::matmul_blas()` returns `false`.

Status of the BLAS baseline: **NOT RUN** locally.  The code path is written and
unit-tested for the "no BLAS" contract (it must return `false` and leave `C`
untouched — `MatmulBasics.BlasAvailabilityIsReported` checks exactly that), and
the CI `benchmark-smoke` job installs `libopenblas-dev` so the comparison runs
there.  Until that job's artifact has been inspected, **no speedup against a
third-party BLAS is claimed anywhere in this repository.**

## 3. AArch64 / NEON: implemented, not executed locally

The NEON kernels in `src/simd/simd_kernels.cpp` are written for AArch64 and are
compiled by the `aarch64-neon` CI job, but they were **not executed on the
development host**, which is x86-64.  Their local status is therefore
**NOT RUN**, not "verified".

Consequences to keep in mind when reading those numbers:

* The double-precision NEON path relies on the compiler contracting a 2-lane
  loop into `vld1q`/`vst1q` + `vfmaq` pairs rather than using explicit
  intrinsics, so its vectorisation depends on the compiler and was not inspected
  on real hardware here.
* The AArch64 SIMD GEMM micro-kernel assumes `ncols >= 4` for `float`; the
  narrower cases are dispatched to a scalar loop by the caller.

## 4. The development machine is small and thermally shared

```
CPU:      Intel(R) Core(TM) i5-8265U CPU @ 1.60GHz  (4 cores / 8 threads, 1 socket)
Caches:   L1d 32 KiB x4, L2 256 KiB x4, L3 6 MiB x1
Memory:   ~7.5 GiB
Scaling:  CPU scaling enabled (governor is not `performance`)
```

* `intel_pstate` scaling is active, so timings drift with thermal and power
  state.  Google Benchmark warns about this on startup.  Repetitions, random
  interleaving and median reporting mitigate it; they do not eliminate it.
* Only 8 logical cores means the maximum thread count is small and the scaling
  curves are short.  Nothing in the published results should be read as a
  statement about large-scale machines.
* It is a laptop, so it is shared with whatever else is running.  The recorded
  `load_average` and dispersion columns are the evidence for how much that
  mattered.

## 5. Deliberate scope limits

These are choices, not accidents:

* **No `Backend::best` that includes OpenMP.** `Backend::best` picks the fastest
  *single-threaded* variant.  Folding threading into an autotuning choice would
  make "best vs reference at T=1" meaningless.  Thread scaling is measured
  separately and explicitly.
* **No explicit SIMD kernel for `min`/`max`.** Comparison-based reductions are
  branch bound and did not justify hand-written intrinsics.  `Backend::simd` for
  those maps to the serial fold, and the benchmark registers that entry anyway
  so the claim is visible in the data rather than only in this file.
* **No hand-written assembly.** No profiling result on any available host
  demonstrated a need for it.
* **No custom task scheduler.** OpenMP's `static`/`dynamic`/`guided` are
  measured first; nothing in the results identified a scheduling problem that
  would justify writing one.
* **No AVX-512 kernels.** The development CPU has no AVX-512.  Writing kernels
  that cannot be executed here would violate the "never execute an unsupported
  instruction" rule in spirit and could not be validated.
* **No f32 element-wise benchmarks.** The library implements `float` and tests
  it, but the benchmark suite measures `f64` for element-wise and memory
  workloads and `f64` for GEMM (plus one float GEMM numerical test).  f32
  bandwidth numbers are **not available**.

## 6. Numerical limitations

* Reduction and GEMM results are only defined to within a tolerance, because
  parallel and SIMD variants reassociate the sum.  The reference variant's
  accumulation order is the strict left-to-right one; everything else is
  unspecified and documented as such.
* `mean` is `sum / n`.  It is **not** more accurate than `sum`, and on
  adversarial input (large dynamic range) it is the *less* stable of the two.
  The library exposes it because the workloads need it, not because it is good.
* A tree reduction cannot recover information a flat summation also loses.
  `tests/numerical` documents this limit rather than hiding it.

## 7. Analysis-tool limitations

* `scripts/analyze_results.py` needs `matplotlib` for plots.  Without it the
  CSV and markdown report are still produced and a note says the plots were
  skipped.
* Dispersion statistics are computed over the repetitions Google Benchmark
  recorded in a single process.  They describe within-run variability, not
  run-to-run reproducibility across machines or days.  Repeating a run and
  comparing the medians is the way to check that.
* `rel_iqr` is `NaN` for fewer than 4 repetitions, because a quartile of one or
  two samples is not informative.  The column is left empty rather than
  filled with a misleading number.

## 8. Things that were tried and rejected

Recorded so they are not re-tried blindly; the numbers behind these statements
are in `results/analysis/report.md`.

* **`-march=native` as the supported configuration.** Rejected: it produces a
  binary that cannot be compared against, or even run on, another machine.  It
  survives only as the `gcc-avx2-native` preset, used for inspecting generated
  code.
* **`Backend::best` = "fastest overall".** Rejected: it silently mixes thread
  counts into what is otherwise a single-threaded comparison.
* **A hand-written OpenMP scheduling queue.** Rejected before implementation:
  the scheduling experiment measures OpenMP's own policies first, and no
  measured gap justified a bespoke scheduler.
