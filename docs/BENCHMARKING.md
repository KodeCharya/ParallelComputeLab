# Benchmarking

## The command that produces a reproducible result set

```bash
python3 scripts/run_benchmarks.py --build build/gcc --preset full --label gcc-release
```

It writes four files to `results/raw/`:

| File | Contents |
|---|---|
| `gcc-release.json` | raw Google Benchmark measurements — **every repetition is preserved** |
| `gcc-release.meta.json` | machine, toolchain, flags, dependencies, seed, byte convention |
| `gcc-release.log` | the full console output |
| `gcc-release.system.json` | an independent record of the host, collected before the run |

Then it validates the result set and prints the `analyze_results.py` command to
run next.  Validation is not optional: a measurement without its `.meta.json`
has unknown provenance and cannot be reproduced.

## Direct invocation

Everything the wrapper does can be done by hand:

```bash
./build/gcc/benchmarks/pcl_benchmarks \
  --pcl_sizes=1048576,16777216 \
  --pcl_matrix_sizes=128,256,512,1024 \
  --pcl_threads=1,2,4,8 \
  --pcl_blocks=32,64,128 \
  --pcl_chunks=1,16,1024 \
  --pcl_seed=1234 \
  --pcl_label=my-run \
  --benchmark_repetitions=5 \
  --benchmark_enable_random_interleaving=true \
  --benchmark_out=results/raw/my-run.json \
  --benchmark_out_format=json
```

### `pcl_benchmarks` flags

| Flag | Meaning |
|---|---|
| `--pcl_sizes=` | element counts for vector kernels (default `1048576,16777216`) |
| `--pcl_matrix_sizes=` | square GEMM edge lengths (default `128,256,512`) |
| `--pcl_threads=` | thread counts to sweep (default `1,2,4`) |
| `--pcl_blocks=` | GEMM cache-block sizes (default `32,64`) |
| `--pcl_chunks=` | chunk sizes for dynamic/guided scheduling (default `1,64`) |
| `--pcl_seed=` | seed for the deterministic input generator |
| `--pcl_label=` | free-form tag copied into the metadata |

Google Benchmark's own flags (`--benchmark_filter`, `--benchmark_min_time`,
`--benchmark_repetitions`, `--benchmark_out`, …) all work as usual.

### Presets

`configs/benchmark_defaults.json` defines `smoke`, `quick`, `full`, `gemm` and
`scheduling` presets plus the default workload lists.

```bash
python3 scripts/run_benchmarks.py --build build/gcc --preset smoke  --label smoke
python3 scripts/run_benchmarks.py --build build/gcc --preset gemm    --label gemm
```

## Benchmark names

```
PCL/<kernel>/<dtype>/n=<elements>/<variant>/T=<threads>
PCL/gemm/f64/<variant>/n=<edge>/block=<block>/T=<threads>
PCL/reduce_<op>/f64/n=<elements>/<variant>/T=<threads>
PCL/mem_<workload>/f64/n=<elements>/<variant>/T=<threads>
PCL/scheduling/<policy>/items=<n>/chunk=<c>/T=<threads>
PCL/false_sharing/<layout>/T=<threads>
```

Every name component is machine readable, which is what lets the analysis group
measurements without guessing.  `T=1` is always present for every variant, so a
speedup always has its own denominator.

## Counters

Each entry publishes the same counter names:

| Counter | Meaning |
|---|---|
| `elements` | elements processed per iteration |
| `bytes_per_iter` | bytes per iteration under the convention below |
| `bytes_per_second` | derived by Google Benchmark from `bytes_per_iter` |
| `gflops_per_sec` | rate counter; absent for non-GEMM kernels |
| `threads` | participating threads |
| `validated` | `1` if the pre-timing correctness gate passed |
| `simd_lanes` | lanes per vector in the SIMD kernel that actually ran |
| `iterations` | iterations in this repetition |
| `m`, `n`, `k`, `block`, `stride`, `chunk` | workload shape and tuning |

plus `label`, a string of the form
`backend=<variant>,threads=<p>,schedule=<policy>,chunk=<c>`.

## Byte-counting convention

Bandwidth is meaningless without stating what was counted.  The convention used
everywhere in this project:

| Workload | Bytes per element | Reasoning |
|---|---|---|
| `add`, `multiply`, `saxpy`, `triad` | `3 * n * sizeof(T)` | two reads, one write |
| `copy`, `scale` | `2 * n * sizeof(T)` | one read, one write |
| `sequential` (streaming read) | `1 * n * sizeof(T)` | logical bytes read |
| `strided` (stride 16, f64) | `1 * n * sizeof(T)` | **logical lower bound**: the hardware moves whole cache lines, so the reported number overstates nothing and understates the traffic |
| GEMM | `2 * (m*n + k*n + m*k) * sizeof(T)` | **model** of the traffic a cache-blocked kernel would cause from main memory; not a measurement of it |

The `strided` and GEMM rows are called out because a naive reading of the number
would overstate achievable bandwidth.  A strided read at stride 16 touches
128 bytes per 8-byte element, so its "bandwidth" is at most a sixteenth of its
actual DRAM traffic.

## Workload sizes

| n (f64) | Bytes per array | Intent |
|---|---|---|
| 1 048 576 | 8 MiB | L2/L3 resident — measures compute and cache bandwidth |
| 16 777 216 | 128 MiB | larger than any laptop L3 — measures main-memory bandwidth |

The default GEMM sizes (128, 256, 512) are chosen so a naive `i-j-k` reference
finishes in seconds, not minutes: at `n = 1024` the reference kernel alone
dominates the run time.  Use `--pcl_matrix_sizes=1024` deliberately, and expect
the reference entry to be slow — that *is* the finding.

## Correctness gate

A variant is checked against the scalar reference **before** it is timed.

| Kernel class | Criterion |
|---|---|
| element-wise (`add`, `multiply`, `scale`, `saxpy`, `copy`, `triad`) | **exact** equality — these have no reduction, so any difference is a defect |
| `min` / `max` | **exact** equality — comparisons do not reassociate |
| `sum`, `mean`, `dot`, streaming/strided reads | forward-error bound `2*(n-1)*eps*sum|x_i|` |
| GEMM, every variant | per-entry forward-error bound `2*(k-1)*eps*sum|a*b|` |

A variant that fails is reported by Google Benchmark as an error, is counted as
a validation problem by `validate_results.py`, and never contributes a timing.
The rationale — and the concrete case where a flat relative tolerance produced
a false alarm — is in
[`PERFORMANCE_METHODOLOGY.md`](PERFORMANCE_METHODOLOGY.md) §2a.

## Analysis

```bash
python3 scripts/validate_results.py results/raw/run.json --meta results/raw/run.meta.json
python3 scripts/analyze_results.py  results/raw/run.json --meta results/raw/run.meta.json \
    --outdir results/analysis
```

Outputs:

* `measurements.csv` — one row per measurement: median, min, mean, MAD,
  relative MAD, relative IQR, speedup, efficiency, bandwidth, GFLOP/s, source file;
* `report.md` — environments, validation status, thread-scaling table,
  implementation-comparison table, missing configurations, and every problem
  found while reading the data;
* `plots/*.png` — per-kernel thread scaling and workload-size curves.

`analyze_results.py` refuses to compute a derived number from a missing
baseline, groups results by environment fingerprint, and warns when a single
input set mixes CPUs or builds.  It never produces a number that is not in the
raw JSON.
