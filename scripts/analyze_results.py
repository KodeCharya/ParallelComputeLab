#!/usr/bin/env python3
"""Analyse ParallelComputeLab benchmark results.

Reads raw Google Benchmark JSON plus its `*.meta.json` sidecar and produces:

  * a CSV of every measurement (median, dispersion, bandwidth, GFLOP/s)
  * derived speedup / efficiency tables against the T(1) baseline of the SAME
    algorithm, as required by the methodology in docs/PERFORMANCE_METHODOLOGY.md
  * scaling plots and workload-size comparison plots
  * a list of missing configurations and of results that failed validation
  * a markdown report that keeps every number tied to its source file

Design rules
------------
* No synthetic numbers are ever produced.  If a baseline is missing, the
  derived column is left empty and the gap is reported.
* Results from different CPUs or different build configurations are never
  averaged together.  They are grouped by an environment fingerprint and the
  tool refuses to mix them unless --allow-mixed-environments is given.
* Every row keeps a `source` column naming the raw JSON file it came from.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:  # matplotlib is optional for the CSV/table outputs
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except Exception:  # pragma: no cover - depends on the environment
    HAVE_MPL = False

SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Measurement:
    """One (benchmark, repetition-set) aggregate."""

    name: str
    family: str
    token: str                      # free-form name segment, e.g. "add" or "gemm"
    fields: dict[str, str]
    times_ns: list[float]
    entries: int
    elements: float | None
    bytes_per_iter: float | None
    threads: float
    schedule: str
    chunk: float | None
    block: float | None
    validated: bool
    gflops_per_sec: list[float]
    bytes_per_second: list[float]
    repetitions: int
    source: str
    environment: str

    # -- derived ---------------------------------------------------------
    @property
    def median_ns(self) -> float:
        return statistics.median(self.times_ns)

    @property
    def min_ns(self) -> float:
        return min(self.times_ns)

    @property
    def mean_ns(self) -> float:
        return statistics.fmean(self.times_ns)

    @property
    def mad_ns(self) -> float:
        """Median absolute deviation -- robust dispersion for small samples."""
        med = self.median_ns
        return statistics.median([abs(t - med) for t in self.times_ns])

    @property
    def rel_iqr(self) -> float:
        """(Q3 - Q1) / median; a scale-free dispersion measure."""
        if len(self.times_ns) < 4:
            return float("nan")
        ordered = sorted(self.times_ns)
        q1 = statistics.quantiles(ordered, n=4, method="inclusive")[0]
        q3 = statistics.quantiles(ordered, n=4, method="inclusive")[2]
        return (q3 - q1) / self.median_ns

    @property
    def rel_mad(self) -> float:
        return self.mad_ns / self.median_ns if self.median_ns else float("nan")

    @property
    def gib_per_s(self) -> float | None:
        if not self.bytes_per_second:
            return None
        return statistics.median(self.bytes_per_second) / (1024**3)

    @property
    def gflops(self) -> float | None:
        if not self.gflops_per_sec:
            return None
        return statistics.median(self.gflops_per_sec)

    @property
    def size(self) -> float | None:
        for key in ("n", "edge", "items"):
            if key in self.fields:
                try:
                    return float(self.fields[key])
                except ValueError:
                    return None
        return None

    @property
    def shape(self) -> tuple[str, ...]:
        """Identity of the workload + tuning, excluding variant and thread count."""
        return (
            self.family,
            self.token,
            self.fields.get("dtype", ""),
            self.fields.get("n", self.fields.get("edge", self.fields.get("items", ""))),
            self.fields.get("block", ""),
            self.fields.get("chunk", ""),
        )

    @property
    def group(self) -> tuple[str, ...]:
        """Identity of the *algorithm*; used to pick the T(1) baseline."""
        return self.shape + (self.fields.get("variant", ""),)

    @property
    def identity(self) -> tuple[str, ...]:
        return self.group + (self.fields.get("threads", ""),)


@dataclass
class Environment:
    label: str
    meta: dict[str, Any]

    @property
    def fingerprint(self) -> tuple[str, ...]:
        build = self.meta.get("build", {})
        cpu = self.meta.get("cpu", {})
        return (
            cpu.get("model", "?"),
            cpu.get("logical_cores", "?"),
            build.get("compiler_id", "?") + " " + build.get("compiler_version", "?"),
            build.get("build_type", "?"),
            build.get("cxx_flags", "?"),
            build.get("simd_level", "?"),
        )


@dataclass
class Analysis:
    measurements: list[Measurement] = field(default_factory=list)
    environments: dict[str, Environment] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_name(name: str) -> tuple[str, str, dict[str, str]]:
    """`PCL/reduce_sum/f64/n=1024/simd/T=1` -> ('PCL', 'reduce_sum', {...})."""
    parts = name.split("/")
    family = parts[0] if parts else ""
    tokens = [p for p in parts[1:] if "=" not in p]
    token = tokens[0] if tokens else ""
    fields: dict[str, str] = {}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    return family, token, fields


def _float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def load(raw_path: Path, meta_path: Path | None) -> Analysis:
    analysis = Analysis()
    try:
        data = json.loads(raw_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        analysis.problems.append(f"{raw_path}: cannot read results ({exc})")
        return analysis

    meta: dict[str, Any] = {}
    if meta_path is not None and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError as exc:
            analysis.problems.append(f"{meta_path}: invalid JSON ({exc})")
            meta = {}
    else:
        analysis.problems.append(
            f"{raw_path}: no metadata sidecar -- provenance of these numbers is unknown"
        )

    env_label = raw_path.stem
    if meta:
        version = meta.get("schema_version")
        if version != SCHEMA_VERSION:
            analysis.problems.append(
                f"{raw_path}: schema_version {version!r} != supported {SCHEMA_VERSION!r}"
            )
        env = Environment(env_label, meta)
        fingerprint = env.fingerprint
        env_label = f"{meta.get('cpu', {}).get('model', '?')} / {meta.get('build', {}).get('compiler_id', '?')} {meta.get('build', {}).get('compiler_version', '?')} / {meta.get('build', {}).get('build_type', '?')}"
        analysis.environments[env_label] = env
        analysis.problems.extend(_environment_problems(env_label, fingerprint))

    # Group raw entries by run_name so repetitions become one Measurement.
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in data.get("benchmarks", []):
        if entry.get("run_type") == "aggregate" and entry.get("aggregate_name"):
            continue
        buckets[entry.get("run_name") or entry.get("name", "")].append(entry)

    for name, entries in buckets.items():
        times = [
            t
            for t in (_float(e.get("real_time")) or _float(e.get("cpu_time")) for e in entries)
            if t is not None and t > 0
        ]
        if not times:
            analysis.problems.append(f"{raw_path}: {name} has no usable timing sample")
            continue

        first = entries[0]
        family, token, fields = parse_name(name)
        label = (first.get("label") or "")
        schedule = _label_field(label, "schedule", "")
        validated_values = [_float(e.get("validated")) for e in entries]
        validated = all(v is not None and v != 0.0 for v in validated_values)

        analysis.measurements.append(
            Measurement(
                name=name,
                family=family,
                token=token,
                fields=fields,
                times_ns=times,
                entries=len(entries),
                elements=_float(first.get("elements")),
                bytes_per_iter=_float(first.get("bytes_per_iter")),
                threads=_float(fields.get("threads", first.get("threads", 1))) or 1.0,
                schedule=schedule or fields.get("schedule", ""),
                chunk=_float(fields.get("chunk")),
                block=_float(fields.get("block")),
                validated=validated,
                gflops_per_sec=[
                    g for g in (_float(e.get("gflops_per_sec")) for e in entries) if g is not None
                ],
                bytes_per_second=[
                    b
                    for b in (_float(e.get("bytes_per_second")) for e in entries)
                    if b is not None
                ],
                repetitions=len(times),
                source=str(raw_path),
                environment=env_label,
            )
        )
    return analysis


def _label_field(label: str, key: str, default: str) -> str:
    for part in label.split(","):
        part = part.strip()
        if part.startswith(key + "="):
            return part.split("=", 1)[1]
    return default


def _environment_problems(label: str, fingerprint: tuple[str, ...]) -> list[str]:
    problems: list[str] = []
    model, cores, compiler, build_type, flags, simd = fingerprint
    if "n/a" in (model, ""):
        problems.append(f"{label}: CPU model could not be determined")
    if cores in ("?", 0, "0"):
        problems.append(f"{label}: logical core count is unknown")
    if flags in ("?", ""):
        problems.append(f"{label}: compiler flags were not recorded")
    return problems


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------
def baselines_by_group(measurements: Iterable[Measurement]) -> dict[tuple[str, ...], Measurement]:
    """T(1) of the same algorithm, used as the only valid speedup denominator."""
    best: dict[tuple[str, ...], Measurement] = {}
    for m in measurements:
        if m.threads != 1:
            continue
        current = best.get(m.group)
        # Prefer the explicit `reference` variant when several single-threaded
        # variants exist for the same shape; otherwise take the fastest, which
        # makes speedups against it conservative.
        if current is None:
            best[m.group] = m
            continue
        if m.fields.get("variant") == "reference" and current.fields.get("variant") != "reference":
            best[m.group] = m
        elif current.median_ns > m.median_ns:
            best[m.group] = m
    return best


def reference_by_group(measurements: Iterable[Measurement]) -> dict[tuple[str, ...], Measurement]:
    """The `reference` variant at T(1), used for variant-to-variant comparisons."""
    out: dict[tuple[str, ...], Measurement] = {}
    for m in measurements:
        if m.threads != 1:
            continue
        if m.fields.get("variant") != "reference":
            continue
        # The reference and the variants must share the workload shape but not
        # the variant name.
        out[m.shape] = m
    return out


def enrich(measurements: list[Measurement]) -> None:
    single = baselines_by_group(measurements)
    refs = reference_by_group(measurements)

    for m in measurements:
        m.fields.setdefault("variant", m.fields.get("variant", m.token))
        # Variant name lives in the segment after the size/dtype for kernels and
        # in `token` for gemm/blas style names.
        if "variant" not in m.fields:
            parts = m.name.split("/")
            for part in parts:
                if "=" not in part and part not in ("f32", "f64", "PCL", m.token):
                    m.fields["variant"] = part
                    break
            else:
                m.fields["variant"] = m.token

    for m in measurements:
        base = single.get(m.group)
        if base is None:
            m.fields["speedup"] = ""
            m.fields["efficiency"] = ""
            continue
        if base is m:
            m.fields["speedup"] = "1.000"
            m.fields["efficiency"] = "1.000"
            continue
        speedup = base.median_ns / m.median_ns
        m.fields["speedup"] = f"{speedup:.4f}"
        # Efficiency is only meaningful when more than one thread participates;
        # a single-threaded variant gets an efficiency of 1 by definition.
        m.fields["efficiency"] = f"{speedup / m.threads:.4f}" if m.threads > 1 else "1.000"

    for m in measurements:
        key = m.shape
        ref = refs.get(key)
        if ref is None or ref is m:
            m.fields["variant_speedup"] = ""
            continue
        m.fields["variant_speedup"] = f"{ref.median_ns / m.median_ns:.4f}"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
CSV_FIELDS = [
    "source",
    "environment",
    "name",
    "kernel",
    "dtype",
    "variant",
    "size",
    "threads",
    "schedule",
    "chunk",
    "block",
    "repetitions",
    "median_ns",
    "min_ns",
    "mean_ns",
    "mad_ns",
    "rel_mad",
    "rel_iqr",
    "speedup",
    "efficiency",
    "variant_speedup",
    "gib_per_s",
    "gflops_per_s",
    "validated",
    "elements",
    "bytes_per_iter",
]


def measurement_row(m: Measurement) -> dict[str, Any]:
    return {
        "source": m.source,
        "environment": m.environment,
        "name": m.name,
        "kernel": m.token,
        "dtype": m.fields.get("dtype", ""),
        "variant": m.fields.get("variant", ""),
        "size": m.fields.get("n", m.fields.get("edge", m.fields.get("items", ""))),
        "threads": int(m.threads),
        "schedule": m.schedule,
        "chunk": "" if m.chunk is None else int(m.chunk),
        "block": "" if m.block is None else int(m.block),
        "repetitions": m.repetitions,
        "median_ns": f"{m.median_ns:.4f}",
        "min_ns": f"{m.min_ns:.4f}",
        "mean_ns": f"{m.mean_ns:.4f}",
        "mad_ns": f"{m.mad_ns:.4f}",
        "rel_mad": f"{m.rel_mad:.6f}",
        "rel_iqr": "" if math.isnan(m.rel_iqr) else f"{m.rel_iqr:.6f}",
        "speedup": m.fields.get("speedup", ""),
        "efficiency": m.fields.get("efficiency", ""),
        "variant_speedup": m.fields.get("variant_speedup", ""),
        "gib_per_s": "" if m.gib_per_s is None else f"{m.gib_per_s:.4f}",
        "gflops_per_s": "" if m.gflops is None else f"{m.gflops:.4f}",
        "validated": "yes" if m.validated else "no",
        "elements": "" if m.elements is None else int(m.elements),
        "bytes_per_iter": "" if m.bytes_per_iter is None else int(m.bytes_per_iter),
    }


def write_csv(path: Path, measurements: list[Measurement]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for m in sorted(measurements, key=lambda x: x.name):
            writer.writerow(measurement_row(m))


def missing_configurations(measurements: list[Measurement]) -> list[str]:
    """Report combinations that would be needed for a complete comparison."""
    notes: list[str] = []
    by_kernel: dict[tuple[str, str, str], list[Measurement]] = defaultdict(list)
    for m in measurements:
        by_kernel[(m.family, m.token, m.fields.get("n", m.fields.get("edge", "")))].append(m)

    for (family, token, size), group in sorted(by_kernel.items()):
        variants = {m.fields.get("variant", "") for m in group}
        threads = {m.threads for m in group}
        shape = f"{family}/{token} size={size}"
        if "reference" not in variants:
            notes.append(f"{shape}: no `reference` baseline, variant speedups cannot be computed")
        if 1.0 not in threads:
            notes.append(f"{shape}: no T=1 measurement, thread scaling cannot be computed")
        if len(threads) > 1 and len(threads) < len({t for t in threads if t > 1}):
            notes.append(f"{shape}: thread counts present: {sorted(int(t) for t in threads)}")
    return notes


def scaling_table(measurements: list[Measurement]) -> list[tuple[str, str, int, float, float, float, int]]:
    rows = []
    for m in sorted(measurements, key=lambda x: (x.token, x.name)):
        if m.threads <= 1:
            continue
        speedup = float(m.fields.get("speedup") or "nan")
        rows.append(
            (
                m.environment,
                m.name,
                int(m.threads),
                m.median_ns,
                speedup,
                speedup / m.threads if m.threads else float("nan"),
                m.repetitions,
            )
        )
    return rows


def variant_table(measurements: list[Measurement]) -> list[Measurement]:
    out = []
    for m in measurements:
        if m.threads != 1:
            continue
        if not m.fields.get("variant_speedup"):
            continue
        out.append(m)
    return sorted(out, key=lambda x: (x.token, x.name))


def _plot_scaling(measurements: list[Measurement], out: Path) -> list[Path]:
    if not HAVE_MPL:
        return []
    groups: dict[tuple[str, str], list[Measurement]] = defaultdict(list)
    for m in measurements:
        groups[(m.token, m.fields.get("variant", ""))].append(m)
    written: list[Path] = []
    out.mkdir(parents=True, exist_ok=True)
    for (token, variant), group in sorted(groups.items()):
        by_size: dict[float, list[Measurement]] = defaultdict(list)
        for m in group:
            if m.size is not None:
                by_size[m.size].append(m)
        if len(by_size) < 1 or max((max(x.threads for x in v) for v in by_size.values()), default=1) < 2:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for size, items in sorted(by_size.items()):
            items = sorted(items, key=lambda x: x.threads)
            xs = [m.threads for m in items]
            ys = [m.median_ns for m in items]
            ax.plot(xs, ys, marker="o", label=f"n={int(size)}")
        ax.set_xlabel("threads")
        ax.set_ylabel("median time (ns)")
        ax.set_title(f"{token} / {variant}: thread scaling")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        safe = f"{token}_{variant}".replace("/", "_")
        path = out / f"scaling_{safe}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written


def _plot_sizes(measurements: list[Measurement], out: Path) -> list[Path]:
    if not HAVE_MPL:
        return []
    groups: dict[tuple[str, str, int], list[Measurement]] = defaultdict(list)
    for m in measurements:
        if m.size is None or m.threads != 1:
            continue
        groups[(m.token, m.fields.get("variant", ""), int(m.threads))].append(m)
    written: list[Path] = []
    out.mkdir(parents=True, exist_ok=True)
    for (token, variant, threads), group in sorted(groups.items()):
        by_size: dict[float, list[Measurement]] = defaultdict(list)
        for m in group:
            by_size[float(m.size)].append(m)
        if len(by_size) < 2:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for size, items in sorted(by_size.items()):
            items = sorted(items, key=lambda x: x.median_ns)
            ax.plot([float(size) for _ in items], [m.median_ns for m in items], marker="o",
                    label=f"n={int(size)}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("elements n")
        ax.set_ylabel("median time (ns)")
        ax.set_title(f"{token} / {variant} T={threads}: workload size")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        safe = f"{token}_{variant}_T{threads}".replace("/", "_")
        path = out / f"size_{safe}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written


def write_report(
    path: Path,
    analysis: Analysis,
    measurements: list[Measurement],
    csv_path: Path,
    plot_paths: list[Path],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("# ParallelComputeLab benchmark analysis\n\n")
        handle.write("All numbers below are derived from the raw result files listed in "
                     "`Measurements analysed`; none are synthesised.\n\n")

        handle.write("## Environments\n\n")
        if not analysis.environments:
            handle.write("_No metadata sidecar was found, so the provenance of these results is unknown._\n\n")
        for label, env in sorted(analysis.environments.items()):
            build = env.meta.get("build", {})
            cpu = env.meta.get("cpu", {})
            repo = env.meta.get("repo", {})
            handle.write(f"### {label}\n\n")
            handle.write(f"- source file: `{env.meta.get('raw_results', '?')}`\n")
            handle.write(f"- compiler: {build.get('compiler_id', '?')} {build.get('compiler_version', '?')}\n")
            handle.write(f"- build type: {build.get('build_type', '?')}, "
                         f"C++{build.get('cxx_standard', '?')}\n")
            handle.write(f"- flags: `{build.get('cxx_flags', '?')}`\n")
            handle.write(f"- SIMD level: {build.get('simd_level', '?')}, "
                         f"OpenMP: {build.get('openmp', '?')}\n")
            handle.write(f"- CPU: {cpu.get('model', '?')} "
                         f"({cpu.get('logical_cores', '?')} logical / {cpu.get('physical_cores', '?')} physical), "
                         f"supported SIMD: {cpu.get('supported_simd', '?')}\n")
            handle.write(f"- caches: L1d {cpu.get('l1d_bytes', 0)} B, "
                         f"L2 {cpu.get('l2_bytes', 0)} B, L3 {cpu.get('l3_bytes', 0)} B\n")
            handle.write(f"- OS: {env.meta.get('os', {}).get('system', '?')} "
                         f"{env.meta.get('os', {}).get('release', '?')}\n")
            handle.write(f"- revision: {repo.get('revision', '?')}"
                         f"{' (dirty)' if repo.get('dirty') else ''}\n")
            handle.write(f"- seed: {env.meta.get('seed', '?')}\n\n")

        if len(analysis.environments) > 1:
            handle.write("> **Warning**: results from more than one environment are present. "
                         "Speedups are only computed within an environment; cross-environment "
                         "numbers must not be compared directly.\n\n")

        unvalidated = [m for m in measurements if not m.validated]
        handle.write("## Validation\n\n")
        if unvalidated:
            handle.write("The following entries reported `validated=0` and their timings are "
                         "**not** trustworthy:\n\n")
            for m in unvalidated:
                handle.write(f"- `{m.name}`\n")
            handle.write("\n")
        else:
            handle.write("Every analysed entry passed its pre-timing correctness gate.\n\n")

        handle.write("## Thread scaling\n\n")
        handle.write("`speedup(p) = T(1) / T(p)` and `efficiency(p) = speedup(p) / p`, "
                     "where `T(1)` is the same algorithm with one thread.\n\n")
        rows = scaling_table(measurements)
        if rows:
            handle.write("| measurement | threads | median ns | speedup | efficiency | reps |\n")
            handle.write("|---|---:|---:|---:|---:|---:|\n")
            for env, name, threads, median, speedup, eff, reps in rows:
                handle.write(
                    f"| `{name}` | {threads} | {median:.0f} | {speedup:.3f} | {eff:.3f} | {reps} |\n"
                )
            handle.write("\n")
        else:
            handle.write("_No multi-threaded measurements were found._\n\n")

        handle.write("## Implementation comparison (single threaded, vs `reference`)\n\n")
        variants = variant_table(measurements)
        if variants:
            handle.write("| measurement | median ns | vs reference | dispersion (MAD/median) | "
                         "GiB/s | GFLOP/s |\n")
            handle.write("|---|---:|---:|---:|---:|---:|\n")
            for m in variants:
                gib = "-" if m.gib_per_s is None else f"{m.gib_per_s:.2f}"
                gf = "-" if m.gflops is None else f"{m.gflops:.2f}"
                handle.write(
                    f"| `{m.name}` | {m.median_ns:.0f} | {m.fields['variant_speedup']} | "
                    f"{m.rel_mad:.4f} | {gib} | {gf} |\n"
                )
            handle.write("\n")
        else:
            handle.write("_No comparable single-threaded variants were found._\n\n")

        gaps = missing_configurations(measurements)
        handle.write("## Missing configurations\n\n")
        if gaps:
            for note in gaps:
                handle.write(f"- {note}\n")
        else:
            handle.write("_None detected._")
        handle.write("\n\n")

        handle.write("## Problems found while reading the data\n\n")
        if analysis.problems:
            for problem in sorted(set(analysis.problems)):
                handle.write(f"- {problem}\n")
        else:
            handle.write("_None._")
        handle.write("\n\n")

        handle.write("## Artefacts\n\n")
        handle.write(f"- per-measurement CSV: `{csv_path}`\n")
        for plot in plot_paths:
            handle.write(f"- plot: `{plot}`\n")
        handle.write("\n")

        handle.write("## Measurements analysed\n\n")
        for source in sorted({m.source for m in measurements}):
            handle.write(f"- `{source}`\n")
        handle.write("\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", type=Path, help="raw benchmark JSON files")
    parser.add_argument("--meta", type=Path, action="append", default=[],
                        help="metadata sidecar for the corresponding raw file (repeatable)")
    parser.add_argument("--outdir", type=Path, default=Path("results/analysis"))
    parser.add_argument("--allow-mixed-environments", action="store_true",
                        help="analyse even when the inputs come from different CPUs/builds")
    args = parser.parse_args(argv)

    metas: list[Path | None] = list(args.meta)
    if metas and len(metas) != len(args.results):
        print("error: --meta must be given once per results file", file=sys.stderr)
        return 2

    analysis = Analysis()
    measurements: list[Measurement] = []
    for index, raw in enumerate(args.results):
        meta = metas[index] if metas else raw.with_suffix("").with_suffix(".meta.json")
        if not meta.is_file():
            meta = Path(str(raw).replace(".json", ".meta.json"))
        loaded = load(raw, meta)
        analysis.problems.extend(loaded.problems)
        for label, env in loaded.environments.items():
            analysis.environments[label] = env
        measurements.extend(loaded.measurements)

    if not measurements:
        print("error: no usable measurements found", file=sys.stderr)
        for problem in analysis.problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    environments = {m.environment for m in measurements}
    if len(environments) > 1 and not args.allow_mixed_environments:
        print(
            "warning: results come from more than one environment "
            f"({sorted(environments)}); speedups are computed within each environment only.",
            file=sys.stderr,
        )

    enrich(measurements)

    outdir = args.outdir
    csv_path = outdir / "measurements.csv"
    write_csv(csv_path, measurements)
    plots = _plot_scaling(measurements, outdir / "plots") + _plot_sizes(measurements, outdir / "plots")
    report_path = outdir / "report.md"
    write_report(report_path, analysis, measurements, csv_path, plots)

    print(f"measurements: {len(measurements)}")
    print(f"environments: {len(environments)}")
    print(f"csv:    {csv_path}")
    print(f"report: {report_path}")
    for plot in plots:
        print(f"plot:   {plot}")
    if not HAVE_MPL:
        print("note: matplotlib is not installed, plots were skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
