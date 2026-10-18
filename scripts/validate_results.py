#!/usr/bin/env python3
"""Validate ParallelComputeLab benchmark result files against the documented
schema (docs/BENCHMARKING.md, schema 1.0.0).

Exit status
-----------
0  every file validated
1  at least one file is missing, malformed or incomplete
2  usage error

This tool never repairs data and never invents values: it reports exactly what
is wrong so a broken measurement cannot quietly become part of an analysis.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0.0"

# Counters every benchmark entry must publish.  Google Benchmark folds simple
# numeric counters to the top level of the entry, which is what we read.
REQUIRED_COUNTERS = ("elements", "bytes_per_iter", "threads", "validated")

REQUIRED_META_SECTIONS = ("schema_version", "build", "cpu", "os", "repo", "measurement")
REQUIRED_META_BUILD = ("compiler_id", "compiler_version", "build_type", "cxx_flags")


@dataclass
class Problem:
    path: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"{self.path}: {self.message}"


@dataclass
class Report:
    problems: list[Problem] = field(default_factory=list)
    warnings: list[Problem] = field(default_factory=list)
    files_checked: int = 0
    entries_checked: int = 0
    unvalidated_entries: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _check_meta(path: Path, meta: Any, report: Report) -> dict[str, Any]:
    if not isinstance(meta, dict):
        report.problems.append(Problem(str(path), "metadata is not a JSON object"))
        return {}

    version = meta.get("schema_version")
    if version != SCHEMA_VERSION:
        report.problems.append(
            Problem(str(path), f"schema_version {version!r} != supported {SCHEMA_VERSION!r}")
        )

    for section in REQUIRED_META_SECTIONS:
        if section not in meta:
            report.problems.append(Problem(str(path), f"metadata is missing section {section!r}"))

    build = meta.get("build")
    if isinstance(build, dict):
        for key in REQUIRED_META_BUILD:
            if not build.get(key):
                report.problems.append(Problem(str(path), f"metadata build.{key} is missing/empty"))
    elif "build" in meta:
        report.problems.append(Problem(str(path), "metadata build is not an object"))

    cpu = meta.get("cpu")
    if isinstance(cpu, dict):
        if not cpu.get("model"):
            report.problems.append(Problem(str(path), "metadata cpu.model is missing"))
        if not cpu.get("logical_cores"):
            report.problems.append(Problem(str(path), "metadata cpu.logical_cores is missing"))
    elif "cpu" in meta:
        report.problems.append(Problem(str(path), "metadata cpu is not an object"))

    if "repo" in meta and isinstance(meta["repo"], dict):
        if not meta["repo"].get("revision"):
            # Not fatal: a source tree unpacked from an archive has no revision.
            # It does mean the results cannot be tied to a specific commit.
            report.warnings.append(
                Problem(str(path), "metadata repo.revision is empty (no commit to trace the run to)")
            )

    if "seed" not in meta:
        report.problems.append(Problem(str(path), "metadata seed is missing"))

    return meta if isinstance(meta, dict) else {}


def _check_entries(path: Path, data: Any, report: Report) -> None:
    if not isinstance(data, dict):
        report.problems.append(Problem(str(path), "raw results are not a JSON object"))
        return
    entries = data.get("benchmarks")
    if not isinstance(entries, list) or not entries:
        report.problems.append(Problem(str(path), "raw results contain no 'benchmarks' array"))
        return

    for entry in entries:
        report.entries_checked += 1
        name = entry.get("run_name", entry.get("name", "<unnamed>"))
        if entry.get("error_occurred"):
            # Google Benchmark records a failed/skipped run this way.  It is a
            # hard problem: the benchmark did not produce a usable measurement.
            report.problems.append(
                Problem(
                    str(path),
                    f"{name}: benchmark reported error_occurred "
                    "(the pre-timing correctness gate rejected this variant)",
                )
            )
            continue
        if entry.get("run_type") not in (None, "iteration", "aggregate"):
            report.problems.append(Problem(str(path), f"{name}: unknown run_type {entry.get('run_type')!r}"))

        timing = entry.get("real_time", entry.get("cpu_time"))
        if not isinstance(timing, (int, float)) or timing <= 0:
            report.problems.append(Problem(str(path), f"{name}: missing or non-positive time"))
            continue

        for counter in REQUIRED_COUNTERS:
            if counter not in entry:
                report.problems.append(Problem(str(path), f"{name}: missing counter {counter!r}"))

        validated = entry.get("validated")
        if validated is None:
            continue
        try:
            is_valid = float(validated) != 0.0
        except (TypeError, ValueError):
            report.problems.append(Problem(str(path), f"{name}: non-numeric 'validated' counter"))
            continue
        if not is_valid:
            report.unvalidated_entries.append(name)


def validate_file(raw_path: Path, meta_path: Path | None, report: Report) -> None:
    report.files_checked += 1
    if not raw_path.is_file():
        report.problems.append(Problem(str(raw_path), "file does not exist"))
        return
    try:
        data = json.loads(raw_path.read_text())
    except json.JSONDecodeError as exc:
        report.problems.append(Problem(str(raw_path), f"invalid JSON: {exc}"))
        return

    _check_entries(raw_path, data, report)

    if meta_path is None:
        report.problems.append(
            Problem(str(raw_path), "no metadata sidecar given; provenance would be unknown")
        )
        return
    if not meta_path.is_file():
        report.problems.append(Problem(str(meta_path), "metadata sidecar does not exist"))
        return
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError as exc:
        report.problems.append(Problem(str(meta_path), f"invalid JSON: {exc}"))
        return
    _check_meta(meta_path, meta, report)

    recorded = meta.get("raw_results")
    if recorded and Path(recorded).name != raw_path.name:
        report.problems.append(
            Problem(
                str(meta_path),
                f"metadata points at a different raw file ({recorded!r}) than the one supplied",
            )
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", type=Path, help="raw benchmark JSON files")
    parser.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="metadata sidecar for a single raw file (repeatable via positional order)",
    )
    parser.add_argument(
        "--no-meta",
        action="store_true",
        help="skip provenance checks (structure only)",
    )
    parser.add_argument("--quiet", action="store_true", help="only print the summary line")
    args = parser.parse_args(argv)

    if args.no_meta:
        report = Report()
        for raw in args.results:
            report.files_checked += 1
            if not raw.is_file():
                report.problems.append(Problem(str(raw), "file does not exist"))
                continue
            try:
                _check_entries(raw, json.loads(raw.read_text()), report)
            except json.JSONDecodeError as exc:
                report.problems.append(Problem(str(raw), f"invalid JSON: {exc}"))
    else:
        if args.meta is not None and len(args.meta) != len(args.results):
            print("error: --meta must be given once per results file", file=sys.stderr)
            return 2
        report = Report()
        for index, raw in enumerate(args.results):
            meta = None if args.meta is None else args.meta[index]
            validate_file(raw, meta, report)

    if not args.quiet:
        for problem in report.warnings:
            print(f"WARN {problem}")
        for problem in report.problems:
            print(f"FAIL {problem}")
        for name in report.unvalidated_entries:
            print(f"WARN {name}: reported validated=0 (skipped or failed validation)")

    print(
        f"checked {report.files_checked} file(s), {report.entries_checked} benchmark entry(ies): "
        f"{'OK' if report.ok else str(len(report.problems)) + ' problem(s)'}"
    )
    if report.warnings or report.unvalidated_entries:
        print(
            f"warning: {len(report.warnings)} provenance warning(s), "
            f"{len(report.unvalidated_entries)} unvalidated entry/entries"
        )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
