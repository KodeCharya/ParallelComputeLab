#!/usr/bin/env python3
"""Run the ParallelComputeLab benchmark suite reproducibly.

Wraps the `pcl_benchmarks` binary so that a run is described entirely by a
label, a preset and an output path, and so that the raw JSON, its metadata
sidecar, the command line and the system information all land in the same
directory.  Afterwards the results are validated; an invalid result set is
reported as such rather than being analysed.

Examples
--------
    # quick look, ~1 minute
    python3 scripts/run_benchmarks.py --build build/gcc --preset quick --label gcc-quick

    # the configuration recorded in configs/benchmark_defaults.json
    python3 scripts/run_benchmarks.py --build build/gcc --preset full --label gcc-release

    # everything, as used to produce results/raw/gcc-release.json
    python3 scripts/run_benchmarks.py --build build/gcc --preset full --label gcc-release \
        --sizes 1048576,16777216 --threads 1,2,4,8
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_CONFIG = ROOT / "configs" / "benchmark_defaults.json"


def load_config(path: Path) -> dict:
    if not path.is_file():
        print(f"error: config {path} not found", file=sys.stderr)
        raise SystemExit(2)
    return json.loads(path.read_text())


def build_command(args, preset: dict, label: str, out_json: Path, log_path: Path) -> list[str]:
    binary = args.binary or (Path(args.build) / "benchmarks" / "pcl_benchmarks")
    if not Path(binary).is_file():
        print(f"error: benchmark binary {binary} not found; build it first", file=sys.stderr)
        raise SystemExit(2)

    cmd = [str(binary)]
    cmd += ["--pcl_sizes=" + ",".join(str(v) for v in args.sizes or preset["sizes"])]
    cmd += ["--pcl_matrix_sizes=" + ",".join(str(v) for v in preset["matrix_sizes"])]
    cmd += ["--pcl_threads=" + ",".join(str(v) for v in args.threads or preset["threads"])]
    if preset.get("blocks"):
        cmd += ["--pcl_blocks=" + ",".join(str(v) for v in preset["blocks"])]
    if preset.get("chunks"):
        cmd += ["--pcl_chunks=" + ",".join(str(v) for v in preset["chunks"])]
    cmd += ["--pcl_label=" + label]
    if args.filter:
        cmd += ["--benchmark_filter=" + args.filter]
    elif preset.get("filter"):
        cmd += ["--benchmark_filter=" + preset["filter"]]
    cmd += ["--benchmark_repetitions=" + str(args.repeat or preset.get("repeat", 5))]
    cmd += ["--benchmark_min_time=" + str(args.min_time or preset.get("min_time", "0.05"))]
    cmd += ["--benchmark_enable_random_interleaving=true"]
    cmd += ["--benchmark_out=" + str(out_json), "--benchmark_out_format=json"]
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", default=str(ROOT / "build" / "gcc"),
                        help="build directory containing benchmarks/pcl_benchmarks")
    parser.add_argument("--binary", default=None, help="explicit path to pcl_benchmarks")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preset", default="full",
                        help="preset name from the config file, or 'custom' to pass everything")
    parser.add_argument("--label", default=None, help="run label (defaults to <preset>-<compiler>)")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results" / "raw")
    parser.add_argument("--sizes", type=int, nargs="*", default=None)
    parser.add_argument("--threads", type=int, nargs="*", default=None)
    parser.add_argument("--filter", default=None)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--min-time", dest="min_time", default=None)
    parser.add_argument("--no-validate", action="store_true",
                        help="skip the post-run schema validation")
    parser.add_argument("--dry-run", action="store_true", help="print the command and exit")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    presets = config.get("filters", {})
    if args.preset != "custom" and args.preset not in presets:
        print(f"error: unknown preset {args.preset!r}; available: {sorted(presets)}",
              file=sys.stderr)
        return 2
    preset = dict(config["workloads"])
    if args.preset != "custom":
        preset.update(presets[args.preset])

    label = args.label or f"{args.preset}-{Path(args.build).name}"
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_json = args.outdir / f"{label}.json"
    log_path = args.outdir / f"{label}.log"

    cmd = build_command(args, preset, label, out_json, log_path)

    # Collect the machine description next to the results so the analysis has a
    # second, independent record of the environment.
    sysinfo_path = args.outdir / f"{label}.system.json"
    subprocess.run(
        [sys.executable, str(HERE / "collect_system_info.py"),
         "--out", str(sysinfo_path), "--repo", str(ROOT), "--quiet"],
        check=False,
    )

    print("command:", " ".join(shlex.quote(c) for c in cmd))
    if args.dry_run:
        return 0

    started = time.time()
    with log_path.open("w") as log:
        log.write("# command: " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")
        log.flush()
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.time() - started
    print(f"exit status {result.returncode} after {elapsed:.1f}s -> {log_path}")

    if result.returncode != 0:
        print(f"error: benchmark run failed; see {log_path}", file=sys.stderr)
        return result.returncode

    meta_json = out_json.with_suffix("").with_suffix(".meta.json")
    if not meta_json.is_file():
        meta_json = Path(str(out_json).replace(".json", ".meta.json"))
    if not meta_json.is_file():
        print(f"error: metadata sidecar missing next to {out_json}", file=sys.stderr)
        return 1

    if not args.no_validate:
        validate = subprocess.run(
            [sys.executable, str(HERE / "validate_results.py"), str(out_json),
             "--meta", str(meta_json)]
        )
        if validate.returncode != 0:
            print("error: result validation failed", file=sys.stderr)
            return validate.returncode

    print(f"raw results:   {out_json}")
    print(f"metadata:      {meta_json}")
    print(f"system info:   {sysinfo_path}")
    print(f"console log:   {log_path}")
    print(f"\nnext: python3 scripts/analyze_results.py {out_json} --meta {meta_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
