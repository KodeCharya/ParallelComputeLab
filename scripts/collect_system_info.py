#!/usr/bin/env python3
"""Collect the machine / toolchain facts that a benchmark result depends on.

Writes a JSON document that can be diffed between two runs, and optionally a
short human readable summary.  This is the Python counterpart of the C++
`system::query_cpu_info()` used to stamp results; running both makes it easy to
confirm that the metadata recorded by the harness is accurate.

    python3 scripts/collect_system_info.py --out results/system_info.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0.0"


def run(cmd: list[str]) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cores": os.cpu_count(),
    }
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        text = ""
    for key in ("model name", "vendor_id", "cpu family", "model", "stepping", "cpu cores",
                "siblings", "cpu MHz"):
        match = re.search(rf"^{re.escape(key)}\s*:\s*(.+)$", text, re.MULTILINE)
        if match:
            info[key.replace(" ", "_")] = match.group(1).strip()

    flags_match = re.search(r"^flags\s*:\s*(.+)$", text, re.MULTILINE)
    if flags_match:
        flags = set(flags_match.group(1).split())
        interesting = ["sse2", "sse4_2", "avx", "avx2", "fma", "avx512f", "avx512bw",
                       "amx_tile", "asimd"]
        info["relevant_isa_flags"] = sorted(f for f in flags if f in interesting)

    info["caches"] = {}
    cache_dir = Path("/sys/devices/system/cpu/cpu0/cache")
    if cache_dir.is_dir():
        for entry in sorted(cache_dir.glob("index*")):
            try:
                level = (entry / "level").read_text().strip()
                kind = (entry / "type").read_text().strip()
                size = (entry / "size").read_text().strip()
            except OSError:
                continue
            info["caches"][f"L{level}{kind[0]}"] = size

    info["max_frequency_khz"] = None
    for path in (
        "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq",
        "/sys/devices/system/cpu/cpufreq/cpuinfo_max_freq",
    ):
        try:
            info["max_frequency_khz"] = int(Path(path).read_text().strip())
            break
        except (OSError, ValueError):
            continue

    # Scaling governor: with 'performance' disabled, timings are noisy.
    try:
        info["scaling_governor"] = (
            Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
        )
    except OSError:
        info["scaling_governor"] = None
    return info


def memory_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(("MemTotal:", "SwapTotal:", "Hugepagesize:")):
                key, value = line.split(":", 1)
                info[key] = value.strip()
    except OSError:
        pass
    return info


def os_info() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "libc": "-".join(x for x in platform.libc_ver() if x),
    }


def toolchain_info() -> dict[str, Any]:
    gcc = run(["g++", "--version"])
    clang = run(["clang++", "--version"])
    cmake = run(["cmake", "--version"])
    perf = run(["perf", "--version"])
    info = {
        "g++": gcc.splitlines()[0] if gcc else None,
        "clang++": clang.splitlines()[0] if clang else None,
        "cmake": cmake.splitlines()[0] if cmake else None,
        "perf": perf.splitlines()[0] if perf else None,
    }
    # perf_event access decides whether hardware counters can be used at all.
    paranoid = Path("/proc/sys/kernel/perf_event_paranoid")
    try:
        info["perf_event_paranoid"] = int(paranoid.read_text().strip())
    except (OSError, ValueError):
        info["perf_event_paranoid"] = None
    info["python"] = sys.version.split()[0]
    return info


def perf_access() -> dict[str, Any]:
    """Can hardware counters actually be collected here?"""
    available = {
        "perf_binary": shutil.which("perf"),
        "perf_event_paranoid": None,
        "usable": False,
    }
    try:
        available["perf_event_paranoid"] = int(
            Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip()
        )
    except (OSError, ValueError):
        pass
    # paranoid <= 1 permits user-space measurement of own processes.
    available["usable"] = bool(
        available["perf_binary"] and available["perf_event_paranoid"] is not None
        and available["perf_event_paranoid"] <= 1
    )
    return available


def git_info(repo: Path) -> dict[str, Any]:
    if not (repo / ".git").exists():
        return {"revision": None, "branch": None, "dirty": None, "note": "not a git checkout"}
    revision = run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"])
    branch = run(["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"])
    status = run(["git", "-C", str(repo), "status", "--porcelain"])
    return {
        "revision": revision,
        "branch": branch,
        "dirty": bool(status),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None, help="write JSON here")
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="repository to describe")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    document = {
        "schema_version": SCHEMA_VERSION,
        "cpu": cpu_info(),
        "memory": memory_info(),
        "os": os_info(),
        "toolchain": toolchain_info(),
        "perf": perf_access(),
        "git": git_info(args.repo),
        "load_average": os.getloadavg(),
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(document, indent=2) + "\n")
        if not args.quiet:
            print(f"system information written to {args.out}")
    elif not args.quiet:
        print(json.dumps(document, indent=2))

    if not args.quiet:
        cpu = document["cpu"]
        print(f"CPU:    {cpu.get('model_name', '?')}")
        print(f"Cores:  {cpu.get('logical_cores')} logical")
        print(f"Caches: {cpu.get('caches')}")
        print(f"ISA:    {', '.join(cpu.get('relevant_isa_flags', [])) or 'unknown'}")
        print(f"Governor: {cpu.get('scaling_governor')}")
        print(f"perf usable: {document['perf']['usable']} "
              f"(paranoid={document['perf']['perf_event_paranoid']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
