#!/usr/bin/env python3
"""Measure host compilation and executable sizes using isolated downstream crates."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "benches" / "fixtures"
PHASES = (
    "check_clean", "build_clean", "check_rebuild", "build_rebuild", "check_noop", "build_noop"
)


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def output(args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def digest(paths):
    result = hashlib.sha256()
    for path in sorted(paths):
        result.update(path.relative_to(ROOT).as_posix().encode())
        result.update(b"\0")
        result.update(path.read_bytes())
        result.update(b"\0")
    return result.hexdigest()


def source(mode, count):
    template = (FIXTURES / f"{mode}.rs").read_text()
    if mode == "empty":
        return template
    if mode == "static":
        instances = "\n".join(f"instance!(case_{n}, {n});" for n in range(1, count + 1))
        calls = "\n".join(f"    case_{n}();" for n in range(1, count + 1))
        return f"{template}\n{instances}\nfn main() {{\n{calls}\n}}\n"
    return template + f"\nfn main() {{\n    for n in 1..={count} {{\n        instance(black_box(n));\n    }}\n}}\n"


def cargo(phase, manifest, target, host, env):
    command = [
        "cargo", phase.split("_")[0], "--offline", "--jobs=1", "--color=never",
        "--manifest-path", str(manifest), "--target-dir", str(target),
        "--target", host, "--message-format=json",
    ]
    if phase.startswith("build"):
        command.append("--release")
    if phase.endswith("clean"):
        command.append("--timings")
    start = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    if result.returncode:
        raise RuntimeError(f"{' '.join(command)}\n{result.stdout}\n{result.stderr}")
    artifacts = {}
    for line in result.stdout.splitlines():
        message = json.loads(line)
        if message["reason"] == "compiler-artifact":
            artifacts[message["target"]["name"]] = message
        elif message["reason"] == "compiler-message":
            print(message["message"]["rendered"], end="", flush=True)
    expected_fresh = phase.endswith("noop")
    if artifacts["workload"]["fresh"] != expected_fresh:
        raise RuntimeError(f"{phase}: unexpected downstream artifact reuse")
    if artifacts["noodle_cute"]["fresh"] != (not phase.endswith("clean")):
        raise RuntimeError(f"{phase}: unexpected dependency artifact reuse")
    return elapsed, artifacts["workload"].get("executable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=positive, default=3)
    parser.add_argument("--instances", type=positive, nargs="+", default=[1, 8, 32])
    parser.add_argument("--output", type=Path, required=True, help="new results directory")
    args = parser.parse_args()
    if len(set(args.instances)) != len(args.instances):
        parser.error("--instances must be distinct")
    if args.output.exists():
        parser.error("--output must be a new directory")
    args.output.mkdir(parents=True)
    rustc = output(["rustc", "-Vv"])
    host = next(line.split(": ", 1)[1] for line in rustc.splitlines() if line.startswith("host: "))
    env = dict(os.environ)
    env.update(CARGO_INCREMENTAL="0", RUSTC_WRAPPER="", RUSTC_WORKSPACE_WRAPPER="")
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rustc": rustc,
        "cargo": output(["cargo", "-V"]),
        "host": host,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "git_head": output(["git", "rev-parse", "HEAD"]),
        "worktree_dirty": bool(output(["git", "-c", "core.fsmonitor=false", "status", "--porcelain"])),
        "library_source_sha256": digest([
            *ROOT.joinpath("src").rglob("*.rs"), ROOT / "Cargo.toml", ROOT / "rust-toolchain.toml"
        ]),
        "measurement_source_sha256": digest([
            *FIXTURES.glob("*.rs"), Path(__file__).resolve(), ROOT / "benches" / "layout.rs"
        ]),
        "settings": {
            "samples": args.samples, "instances": args.instances, "jobs": 1,
            "incremental": False, "compiler_wrappers": False,
            "release": {"codegen-units": 1, "lto": False, "debug": 0, "strip": "symbols"},
            "environment": {key: env.get(key) for key in (
                "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTC", "RUSTUP_TOOLCHAIN"
            )},
        },
        "cases": [],
    }
    cases = [("empty", 0)] + [
        (mode, count) for mode in ("static", "mixed", "dynamic") for count in args.instances
    ]
    with tempfile.TemporaryDirectory(prefix="noodle-cute-compile-cost-") as directory:
        for mode, count in cases:
            name = f"{mode}-{count}"
            crate = Path(directory) / name
            crate.mkdir()
            shutil.copy(FIXTURES / "typed.rs", crate / "typed.rs")
            program = source(mode, count)
            main_rs = crate / "main.rs"
            manifest = crate / "Cargo.toml"
            manifest.write_text(
                '[package]\nname = "noodle-cute-compile-cost"\nversion = "0.0.0"\n'
                'edition = "2024"\nautobins = false\n[workspace]\n'
                '[[bin]]\nname = "workload"\npath = "main.rs"\n'
                '[dependencies]\n'
                f'noodle-cute = {{ path = {json.dumps(ROOT.as_posix())} }}\n'
                '[profile.dev]\nincremental = false\n'
                '[profile.release]\nincremental = false\ncodegen-units = 1\n'
                'lto = false\ndebug = 0\nstrip = "symbols"\n'
            )
            case = {"mode": mode, "instances": count, "runs": []}
            for sample in range(args.samples):
                print(f"{name}: sample {sample + 1}/{args.samples}", flush=True)
                main_rs.write_text(program)
                target = crate / f"target-{sample}"
                run = {}
                for phase in PHASES:
                    if phase == "check_rebuild":
                        main_rs.write_text(program + f"\n// Rebuild sample {sample}.\n")
                    elapsed, executable = cargo(phase, manifest, target, host, env)
                    run[phase] = elapsed
                    if phase == "build_clean":
                        run["executable_bytes"] = Path(executable).stat().st_size
                        subprocess.run([executable], check=True)
                    if sample == 0 and phase.endswith("clean"):
                        timing = target / "cargo-timings" / "cargo-timing.html"
                        shutil.copy(timing, args.output / f"{name}-{phase}.html")
                case["runs"].append(run)
            case["summary"] = {
                metric: {
                    "median": statistics.median(run[metric] for run in case["runs"]),
                    "min": min(run[metric] for run in case["runs"]),
                    "max": max(run[metric] for run in case["runs"]),
                }
                for metric in (*PHASES, "executable_bytes")
            }
            report["cases"].append(case)
            (args.output / "compile-cost.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# Compilation baseline", "", f"Host: `{host}`. Median of {args.samples} samples; times in seconds.", "",
        "| Mode | Instances | Clean check | Recheck | No-op check | Clean build | Rebuild | No-op build | Executable bytes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in report["cases"]:
        summary = case["summary"]
        values = [summary[phase]["median"] for phase in (
            "check_clean", "check_rebuild", "check_noop", "build_clean", "build_rebuild", "build_noop"
        )]
        lines.append(
            f"| {case['mode']} | {case['instances']} | "
            + " | ".join(f"{value:.3f}" for value in values)
            + f" | {int(summary['executable_bytes']['median'])} |"
        )
    lines += ["", "Raw samples, ranges, settings, and source hashes are in `compile-cost.json`.", ""]
    (args.output / "summary.md").write_text("\n".join(lines))
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
