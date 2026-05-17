#!/usr/bin/env python3
"""Run the expression baseline and save provenance without assuming backend use."""
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def command(args, env=None):
    result = subprocess.run(args, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    return result.returncode, result.stdout.strip()


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new or existing result directory")
    parser.add_argument("--mode", choices=("scalar", "simd"), required=True)
    parser.add_argument("--manifest-only", action="store_true", help="do not execute Gradle")
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    revision_code, revision = command(["git", "rev-parse", "HEAD"])
    status_code, status = command(["git", "status", "--porcelain", "--untracked-files=normal"])
    _, cpu = command(["lscpu"])
    _, java = command([str(Path(os.environ["JAVA_HOME"]) / "bin/java") if "JAVA_HOME" in os.environ else "java", "-version"])
    options = "-Dnumjvm.simd=" + ("required --add-modules=jdk.incubator.vector" if args.mode == "simd" else "off")
    env = os.environ.copy()
    env["JAVA_TOOL_OPTIONS"] = (env.get("JAVA_TOOL_OPTIONS", "") + " " + options).strip()
    simd_arg = ["-PsimdBench"] if args.mode == "simd" else []
    jmh_args = ("ExpressionBenchmark -p size=256,65536,1048576 -f 3 -wi 3 -w 1s "
                "-i 5 -r 1s -prof gc -rf json -rff " + str(out / "m3-expression.json"))
    manifest = {
        "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_revision": revision if revision_code == 0 else None,
        "git_dirty": bool(status) if status_code == 0 else None,
        "system": platform.platform(), "cpu_lscpu": cpu or None,
        "memory_meminfo": read("/proc/meminfo"),
        "smt_active": read("/sys/devices/system/cpu/smt/active"),
        "cpu0_governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "java_version": java or None, "java_tool_options": env["JAVA_TOOL_OPTIONS"],
        "thread_settings": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "jmh_threads": 1, "configured_mode": args.mode,
        "actual_provider_probe": None, "provider_probe_note": "not run",
        "jmh_args": jmh_args, "jmh_exit_code": None,
    }
    manifest_path = out / "manifest.json"
    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    save()
    if args.manifest_only:
        print(manifest_path)
        return 0
    code, probe = command(["./gradlew", "--quiet", ":numjvm-bench:providerProbe", *simd_arg], env)
    (out / "provider-probe.log").write_text(probe + "\n")
    manifest["actual_provider_probe"] = dict(line.split("=", 1) for line in probe.splitlines()
                                               if line.startswith("NUMJVM_") and "=" in line)
    manifest["provider_probe_note"] = "separate process; JMH forks must use the same settings"
    manifest["provider_probe_exit_code"] = code
    save()
    if code != 0:
        print("Provider probe failed; see", out / "provider-probe.log", file=sys.stderr)
        return code
    code, log = command(["./gradlew", ":numjvm-bench:jmh", *simd_arg, "-PjmhArgs=" + jmh_args], env)
    (out / "jmh.log").write_text(log + "\n")
    if code == 0 and ("<failure>" in log or not (out / "m3-expression.json").is_file()):
        code = 1
    manifest["jmh_exit_code"] = code
    save()
    print(manifest_path)
    return code


if __name__ == "__main__":
    sys.exit(main())
