#!/usr/bin/env python3
"""Record and run a standalone 0.6.0 worker-scaling campaign."""
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def command_output(command):
    try:
        return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f'unavailable: {error}'


def optional_file(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return 'unavailable'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('scalar', 'simd'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', default='1,2,4,8,16')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if any(c.isspace() for c in str(output)):
        parser.error('output path cannot contain whitespace')
    try:
        workers = [int(value) for value in args.workers.split(',')]
        if not workers or any(value < 1 or value > 256 for value in workers): raise ValueError()
    except ValueError:
        parser.error('workers must be comma-separated integers from 1 to 256')
    jmh = ('ParallelScalingBenchmark -p workers=' + ','.join(map(str, workers))
           + ' -t 1 -f 3 -wi 3 -w 1s -i 5 -r 1s -prof gc -rf json -rff ' + str(output / 'results.json'))
    command = ['./gradlew', ':numjvm-bench:jmh'] + (['-PsimdBench'] if args.mode == 'simd' else []) + ['-PjmhArgs=' + jmh]
    flags = '--add-modules=jdk.incubator.vector -Dnumjvm.simd=required' if args.mode == 'simd' else '-Dnumjvm.simd=off'
    print('Command:', shlex.join(command), flush=True)
    print('JAVA_TOOL_OPTIONS additions:', flags, flush=True)
    if args.dry_run: return 0
    if output.exists() and any(output.iterdir()): parser.error('output directory must be empty')
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['JAVA_TOOL_OPTIONS'] = (env.get('JAVA_TOOL_OPTIONS', '') + ' ' + flags).strip()
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('OPENBLAS_NUM_THREADS', '1')
    status = command_output(['git', 'status', '--porcelain'])
    manifest = {'recorded_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'revision': command_output(['git', 'rev-parse', 'HEAD']), 'git_status': status,
                'system': platform.platform(), 'cpu': command_output(['lscpu']),
                'memory': optional_file('/proc/meminfo'),
                'cpu0_governor': optional_file('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'),
                'smt_active': optional_file('/sys/devices/system/cpu/smt/active'),
                'java': command_output(['java', '-version']), 'mode': args.mode, 'workers': workers,
                'jmh_threads': 1, 'java_tool_options': env['JAVA_TOOL_OPTIONS'],
                'thread_settings': {key: env.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
                'command': command, 'exit_code': None}
    with (output / 'jmh.log').open('w') as log:
        run = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    failed = run.returncode != 0 or not (output / 'results.json').is_file() or '<failure>' in (output / 'jmh.log').read_text()
    manifest['exit_code'] = run.returncode or int(failed)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if failed: print('Benchmark failed; inspect', output / 'jmh.log', file=sys.stderr)
    return manifest['exit_code']


if __name__ == '__main__':
    sys.exit(main())
