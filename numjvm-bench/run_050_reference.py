#!/usr/bin/env python3
"""Run reproducible 0.5.0 JMH suites; select scalar and SIMD separately."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
SIZES = '256,65536,1048576'
SUITES = {
    'contiguous': f'ExpressionMatrixBenchmark.(eagerOut|autoOut|blockOut|scheduledOut|generatedOut) -p size={SIZES} -p layout=C,F -p types=FLOAT32,FLOAT64,MIXED -p chain=ADD,CHAIN',
    'layouts': f'ExpressionMatrixBenchmark.(eagerOut|autoOut|blockOut|scheduledOut) -p size={SIZES} -p layout=STRIDED,BROADCASTED -p types=FLOAT32,FLOAT64,MIXED -p chain=ADD,CHAIN',
    'inplace': f'ExpressionInPlaceBenchmark -p size={SIZES} -p layout=C,F,STRIDED -p types=FLOAT32,FLOAT64,MIXED -p strategy=EAGER,AUTO,BLOCK',
    'inplace-generated': f'ExpressionInPlaceBenchmark -p size={SIZES} -p layout=C,F -p types=FLOAT32,FLOAT64,MIXED -p strategy=REQUIRED',
    'lifecycle': 'ExpressionLifecycleBenchmark',
    'scratch': 'ExpressionScratchBenchmark',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='result directory, without whitespace')
    parser.add_argument('--mode', required=True, choices=('scalar', 'simd'))
    parser.add_argument('--suite', choices=('baseline', 'all', *SUITES), default='baseline')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if any(character.isspace() for character in str(output)):
        parser.error('output path cannot contain whitespace (Gradle JMH argument parser limitation)')
    flags = '-Dnumjvm.simd=off' if args.mode == 'scalar' else '--add-modules=jdk.incubator.vector -Dnumjvm.simd=required'
    base_env = os.environ.copy()
    env = base_env.copy()
    env['JAVA_TOOL_OPTIONS'] = (env.get('JAVA_TOOL_OPTIONS', '') + ' ' + flags).strip()
    selected = list(SUITES) if args.suite == 'all' else ([] if args.suite == 'baseline' else [args.suite])
    base_dir = output / 'baseline'
    base_cmd = [sys.executable, str(ROOT / 'numjvm-bench/run_m3_reference.py'), '--mode', args.mode, '--output', str(base_dir)]
    vector_arg = ['-PsimdBench'] if args.mode == 'simd' else []
    print('JAVA_TOOL_OPTIONS additions:', flags, flush=True)
    print('Baseline:', shlex.join(base_cmd), flush=True)
    for suite in selected:
        jmh = SUITES[suite] + f' -f 3 -wi 3 -w 1s -i 5 -r 1s -prof gc -rf json -rff {output / suite / "results.json"}'
        cmd = ['./gradlew', ':numjvm-bench:jmh', *vector_arg, '-PjmhArgs=' + jmh]
        print(suite + ':', shlex.join(cmd), flush=True)
    if args.dry_run:
        return 0
    if output.exists() and any(output.iterdir()):
        parser.error('output directory is not empty; use a new directory to preserve prior results')
    output.mkdir(parents=True, exist_ok=True)
    base = subprocess.run(base_cmd, cwd=ROOT, env=base_env, check=False)
    if base.returncode:
        return base.returncode
    baseline_manifest = json.loads((base_dir / 'manifest.json').read_text())
    for suite in selected:
        suite_dir = output / suite
        suite_dir.mkdir(parents=True, exist_ok=True)
        jmh = SUITES[suite] + f' -f 3 -wi 3 -w 1s -i 5 -r 1s -prof gc -rf json -rff {suite_dir / "results.json"}'
        cmd = ['./gradlew', ':numjvm-bench:jmh', *vector_arg, '-PjmhArgs=' + jmh]
        manifest = dict(baseline_manifest)
        manifest.update({'suite': suite, 'jmh_args': jmh, 'jmh_exit_code': None})
        (suite_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        with (suite_dir / 'jmh.log').open('w') as log:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        failed_case = '<failure>' in (suite_dir / 'jmh.log').read_text() or not (suite_dir / 'results.json').is_file()
        status = result.returncode or (1 if failed_case else 0)
        manifest['jmh_exit_code'] = status
        (suite_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        if status:
            print('Benchmark failed; inspect', suite_dir / 'jmh.log', file=sys.stderr)
            return status
    return 0


if __name__ == '__main__':
    sys.exit(main())
