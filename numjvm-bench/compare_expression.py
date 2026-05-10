#!/usr/bin/env python3
"""Compare JMH cases only when their recorded environments are comparable."""
import argparse
import json
import math
from pathlib import Path


def _value(value):
    return 'unavailable' if value is None or not isinstance(value, (int, float)) or not math.isfinite(value) else f'{value:.4g}'


def _cpu(manifest):
    text = manifest.get('cpu_lscpu')
    if not text:
        return None
    wanted = {'Architecture', 'CPU(s)', 'Model name', 'Thread(s) per core', 'Core(s) per socket', 'Socket(s)'}
    values = {}
    for line in text.splitlines():
        key, separator, value = line.partition(':')
        if separator and key.strip() in wanted:
            values[key.strip()] = value.strip()
    return values if 'Model name' in values else None


def _memory(manifest):
    for line in (manifest.get('memory_meminfo') or '').splitlines():
        if line.startswith('MemTotal:'):
            return line.split(':', 1)[1].strip()
    return None


def environment_issues(before, after):
    issues = []
    for label, left, right in [
        ('OS', before.get('system'), after.get('system')),
        ('CPU', _cpu(before), _cpu(after)),
        ('RAM', _memory(before), _memory(after)),
        ('JVM version', before.get('java_version'), after.get('java_version')),
        ('JVM options', before.get('java_tool_options'), after.get('java_tool_options')),
        ('SIMD mode', before.get('configured_mode'), after.get('configured_mode')),
        ('provider probe', before.get('actual_provider_probe'), after.get('actual_provider_probe')),
        ('thread settings', before.get('thread_settings'), after.get('thread_settings')),
        ('SMT', before.get('smt_active'), after.get('smt_active')),
        ('CPU governor', before.get('cpu0_governor'), after.get('cpu0_governor')),
    ]:
        if left is None or right is None or left == {} or right == {}:
            issues.append(f'{label} missing')
        elif left != right:
            issues.append(f'{label} differs')
        elif label == 'thread settings' and any(value is None for value in left.values()):
            issues.append('thread settings contain unknown values')
    for label, manifest in [('baseline', before), ('candidate', after)]:
        if not manifest.get('source_revision'):
            issues.append(f'{label} revision missing')
        if manifest.get('git_dirty') is not False:
            issues.append(f'{label} tree dirty or unknown')
        if manifest.get('jmh_exit_code') != 0:
            issues.append(f'{label} JMH run not recorded as successful')
        if manifest.get('provider_probe_exit_code') != 0:
            issues.append(f'{label} provider probe not recorded as successful')
    return issues


def _index(rows, name):
    result = {}
    for row in rows:
        if name and row.get('benchmark', '').split('.')[-1] != name:
            continue
        identity = None if name else row.get('benchmark')
        key = (identity, tuple(sorted((row.get('params') or {}).items())))
        if key in result:
            raise ValueError(f'duplicate case: {key}')
        result[key] = row
    return result


def _policy(row):
    method = row.get('benchmark', '').split('.')[-1].lower()
    if method.endswith('allocating'):
        return 'allocating'
    if method.endswith('out'):
        return 'preallocated output'
    return method


def _case_issues(a, b):
    issues = []
    if a.get('benchmark', '').rpartition('.')[0] != b.get('benchmark', '').rpartition('.')[0]:
        issues.append('benchmark class differs')
    if _policy(a) != _policy(b):
        issues.append('workload/output policy differs')
    for key in ('mode', 'threads', 'jdkVersion', 'vmName', 'vmVersion', 'jvmArgs',
                'forks', 'warmupIterations', 'measurementIterations', 'warmupTime', 'measurementTime'):
        if a.get(key) is None or b.get(key) is None:
            issues.append(f'{key} missing')
        elif a.get(key) != b.get(key):
            issues.append(f'{key} differs')
    if a.get('mode') not in ('avgt', 'ss', 'thrpt'):
        issues.append('unsupported JMH mode')
    if not a.get('primaryMetric', {}).get('scoreUnit') or not b.get('primaryMetric', {}).get('scoreUnit'):
        issues.append('score unit missing')
    elif a.get('primaryMetric', {}).get('scoreUnit') != b.get('primaryMetric', {}).get('scoreUnit'):
        issues.append('score unit differs')
    for row in (a, b):
        if row.get('forks', 0) < 1 or row.get('measurementIterations', 0) < 1:
            issues.append('fork or measurement count missing')
    return issues


def compare(before_rows, after_rows, before_manifest, after_manifest, before_name=None, after_name=None):
    if bool(before_name) != bool(after_name):
        raise ValueError('provide both benchmark names when comparing different methods')
    env = environment_issues(before_manifest, after_manifest)
    left, right = _index(before_rows, before_name), _index(after_rows, after_name)
    report = []
    for key in sorted(set(left) | set(right), key=str):
        a, b = left.get(key), right.get(key)
        issues = list(env)
        if a is None or b is None:
            issues.append('case missing from baseline or candidate')
        else:
            issues.extend(_case_issues(a, b))
        ratio = None
        if not issues:
            av, bv = a['primaryMetric']['score'], b['primaryMetric']['score']
            if all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (av, bv)):
                ratio = bv / av if a['mode'] == 'thrpt' else av / bv
            else:
                issues.append('score is missing or nonpositive')
        report.append({'case': key, 'baseline': a, 'candidate': b, 'speedup': ratio, 'issues': issues})
    return report


def render(report, before_path, after_path):
    lines = ['# Expression benchmark comparison', '',
             f'Baseline raw JMH: `{before_path}`; candidate raw JMH: `{after_path}`.',
             'Heap allocation values are JMH GC metrics; native array and scratch bytes are not inferred.', '',
             '| Case | Baseline | Candidate | Speedup | Comparability |',
             '| --- | ---: | ---: | ---: | --- |']
    for item in report:
        def metric(row):
            if row is None:
                return 'missing'
            m = row['primaryMetric']
            return f"{_value(m.get('score'))} ± {_value(m.get('scoreError'))} {m.get('scoreUnit', '?')}"
        case = str(item['case']).replace('|', '\\|')
        issues = '; '.join(item['issues']) if item['issues'] else 'matched'
        lines.append(f"| {case} | {metric(item['baseline'])} | {metric(item['candidate'])} | {_value(item['speedup'])} | {issues} |")
    if not report:
        lines.append('| No cases matched the selected benchmark names | — | — | — | No comparison |')
    lines += ['', 'Raw per-fork samples, secondary GC metrics and run configuration remain in the linked JMH JSON files.',
              'A ratio greater than one favors the candidate. Unmatched cases do not receive a ratio.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--baseline-manifest', required=True)
    parser.add_argument('--candidate-manifest', required=True)
    parser.add_argument('--baseline-name')
    parser.add_argument('--candidate-name')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    load = lambda path: json.loads(Path(path).read_text())
    report = compare(load(args.baseline), load(args.candidate), load(args.baseline_manifest),
                     load(args.candidate_manifest), args.baseline_name, args.candidate_name)
    Path(args.output).write_text(render(report, args.baseline, args.candidate))


if __name__ == '__main__':
    main()
