import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('compare_expression', Path(__file__).with_name('compare_expression.py'))
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def manifest(revision='abc'):
    return {'source_revision': revision, 'git_dirty': False, 'jmh_exit_code': 0,
            'provider_probe_exit_code': 0, 'system': 'Linux',
            'cpu_lscpu': 'Architecture: x86_64\nModel name: Example\nCPU(s): 16',
            'memory_meminfo': 'MemTotal: 1024 kB', 'java_version': 'JDK 25',
            'java_tool_options': '-Dnumjvm.simd=required', 'configured_mode': 'simd',
            'actual_provider_probe': {'NUMJVM_PROVIDER': 'jdk-vector'},
            'thread_settings': {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
            'smt_active': '1', 'cpu0_governor': 'performance'}


def row(name='eagerOut', score=10):
    return {'benchmark': 'bench.' + name, 'params': {'size': '256'}, 'mode': 'avgt',
            'threads': 1, 'jdkVersion': '25', 'vmName': 'VM', 'vmVersion': '25',
            'jvmArgs': [], 'forks': 3, 'warmupIterations': 3, 'measurementIterations': 5,
            'warmupTime': '1 s', 'measurementTime': '1 s',
            'primaryMetric': {'score': score, 'scoreError': .5, 'scoreUnit': 'us/op', 'rawData': [[score]]}}


class CompareTest(unittest.TestCase):
    def test_matching_out_cases_produce_ratio_and_preserve_samples(self):
        result = comparison.compare([row()], [row('autoOut', 5)], manifest(), manifest('def'), 'eagerOut', 'autoOut')
        self.assertEqual(2, result[0]['speedup'])
        self.assertEqual([[5]], result[0]['candidate']['primaryMetric']['rawData'])

    def test_allocating_and_out_never_form_ratio(self):
        result = comparison.compare([row('eagerAllocating')], [row('autoOut', 5)], manifest(), manifest(), 'eagerAllocating', 'autoOut')
        self.assertIsNone(result[0]['speedup'])
        self.assertIn('workload/output policy differs', result[0]['issues'])

    def test_missing_provider_and_dirty_tree_block_ratio(self):
        candidate = manifest()
        candidate['actual_provider_probe'] = None
        candidate['git_dirty'] = True
        result = comparison.compare([row()], [row('autoOut')], manifest(), candidate, 'eagerOut', 'autoOut')
        self.assertIsNone(result[0]['speedup'])
        self.assertIn('provider probe missing', result[0]['issues'])

    def test_mismatched_units_and_absent_cases_are_reported(self):
        candidate = row('autoOut')
        candidate['primaryMetric']['scoreUnit'] = 'ms/op'
        result = comparison.compare([row()], [candidate], manifest(), manifest(), 'eagerOut', 'autoOut')
        self.assertIn('score unit differs', result[0]['issues'])
        self.assertIsNone(result[0]['speedup'])
        absent = comparison.compare([row()], [], manifest(), manifest(), 'eagerOut', 'autoOut')
        self.assertIn('case missing from baseline or candidate', absent[0]['issues'])


if __name__ == '__main__':
    unittest.main()
