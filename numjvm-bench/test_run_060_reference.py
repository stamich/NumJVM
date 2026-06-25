import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('run_060_reference.py')


class ParallelRunnerTest(unittest.TestCase):
    def test_dry_run_has_explicit_workers_and_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'results'
            run = subprocess.run([sys.executable, str(SCRIPT), '--mode', 'simd', '--workers', '1,2',
                                  '--output', str(output), '--dry-run'], text=True, capture_output=True)
            self.assertEqual(0, run.returncode, run.stderr)
            self.assertIn('-p workers=1,2', run.stdout)
            self.assertIn('-PsimdBench', run.stdout)
            self.assertFalse(output.exists())

    def test_rejects_invalid_worker_counts_and_paths(self):
        for workers in ('0', '257', 'two'):
            run = subprocess.run([sys.executable, str(SCRIPT), '--mode', 'scalar', '--workers', workers,
                                  '--output', '/tmp/example', '--dry-run'], text=True, capture_output=True)
            self.assertNotEqual(0, run.returncode)
        run = subprocess.run([sys.executable, str(SCRIPT), '--mode', 'scalar', '--output', '/tmp/a space',
                              '--dry-run'], text=True, capture_output=True)
        self.assertNotEqual(0, run.returncode)


if __name__ == '__main__':
    unittest.main()
