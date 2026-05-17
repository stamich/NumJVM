import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('run_050_reference.py')


class ReferenceRunnerTest(unittest.TestCase):
    def test_dry_run_lists_suites_without_creating_results(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'results'
            run = subprocess.run([sys.executable, str(SCRIPT), '--mode', 'simd', '--suite', 'all',
                                  '--output', str(output), '--dry-run'], text=True, capture_output=True)
            self.assertEqual(0, run.returncode, run.stderr)
            self.assertIn('inplace-generated:', run.stdout)
            self.assertIn('-PsimdBench', run.stdout)
            self.assertFalse(output.exists())

    def test_rejects_paths_that_gradle_would_split(self):
        run = subprocess.run([sys.executable, str(SCRIPT), '--mode', 'scalar', '--output', '/tmp/a space'],
                             text=True, capture_output=True)
        self.assertNotEqual(0, run.returncode)
        self.assertIn('cannot contain whitespace', run.stderr)


if __name__ == '__main__':
    unittest.main()
