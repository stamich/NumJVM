"""Generate compact periodic NumPy reference values for parallel operations."""
import argparse
import json
from pathlib import Path
import numpy as np

PATH = Path(__file__).with_name('parallel-reference.json')
N = 65537

def render():
    cases = {}
    for dtype in (np.float32, np.float64):
        x = (np.arange(37, dtype=dtype) + dtype(1)) / dtype(8)
        y = ((np.arange(37, dtype=dtype) * dtype(7)) % dtype(37) + dtype(1)) / dtype(8)
        cases[np.dtype(dtype).name] = {
            'x': x.tolist(), 'y': y.tolist(),
            'add': np.add(x, y).tolist(), 'subtract': np.subtract(x, y).tolist(),
            'multiply': np.multiply(x, y).tolist(), 'divide': np.divide(x, y).tolist(),
            'negative': np.negative(x).tolist(), 'absolute': np.absolute(x).tolist(),
            'square': np.square(x).tolist(), 'sqrt': np.sqrt(x).tolist(),
            'sum': float(np.sum(np.resize(x, N), dtype=dtype)),
        }
    return json.dumps({'size': N, 'period': 37, 'cases': cases}, indent=2, sort_keys=True) + '\n'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    expected = render()
    if args.check:
        if not PATH.exists() or PATH.read_text() != expected:
            raise SystemExit('parallel NumPy fixtures differ; regenerate with pinned NumPy')
        print('parallel NumPy fixtures match')
    else:
        PATH.write_text(expected)

if __name__ == '__main__':
    main()
