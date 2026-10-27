"""S5 - NumPy baseline for the spike benchmarks.

Times the same operations and sizes as SpikeBenchmarks.java with ``timeit``:
best-of-5 rounds, each round long enough to amortise Python call overhead.
Run single-threaded so the comparison with the single-threaded Java loops is fair:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 numpy_baseline.py
"""

import math
import platform
import timeit

import numpy as np

SIZES = [1024, 65536, 1048576]


def bench(stmt, number):
    """Microseconds per call, best of five rounds."""
    best = min(timeit.repeat(stmt, number=number, repeat=5))
    return best / number * 1e6


def main():
    print(f"numpy {np.__version__}, python {platform.python_version()}, {platform.machine()}")
    print(f"{'n':>9} {'op':<16} {'us/op':>10}")
    for n in SIZES:
        side = int(math.isqrt(n))
        a = np.arange(n, dtype=np.float64).reshape(side, n // side)
        b = np.arange(n, dtype=np.float64).reshape(side, n // side)
        bt = np.arange(n, dtype=np.float64).reshape(n // side, side).T
        number = max(10, 20_000_000 // n)
        rows = [
            ("zeros", lambda: np.zeros(n)),
            ("add contiguous", lambda: np.add(a, b)),
            ("add transposed", lambda: np.add(a, bt)),
            ("sum", lambda: np.sum(a)),
        ]
        for name, fn in rows:
            print(f"{n:>9} {name:<16} {bench(fn, number):>10.3f}")


if __name__ == "__main__":
    main()
