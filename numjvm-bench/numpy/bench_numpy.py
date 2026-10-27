"""NumPy counterparts of the NumJVM M0 JMH benchmarks.

Times each operation with timeit (best of 5 rounds; each round long enough that one
round takes ~0.2 s) and prints CSV: benchmark,layout,size,ns_per_op.

Run single-threaded on the same machine as JMH:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 bench_numpy.py [--sizes 1024,65536]

Element-by-element reads (ContiguousRead/StridedRead) have no meaningful NumPy
counterpart: a Python loop over elements measures the interpreter. They are omitted.
"""

import argparse
import math
import timeit

import numpy as np

SIZES = [16, 128, 1024, 65536, 1048576, 16777216]
LAYOUTS = ["C", "F", "TRANSPOSED", "STRIDED", "BROADCASTED"]


def rows(n):
    return 1 << (int(math.log2(n)) // 2)


def make(n, layout):
    r, c = rows(n), n // rows(n)
    if layout == "C":
        return np.arange(float(n)).reshape(r, c)
    if layout == "F":
        return np.asfortranarray(np.arange(float(n)).reshape(r, c))
    if layout == "TRANSPOSED":
        return np.arange(float(n)).reshape(c, r).T
    if layout == "STRIDED":
        return np.arange(2.0 * n).reshape(r, 2 * c)[:, ::2]
    if layout == "BROADCASTED":
        return np.broadcast_to(np.arange(float(c)), (r, c))
    raise ValueError(layout)


def ns_per_op(fn):
    t = timeit.Timer(fn)
    number, _ = t.autorange()
    number = max(1, number)
    best = min(t.repeat(repeat=5, number=number))
    return best / number * 1e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default=",".join(map(str, SIZES)))
    args = p.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    print(f"# numpy {np.__version__}")
    print("benchmark,layout,size,ns_per_op")
    for n in sizes:
        a = make(n, "C")
        r, c = rows(n), n // rows(n)
        row = np.arange(float(c))
        ops = {
            "arrayAllocation": lambda: np.zeros(n),
            "viewCreation": lambda: a[0:r:2, ::-1],
            "reshape": lambda: a.reshape(c, r),
            "transpose": lambda: a.T,
            "broadcastCreation": lambda: np.broadcast_to(row, (r, c)),
        }
        for name, fn in ops.items():
            print(f"{name},-,{n},{ns_per_op(fn):.1f}", flush=True)
        b = make(n, "C")
        for layout in LAYOUTS:
            x = make(n, layout)
            print(f"miniAdd,{layout},{n},{ns_per_op(lambda: np.add(x, b)):.1f}", flush=True)
            print(f"miniSum,{layout},{n},{ns_per_op(lambda: np.sum(x)):.1f}", flush=True)


if __name__ == "__main__":
    main()
