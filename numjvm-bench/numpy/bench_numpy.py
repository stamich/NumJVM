"""NumPy counterparts of the NumJVM JMH benchmarks.

Times each operation with timeit (best of 5 rounds; each round long enough that one
round takes ~0.2 s) and prints CSV: benchmark,layout,size,ns_per_op. The m1 suite
names its rows like the JMH benchmarks (UfuncBenchmark.add, ...), puts the dtype in the
layout column for the per-dtype benchmarks, and can be merged with JMH's CSV by
compare.py.

Run single-threaded on the same machine as JMH:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 bench_numpy.py [--suite m0|m1] [--sizes 1024,65536]

Element-by-element reads (ContiguousRead/StridedRead) have no meaningful NumPy
counterpart: a Python loop over elements measures the interpreter. They are omitted.
"""

import argparse
import math
import timeit

import numpy as np

SIZES = [16, 128, 1024, 65536, 1048576, 16777216]
M1_SIZES = [1024, 65536, 1048576]
M1_DTYPES = ["int8", "int32", "int64", "float16", "float32", "float64", "complex128"]
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


def values(n, dtype="float64"):
    """0.5 + 4i/n, as Layouts.values in the JMH sources."""
    return (np.arange(float(n)) * (4.0 / n) + 0.5).astype(dtype)


def make_m1(n, layout, dtype="float64"):
    r, c = rows(n), n // rows(n)
    if layout == "C":
        return values(n, dtype).reshape(r, c)
    if layout == "F":
        return np.asfortranarray(values(n, dtype).reshape(r, c))
    if layout == "TRANSPOSED":
        return values(n, dtype).reshape(c, r).T
    if layout == "STRIDED":
        return values(2 * n, dtype).reshape(r, 2 * c)[:, ::2]
    if layout == "BROADCASTED":
        return np.broadcast_to(values(c, dtype), (r, c))
    raise ValueError(layout)


def m1(sizes):
    for n in sizes:
        b = make_m1(n, "C")
        row = b[0]
        for layout in LAYOUTS:
            a = make_m1(n, layout)
            out = np.zeros(a.shape)
            ufuncs = {
                "add": lambda: np.add(a, b),
                "addScalar": lambda: np.add(a, 1.5),
                "addRow": lambda: np.add(a, row),
                "addOut": lambda: np.add(a, b, out=out),
                "multiply": lambda: np.multiply(a, b),
                "divide": lambda: np.divide(a, b),
                "maximum": lambda: np.maximum(a, b),
                "less": lambda: np.less(a, b),
                "sqrt": lambda: np.sqrt(a),
                "exp": lambda: np.exp(a),
                "sin": lambda: np.sin(a),
            }
            for name, fn in ufuncs.items():
                print(f"UfuncBenchmark.{name},{layout},{n},{ns_per_op(fn):.1f}", flush=True)
            reductions = {
                "sumAll": lambda: np.sum(a),
                "sumAxis0": lambda: np.sum(a, axis=0),
                "sumAxis1": lambda: np.sum(a, axis=1),
                "maxAll": lambda: np.max(a),
                "meanAll": lambda: np.mean(a),
                "prodAxis0": lambda: np.prod(a, axis=0),
            }
            for name, fn in reductions.items():
                print(f"ReductionBenchmark.{name},{layout},{n},{ns_per_op(fn):.1f}", flush=True)
    n = 1048576
    for dt in M1_DTYPES:
        a = make_m1(n, "C", dt)
        b = make_m1(n, "C", dt)
        with np.errstate(all="ignore"):
            ops = {
                "UfuncDTypeBenchmark.add": lambda: np.add(a, b),
                "UfuncDTypeBenchmark.multiply": lambda: np.multiply(a, b),
                "UfuncDTypeBenchmark.divide": lambda: np.divide(a, b),
                "UfuncDTypeBenchmark.sqrt": lambda: np.sqrt(a),
                "UfuncDTypeBenchmark.exp": lambda: np.exp(a),
                "ReductionDTypeBenchmark.sum": lambda: np.sum(a),
                "ReductionDTypeBenchmark.mean": lambda: np.mean(a),
                "ReductionDTypeBenchmark.max": lambda: np.max(a.real),
                "ReductionDTypeBenchmark.any": lambda: np.any(a),
            }
            for name, fn in ops.items():
                print(f"{name},{dt.upper()},{n},{ns_per_op(fn):.1f}", flush=True)


def ns_per_op(fn):
    t = timeit.Timer(fn)
    number, _ = t.autorange()
    number = max(1, number)
    best = min(t.repeat(repeat=5, number=number))
    return best / number * 1e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=["m0", "m1"], default="m0")
    p.add_argument("--sizes", default=None)
    args = p.parse_args()
    default = SIZES if args.suite == "m0" else M1_SIZES
    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else default
    print(f"# numpy {np.__version__}")
    print("benchmark,layout,size,ns_per_op")
    if args.suite == "m1":
        m1(sizes)
        return
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
