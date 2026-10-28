"""Merges JMH results with NumPy's and prints a Markdown table of times and ratios.

    python3 compare.py <jmh.csv> <numpy.csv> [--filter UfuncBenchmark]

<jmh.csv> is JMH's CSV output (-rf csv -rff jmh.csv); <numpy.csv> is the output of
bench_numpy.py --suite m1. Rows are matched on benchmark, layout (or dtype) and size. The
ratio is NumJVM / NumPy: below 1 NumJVM is faster.
"""

import argparse
import csv
import sys


def jmh_rows(path):
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name = r["Benchmark"].rsplit(".", 2)
            bench = f"{name[-2]}.{name[-1]}"
            variant = r.get("Param: layout") or r.get("Param: dtype") or "-"
            size = r.get("Param: size", "-")
            rows[(bench, variant, size)] = (float(r["Score"]), float(r["Score Error (99.9%)"] or "nan"))
    return rows


def numpy_rows(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if not line.startswith("#")]
    for r in csv.DictReader(lines):
        rows[(r["benchmark"], r["layout"], r["size"])] = float(r["ns_per_op"])
    return rows


def fmt(ns):
    if ns != ns:
        return "–"
    if ns >= 1e6:
        return f"{ns / 1e6:.2f} ms"
    if ns >= 1e3:
        return f"{ns / 1e3:.1f} µs"
    return f"{ns:.0f} ns"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jmh")
    p.add_argument("numpy")
    p.add_argument("--filter", default="")
    args = p.parse_args()
    jmh = jmh_rows(args.jmh)
    npy = numpy_rows(args.numpy)
    print("| Benchmark | Layout / dtype | Size | NumJVM | ± (99.9%) | NumPy | NumJVM / NumPy |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for key in sorted(jmh, key=lambda k: (k[0], int(k[2]) if k[2].isdigit() else 0, k[1])):
        if args.filter not in key[0]:
            continue
        score, err = jmh[key]
        other = npy.get(key)
        ratio = f"{score / other:.2f}" if other else "–"
        print(f"| {key[0]} | {key[1]} | {key[2]} | {fmt(score)} | {fmt(err)} | {fmt(other) if other else '–'} | {ratio} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
