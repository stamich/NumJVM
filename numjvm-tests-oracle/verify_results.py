"""Checks that NumPy reads the arrays NumJVM wrote during OracleTest and that they equal
NumPy's own results.

OracleTest (numjvm-io) writes every array-valued result to an output directory, one
`<case id>.npy` per case. This script loads each of them with np.load and compares
dtype, shape, memory order and values with the expected fixture: exactly (NaN equal to
NaN), or within the case's "ulp" tolerance for transcendental functions (decision D6).

Usage:
  python3 verify_results.py <oracle-out directory>     # e.g. numjvm-io/build/oracle-out
"""

import json
import os
import sys

import numpy as np

from generate_cases import ulp_distance


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    out_dir = sys.argv[1]
    fixtures = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
    with open(os.path.join(fixtures, "manifest.json"), encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    checked, failures = 0, []
    for case in cases:
        expect = case["expect"]
        if "file" not in expect:
            continue
        path = os.path.join(out_dir, case["id"] + ".npy")
        if not os.path.exists(path):
            failures.append(f"{case['id']}: NumJVM wrote no result")
            continue
        try:
            actual = np.load(path, allow_pickle=False)
        except Exception as e:  # noqa: BLE001 - any load failure is a test failure
            failures.append(f"{case['id']}: np.load failed: {e}")
            continue
        expected = np.load(os.path.join(fixtures, expect["file"]), allow_pickle=False)
        if actual.dtype != expected.dtype:
            failures.append(f"{case['id']}: dtype {actual.dtype} != {expected.dtype}")
        elif actual.shape != expected.shape:
            failures.append(f"{case['id']}: shape {actual.shape} != {expected.shape}")
        elif "ulp" in case["params"]:
            worst = int(np.max(ulp_distance(actual, expected), initial=0))
            if worst > case["params"]["ulp"]:
                failures.append(f"{case['id']}: values differ by {worst} ulp (tolerance {case['params']['ulp']})")
        elif not np.array_equal(actual, expected, equal_nan=True):
            failures.append(f"{case['id']}: values differ")
        elif case["op"] == "copy" and expect["fortran"] and not actual.flags.f_contiguous:
            failures.append(f"{case['id']}: expected Fortran order")
        checked += 1
    for line in failures:
        print("FAIL", line)
    print(f"{checked} NumJVM-written files read by numpy {np.__version__}; {len(failures)} failure(s)")
    return 1 if failures or checked == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
