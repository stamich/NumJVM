"""Generates the NumPy oracle: input arrays, expected results and manifest.json.

NumJVM's differential tests (numjvm-io, OracleTest) replay every case in the manifest
and compare with the expected result. Every case carries a compatibility tag:

  ARRAY_API            behaviour defined by the Python Array API Standard
  NUMPY_COMPATIBILITY  NumPy behaviour beyond the standard that NumJVM copies
  NUMJVM_SPECIFIC      a deliberate NumJVM difference, checked against NumJVM's own rule

Usage:
  python3 generate_cases.py            # (re)write fixtures/ and fixtures/manifest.json
  python3 generate_cases.py --check    # regenerate into a temp dir, fail if anything differs

The output is deterministic: inputs are built from integer formulas, never from a RNG.
"""

import argparse
import filecmp
import json
import os
import sys
import tempfile

import numpy as np

ARRAY_API = "ARRAY_API"
NUMPY = "NUMPY_COMPATIBILITY"
NUMJVM = "NUMJVM_SPECIFIC"


def hash_unit(n):
    """(i * 2654435761) mod 2**32 / 2**32: exact in float64, reproducible in Java."""
    i = np.arange(n, dtype=np.uint64)
    return ((i * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)).astype(np.float64) / 4294967296.0


def inputs():
    return {
        "f8_2x3x4": np.arange(24.0).reshape(2, 3, 4),
        "f8_3x4": np.arange(12.0).reshape(3, 4),
        "f8_hash_7x11": hash_unit(77).reshape(7, 11),
        "f8_row4": np.array([100.0, 200.0, 300.0, 400.0]),
        "f8_col3": np.array([[1000.0], [2000.0], [3000.0]]),
        "f8_scalar": np.array(2.5),
        "f8_empty_0x3": np.zeros((0, 3)),
        "i4_3x4": np.arange(12, dtype=np.int32).reshape(3, 4),
        "i8_neg": np.array([-5, -1, 0, 7, 2 ** 40], dtype=np.int64),
        "u1_5": np.arange(5, dtype=np.uint8) * 60,
        "u8_big": np.array([0, 1, 2 ** 63, 2 ** 64 - 1], dtype=np.uint64),
        "b1_2x3": np.array([[True, False, True], [False, True, False]]),
        "f2_4": np.array([0.1, 1.5, 65504.0, -2.0], dtype=np.float16),
        "f4_2x2": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "c16_2x2": (np.arange(4.0) + 1j * np.arange(4.0, 8.0)).reshape(2, 2),
        "c8_3": np.array([1 + 2j, 3 - 4j, 0j], dtype=np.complex64),
    }


def slice_expr(spec):
    """'1:5:2', ':', '3', 'None', '...' -> Python index component."""
    if spec == "None":
        return None
    if spec == "...":
        return Ellipsis
    if ":" not in spec:
        return int(spec)
    parts = [int(p) if p else None for p in spec.split(":")]
    return slice(*parts)


def cases(arr):
    """Each case: (id, tag, op, inputs, params, function producing the expected value)."""
    c = []

    def add(cid, tag, op, ins, params, fn):
        c.append((cid, tag, op, ins, params, fn))

    # --- views ------------------------------------------------------------------------
    for name in ["f8_2x3x4", "f8_3x4", "i4_3x4", "b1_2x3", "c16_2x2", "f8_scalar", "f8_empty_0x3"]:
        add(f"transpose_{name}", ARRAY_API, "transpose", [name], {}, lambda a: a.T)
    add("permute_dims_102", ARRAY_API, "permute_dims", ["f8_2x3x4"], {"axes": [1, 0, 2]},
        lambda a: np.permute_dims(a, (1, 0, 2)))
    add("permute_dims_neg", ARRAY_API, "permute_dims", ["f8_2x3x4"], {"axes": [-1, 0, 1]},
        lambda a: np.permute_dims(a, (-1, 0, 1)))
    add("swap_axes_0_2", NUMPY, "swap_axes", ["f8_2x3x4"], {"axes": [0, 2]}, lambda a: np.swapaxes(a, 0, 2))

    slice_cases = [
        ("rev", "f8_3x4", ["::-1"]),
        ("rev_both", "f8_3x4", ["::-1", "::-1"]),
        ("step2", "f8_2x3x4", [":", ":", "::2"]),
        ("neg_step_bounds", "f8_3x4", [":", "3:0:-2"]),
        ("row", "f8_3x4", ["1"]),
        ("neg_index", "f8_2x3x4", ["-1", "-2"]),
        ("element", "f8_3x4", ["2", "1"]),
        ("ellipsis_last", "f8_2x3x4", ["...", "0"]),
        ("ellipsis_newaxis", "f8_3x4", ["...", "None"]),
        ("newaxis_front", "i4_3x4", ["None", ":", "1:3"]),
        ("clipped", "f8_3x4", ["1:100", "-100:2"]),
        ("empty_range", "f8_3x4", ["2:2"]),
        ("bool_cols", "b1_2x3", [":", "::2"]),
        ("complex_rev", "c16_2x2", ["::-1"]),
        ("uint8_tail", "u1_5", ["-3:"]),
    ]
    # np.asarray: indexing every axis gives a 0-D array in the Array API (and NumJVM),
    # where NumPy returns a scalar.
    for cid, name, specs in slice_cases:
        add(f"slice_{cid}", ARRAY_API, "slice", [name], {"index": specs},
            lambda a, specs=specs: np.asarray(a[tuple(slice_expr(s) for s in specs)]))

    reshape_cases = [
        ("f8_2x3x4", [6, 4]), ("f8_2x3x4", [4, -1]), ("f8_2x3x4", [24]), ("f8_2x3x4", [2, 1, 12]),
        ("f8_3x4", [12, 1]), ("f8_scalar", [1, 1]), ("f8_empty_0x3", [3, 0]), ("i4_3x4", [2, 6]),
    ]
    for name, shape in reshape_cases:
        add(f"reshape_{name}_{'x'.join(map(str, shape))}", ARRAY_API, "reshape", [name], {"shape": shape},
            lambda a, shape=shape: a.reshape(shape))
    # reshape of a transposed view: NumPy copies; NumJVM's reshape copies too and reshapeView refuses
    add("reshape_transposed_copy", NUMPY, "reshape_transposed", ["f8_3x4"], {"shape": [12]},
        lambda a: a.T.reshape(12))
    add("reshape_view_transposed", NUMJVM, "reshape_view_transposed", ["f8_3x4"], {"shape": [12]}, None)

    for name, shape in [("f8_row4", [3, 4]), ("f8_col3", [3, 4]), ("f8_scalar", [2, 2]), ("i4_3x4", [2, 3, 4]),
                        ("f8_empty_0x3", [2, 0, 3])]:
        add(f"broadcast_to_{name}", ARRAY_API, "broadcast_to", [name], {"shape": shape},
            lambda a, shape=shape: np.broadcast_to(a, shape))

    for shapes in [[[8, 1, 6, 1], [7, 1, 5]], [[3, 1], [1, 4]], [[5, 1, 4], [1, 3, 1]], [[], [2, 3]],
                   [[0, 3], [1, 3]], [[4], [3, 1], [2, 1, 1]]]:
        add("broadcast_shapes_" + "_".join("x".join(map(str, s)) or "scalar" for s in shapes), ARRAY_API,
            "broadcast_shapes", [], {"shapes": shapes}, lambda shapes=shapes: list(np.broadcast_shapes(*shapes)))
    add("broadcast_shapes_mismatch", ARRAY_API, "broadcast_shapes", [], {"shapes": [[2, 3], [4, 3]]},
        lambda: None)

    # --- copies and conversions -----------------------------------------------------------
    add("copy_c_of_transpose", ARRAY_API, "copy", ["f8_2x3x4"], {"view": "transpose", "order": "C"},
        lambda a: np.ascontiguousarray(a.T))
    add("copy_f", NUMPY, "copy", ["f8_3x4"], {"view": "none", "order": "F"}, lambda a: np.asfortranarray(a))
    astype_cases = [
        ("f8_3x4", "<i4"), ("f8_hash_7x11", "<f4"), ("f8_hash_7x11", "<f2"), ("i8_neg", "|u1"),
        ("i8_neg", "<u8"), ("u8_big", "<i8"), ("u8_big", "<f8"), ("b1_2x3", "<f8"), ("f8_3x4", "|b1"),
        ("f4_2x2", "<f8"), ("f8_3x4", "<c16"), ("c16_2x2", "<c8"),
    ]
    # NumPy discards the imaginary part with a ComplexWarning; NumJVM refuses.
    add("astype_complex_to_real", NUMJVM, "astype", ["c16_2x2"], {"dtype": "<f8"}, None)
    for name, descr in astype_cases:
        add(f"astype_{name}_{descr[1:]}", NUMPY, "astype", [name], {"dtype": descr},
            lambda a, descr=descr: a.astype(np.dtype(descr), casting="unsafe"))

    # --- validation kernels ---------------------------------------------------------------
    add("add_contiguous", ARRAY_API, "add", ["f8_3x4", "f8_3x4"], {}, lambda a, b: a + b)
    add("add_row_broadcast", ARRAY_API, "add", ["f8_3x4", "f8_row4"], {}, lambda a, b: a + b)
    add("add_col_row", ARRAY_API, "add", ["f8_col3", "f8_row4"], {}, lambda a, b: a + b)
    add("add_scalar", ARRAY_API, "add", ["f8_scalar", "f8_3x4"], {}, lambda a, b: a + b)
    add("add_transposed", ARRAY_API, "add_transposed", ["f8_hash_7x11", "f8_hash_7x11"], {}, lambda a, b: a.T + b.T)
    add("add_incompatible", ARRAY_API, "add", ["f8_empty_0x3", "f8_row4"], {}, None)  # (0, 3) + (4,): error
    for name in ["f8_3x4", "f8_hash_7x11", "f8_scalar", "f8_empty_0x3"]:
        add(f"sum_{name}", ARRAY_API, "sum", [name], {"view": "none"}, lambda a: float(np.sum(a)))
        add(f"sum_{name}_T", ARRAY_API, "sum", [name], {"view": "transpose"}, lambda a: float(np.sum(a.T)))
    for n in [1, 7, 8, 9, 127, 128, 129, 1000, 65536, 1048576, 16777216]:
        add(f"sum_hash_{n}", ARRAY_API, "sum_generated", [], {"generator": "hash_unit", "n": n},
            lambda n=n: float(np.sum(hash_unit(n))))

    # --- flags ------------------------------------------------------------------------------
    for cid, name, specs in [("row", "f8_3x4", ["1"]), ("col", "f8_3x4", [":", "1:2"]),
                             ("strided", "f8_3x4", [":", "::2"]), ("rev", "f8_3x4", ["::-1"])]:
        add(f"flags_{cid}", NUMPY, "flags", [name], {"index": specs},
            lambda a, specs=specs: {"c": bool(a[tuple(slice_expr(s) for s in specs)].flags.c_contiguous),
                                    "f": bool(a[tuple(slice_expr(s) for s in specs)].flags.f_contiguous)})

    # --- printing ---------------------------------------------------------------------------
    for name in ["f8_3x4", "f8_hash_7x11", "i8_neg", "u1_5", "u8_big", "b1_2x3", "f2_4", "f4_2x2", "c16_2x2",
                 "c8_3", "f8_scalar", "f8_empty_0x3"]:
        add(f"repr_{name}", NUMPY, "repr", [name], {}, lambda a: repr(a))
    return c


def generate(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    arr = inputs()
    for name, a in arr.items():
        np.save(os.path.join(out_dir, f"in_{name}.npy"), a)
    manifest = []
    for cid, tag, op, ins, params, fn in cases(arr):
        entry = {"id": cid, "tag": tag, "op": op, "inputs": [f"in_{n}.npy" for n in ins], "params": params}
        if fn is None:
            entry["expect"] = {"error": True}
        else:
            try:
                result = fn(*[arr[n] for n in ins])
            except ValueError:
                result = None
            if result is None:
                entry["expect"] = {"error": True}
            elif isinstance(result, np.ndarray):
                fname = f"expected_{cid}.npy"
                np.save(os.path.join(out_dir, fname), result)
                entry["expect"] = {"file": fname, "shape": list(result.shape), "dtype": result.dtype.str,
                                   "fortran": bool(result.flags.f_contiguous and not result.flags.c_contiguous)}
            elif isinstance(result, float):
                entry["expect"] = {"value": result}
            elif isinstance(result, str):
                entry["expect"] = {"repr": result}
            elif isinstance(result, list):
                entry["expect"] = {"shape": result}
            elif isinstance(result, dict):
                entry["expect"] = {"flags": result}
            else:
                raise TypeError(f"{cid}: unexpected result type {type(result)}")
        manifest.append(entry)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"numpy_version_used": "2.x", "cases": manifest}, f, indent=1, sort_keys=True)
        f.write("\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="verify the committed fixtures are up to date")
    args = p.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    fixtures = os.path.join(here, "fixtures")
    if not args.check:
        manifest = generate(fixtures)
        print(f"wrote {len(manifest)} cases to {fixtures}")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        generate(tmp)
        expected = sorted(os.listdir(tmp))
        actual = sorted(f for f in os.listdir(fixtures) if not f.startswith("."))
        if expected != actual:
            print("fixture file list differs:", sorted(set(expected) ^ set(actual)))
            return 1
        _, mismatch, errors = filecmp.cmpfiles(tmp, fixtures, expected, shallow=False)
        if mismatch or errors:
            print("fixtures out of date:", mismatch + errors)
            return 1
    print(f"fixtures are up to date ({len(expected)} files, numpy {np.__version__})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
