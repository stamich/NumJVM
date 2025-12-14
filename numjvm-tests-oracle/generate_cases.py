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


# Eight values per dtype that exercise wrap-around, signed zeros, infinities and NaN.
EDGE = {
    "bool": [True, False, True, True, False, False, True, False],
    "int8": [-128, -100, -1, 0, 1, 7, 100, 127],
    "uint8": [0, 1, 7, 100, 128, 200, 254, 255],
    "int16": [-32768, -300, -1, 0, 1, 7, 300, 32767],
    "uint16": [0, 1, 7, 300, 32768, 40000, 65534, 65535],
    "int32": [-2 ** 31, -70000, -1, 0, 1, 7, 70000, 2 ** 31 - 1],
    "uint32": [0, 1, 7, 70000, 2 ** 31, 3000000000, 2 ** 32 - 2, 2 ** 32 - 1],
    "int64": [-2 ** 63, -2 ** 40, -1, 0, 1, 7, 2 ** 40, 2 ** 63 - 1],
    "uint64": [0, 1, 7, 2 ** 40, 2 ** 63, 2 ** 63 + 12345, 2 ** 64 - 2, 2 ** 64 - 1],
    "float16": [-np.inf, -65504.0, -1.5, -0.0, 0.0, 0.1, 3.0, np.nan],
    "float32": [-np.inf, -3.0e38, -1.5, -0.0, 0.0, 0.1, 7.0, np.nan],
    "float64": [-np.inf, -1.0e308, -1.5, -0.0, 0.0, 0.1, 7.0, np.nan],
}
NUMPY_REAL = list(EDGE)


def edge(dtype):
    return np.array(EDGE[dtype], dtype=dtype)


def edge2(dtype):
    """A second operand: the edge values rotated by three, so every pair of positions differs."""
    return np.roll(edge(dtype), 3)


def inputs():
    arr = {}
    for dt in NUMPY_REAL:
        arr[f"edge_{dt}"] = edge(dt)
        arr[f"edge2_{dt}"] = edge2(dt)
    for dt in ["int8", "int16", "int32", "int64"]:
        lo = np.iinfo(dt).min
        arr[f"div_a_{dt}"] = np.array([lo, lo, 5, -5, 0, 7, -7, 7], dtype=dt)
        arr[f"div_b_{dt}"] = np.array([-1, 1, 0, 0, 0, -2, 2, 3], dtype=dt)
    for dt in ["uint8", "uint64"]:
        hi = np.iinfo(dt).max
        arr[f"div_a_{dt}"] = np.array([hi, hi, 5, 0, 7, 7], dtype=dt)
        arr[f"div_b_{dt}"] = np.array([1, 2, 0, 0, 3, hi], dtype=dt)
    for dt in [d for d in NUMPY_REAL if d != "bool"]:
        arr[f"exp_{dt}"] = np.array([0, 1, 2, 3, 7, 8, 0, 1], dtype=dt)
    for dt in ["float64", "float32"]:
        arr[f"pow_x_{dt}"] = np.array([1.0, 1.0, -1.0, -1.0, np.nan, 2.0, 0.0, -0.0, -8.0, 0.5], dtype=dt)
        arr[f"pow_y_{dt}"] = np.array([np.nan, np.inf, np.inf, -np.inf, 0.0, np.nan, -1.0, -1.0, 1 / 3, -np.inf],
                                      dtype=dt)
    # 64 values per float dtype for the transcendental functions: [-12, 12) and (0, 1000)
    for dt in ["float16", "float32", "float64"]:
        u = hash_unit(64)
        arr[f"dense_{dt}"] = (u * 24.0 - 12.0).astype(dt)
        # ldexp is exact; exp2 is not correctly rounded and its last bit depends on the CPU
        arr[f"dense_pos_{dt}"] = np.ldexp(1.0 + u, np.floor(u * 1e6).astype(np.int64) % 20 - 10).astype(dt)
    arr["cmp_i8"] = np.array([2 ** 63 - 1, -1, 5, -2 ** 63, 0, 2 ** 53 + 1], dtype=np.int64)
    arr["cmp_u8"] = np.array([2 ** 63, 2 ** 64 - 1, 5, 0, 0, 2 ** 53], dtype=np.uint64)
    arr["nan_x_float64"] = np.array([np.nan, 1.0, np.nan, -np.inf, 2.0, np.inf])
    arr["nan_y_float64"] = np.array([1.0, np.nan, np.nan, 3.0, -0.5, np.nan])
    arr["div_a_float64"] = np.array([6.0, -6.0, 0.0, -0.0, 7.5, -7.5, 1.0, np.inf, 5.0, 1e300])
    arr["div_b_float64"] = np.array([-3.0, 3.0, -1.0, 2.0, -2.0, 2.0, -0.0, 2.0, np.inf, 1e-300])
    return arr | {
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


PROMOTION_DTYPES = ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
                    "float16", "float32", "float64", "complex64", "complex128"]
CASTINGS = ["no", "equiv", "safe", "same_kind", "unsafe"]


def promotion_tag(a, b):
    """ARRAY_API where the standard defines the result type, else NUMPY_COMPATIBILITY."""
    ka, kb = np.dtype(a).kind, np.dtype(b).kind
    if a == b or ka == kb:
        return ARRAY_API
    ints = {"i", "u"}
    if ka in ints and kb in ints:
        return ARRAY_API if "uint64" not in (a, b) else NUMPY
    if {ka, kb} == {"f", "c"}:
        return ARRAY_API
    return NUMPY


# Decision D6: results of transcendental functions may differ from NumPy's by a few units in
# the last place (NumPy uses its own SIMD implementations or the C library, NumJVM uses
# java.lang.Math). Such cases carry "ulp" in their parameters; OracleTest and --check compare
# them with that tolerance instead of bit for bit.
ULP_TOLERANCE = {"power": 4, "exp": 4, "log": 4, "sin": 4, "cos": 4, "tanh": 4}


def ulp_distance(a, b):
    """Elementwise distance in units in the last place of a's dtype; NaN equals NaN."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.dtype.kind == "c":
        return np.maximum(ulp_distance(a.real, b.real), ulp_distance(a.imag, b.imag))

    def line(x):
        # sign-magnitude bit patterns mapped onto a monotonic integer line (-0.0 == 0.0)
        bits = x.view({2: np.uint16, 4: np.uint32, 8: np.uint64}[x.dtype.itemsize])
        n = 8 * x.dtype.itemsize
        magnitude = (bits & np.array((1 << (n - 1)) - 1, dtype=bits.dtype)).astype(np.int64)
        negative = (bits >> np.array(n - 1, dtype=bits.dtype)).astype(bool)
        return np.where(negative, -magnitude, magnitude)

    d = np.abs(line(a) - line(b))
    return np.where(np.isnan(a) & np.isnan(b), 0, d)


def ufunc(name, *args):
    """Applies a NumPy ufunc with floating-point warnings silenced (NumJVM has no warnings)."""
    with np.errstate(all="ignore"):
        return getattr(np, name)(*args)


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

    # --- arithmetic ufuncs ---------------------------------------------------------------------
    for name in ["add", "subtract", "multiply", "divide", "floor_divide", "remainder", "power", "maximum",
                 "minimum", "fmax", "fmin"]:
        for dt in NUMPY_REAL:
            tag = ARRAY_API if dt != "bool" else NUMPY
            if name == "subtract" and dt == "bool":
                add(f"ufunc_{name}_{dt}", NUMPY, "ufunc", [f"edge_{dt}", f"edge2_{dt}"], {"ufunc": name}, None)
                continue
            add(f"ufunc_{name}_{dt}", tag, "ufunc", [f"edge_{dt}", f"edge2_{dt}"], {"ufunc": name},
                lambda a, b, name=name: ufunc(name, a, b))
        for x, y in [("int8", "uint8"), ("int16", "float16"), ("int64", "uint64"), ("bool", "int8"),
                     ("uint32", "float32"), ("float16", "float64"), ("int32", "uint64"), ("uint16", "int16")]:
            add(f"ufunc_{name}_{x}_{y}", promotion_tag(x, y), "ufunc", [f"edge_{x}", f"edge2_{y}"], {"ufunc": name},
                lambda a, b, name=name: ufunc(name, a, b))
        add(f"ufunc_{name}_broadcast", ARRAY_API, "ufunc", ["f8_col3", "f8_row4"], {"ufunc": name},
            lambda a, b, name=name: ufunc(name, a, b))
        add(f"ufunc_{name}_transposed", ARRAY_API, "ufunc", ["f8_hash_7x11", "f8_hash_7x11"],
            {"ufunc": name, "view": "transpose"}, lambda a, b, name=name: ufunc(name, a.T, b.T))
    # integer MIN // -1 wraps; division and remainder by zero give 0 for integers
    for dt in ["int8", "int16", "int32", "int64", "uint8", "uint64"]:
        for name in ["floor_divide", "remainder", "divide"]:
            add(f"ufunc_{name}_{dt}_min_and_zero", NUMPY, "ufunc", [f"div_a_{dt}", f"div_b_{dt}"], {"ufunc": name},
                lambda a, b, name=name: ufunc(name, a, b))
    for name in ["floor_divide", "remainder"]:
        add(f"ufunc_{name}_float64_signs", NUMPY, "ufunc", ["div_a_float64", "div_b_float64"], {"ufunc": name},
            lambda a, b, name=name: ufunc(name, a, b))

    # --- comparisons ------------------------------------------------------------------------
    for name in ["equal", "not_equal", "less", "less_equal", "greater", "greater_equal"]:
        for dt in NUMPY_REAL:
            add(f"ufunc_{name}_{dt}", ARRAY_API, "ufunc", [f"edge_{dt}", f"edge2_{dt}"], {"ufunc": name},
                lambda a, b, name=name: ufunc(name, a, b))
            add(f"ufunc_{name}_{dt}_self", ARRAY_API, "ufunc", [f"edge_{dt}", f"edge_{dt}"], {"ufunc": name},
                lambda a, b, name=name: ufunc(name, a, b))
        # signed integers against uint64 are compared exactly (NumPy 2), not through float64
        for x, y in [("cmp_i8", "cmp_u8"), ("cmp_u8", "cmp_i8"), ("edge_int8", "edge2_uint64"),
                     ("edge_int16", "edge_float16"), ("edge_bool", "edge_int8")]:
            add(f"ufunc_{name}_{x}_{y}", NUMPY, "ufunc", [x, y], {"ufunc": name},
                lambda a, b, name=name: ufunc(name, a, b))
        # Java long scalars outside the array's range compare exactly
        for dt, value in [("uint8", 300), ("uint8", -1), ("int8", 1000), ("uint64", -1), ("int64", 5),
                          ("bool", 2)]:
            add(f"ufunc_{name}_{dt}_long_{value}", NUMPY, "ufunc", [f"edge_{dt}"],
                {"ufunc": name, "scalar": value, "scalar_type": "long", "scalar_side": "right"},
                lambda a, value=value, name=name: ufunc(name, a, value))
            add(f"ufunc_{name}_long_{value}_{dt}", NUMPY, "ufunc", [f"edge_{dt}"],
                {"ufunc": name, "scalar": value, "scalar_type": "long", "scalar_side": "left"},
                lambda a, value=value, name=name: ufunc(name, value, a))
        add(f"ufunc_{name}_int8_double", NUMPY, "ufunc", ["edge_int8"],
            {"ufunc": name, "scalar": 1.5, "scalar_type": "double", "scalar_side": "right"},
            lambda a, name=name: ufunc(name, a, 1.5))

    # --- logical and bitwise ---------------------------------------------------------------------
    for name in ["logical_and", "logical_or", "logical_xor"]:
        for dt in NUMPY_REAL:
            add(f"ufunc_{name}_{dt}", ARRAY_API, "ufunc", [f"edge_{dt}", f"edge2_{dt}"], {"ufunc": name},
                lambda a, b, name=name: ufunc(name, a, b))
        add(f"ufunc_{name}_float64_int8", NUMPY, "ufunc", ["edge_float64", "edge2_int8"], {"ufunc": name},
            lambda a, b, name=name: ufunc(name, a, b))
        add(f"ufunc_{name}_complex128", NUMPY, "ufunc", ["c16_2x2", "c16_2x2"], {"ufunc": name, "view": "transpose"},
            lambda a, b, name=name: ufunc(name, a.T, b.T))
    for dt in NUMPY_REAL:
        add(f"ufunc_logical_not_{dt}", ARRAY_API, "ufunc", [f"edge_{dt}"], {"ufunc": "logical_not"},
            lambda a: ufunc("logical_not", a))
    for name in ["bitwise_and", "bitwise_or", "bitwise_xor"]:
        for dt in NUMPY_REAL:
            if dt.startswith("float"):
                add(f"ufunc_{name}_{dt}", ARRAY_API, "ufunc", [f"edge_{dt}", f"edge2_{dt}"], {"ufunc": name}, None)
                continue
            add(f"ufunc_{name}_{dt}", ARRAY_API, "ufunc", [f"edge_{dt}", f"edge2_{dt}"], {"ufunc": name},
                lambda a, b, name=name: ufunc(name, a, b))
        for x, y in [("int8", "uint8"), ("bool", "int16"), ("int64", "uint32")]:
            add(f"ufunc_{name}_{x}_{y}", promotion_tag(x, y), "ufunc", [f"edge_{x}", f"edge2_{y}"], {"ufunc": name},
                lambda a, b, name=name: ufunc(name, a, b))
        add(f"ufunc_{name}_uint8_long", NUMPY, "ufunc", ["edge_uint8"],
            {"ufunc": name, "scalar": 15, "scalar_type": "long", "scalar_side": "right"},
            lambda a, name=name: ufunc(name, a, 15))
    for dt in NUMPY_REAL:
        if dt.startswith("float"):
            add(f"ufunc_invert_{dt}", ARRAY_API, "ufunc", [f"edge_{dt}"], {"ufunc": "invert"}, None)
        else:
            add(f"ufunc_invert_{dt}", ARRAY_API, "ufunc", [f"edge_{dt}"], {"ufunc": "invert"},
                lambda a: ufunc("invert", a))

    # --- unary arithmetic --------------------------------------------------------------------
    for name in ["negative", "positive", "absolute", "sign", "square"]:
        for dt in NUMPY_REAL:
            tag = NUMPY if dt == "bool" else ARRAY_API
            add(f"ufunc_{name}_{dt}", tag, "ufunc", [f"edge_{dt}"], {"ufunc": name}, lambda a, name=name: ufunc(name, a))
        add(f"ufunc_{name}_transposed", ARRAY_API, "ufunc", ["f8_hash_7x11"], {"ufunc": name, "view": "transpose"},
            lambda a, name=name: ufunc(name, a.T))

    # --- floating-point functions (decision D6) -------------------------------------------------
    for name in ["sqrt", "exp", "log", "sin", "cos", "tanh"]:
        for dt in NUMPY_REAL:
            add(f"ufunc_{name}_{dt}", NUMPY if not dt.startswith("float") else ARRAY_API, "ufunc", [f"edge_{dt}"],
                {"ufunc": name}, lambda a, name=name: ufunc(name, a))
        for dt in ["float16", "float32", "float64"]:
            src = "dense_pos" if name in ("sqrt", "log") else "dense"
            add(f"ufunc_{name}_{dt}_{src}", ARRAY_API, "ufunc", [f"{src}_{dt}"], {"ufunc": name},
                lambda a, name=name: ufunc(name, a))

    # integer powers with non-negative exponents wrap; floating-point special cases follow C99
    for dt in [d for d in NUMPY_REAL if d != "bool"]:
        add(f"ufunc_power_{dt}_exponents", NUMPY if dt.startswith(("int", "uint")) else ARRAY_API, "ufunc",
            [f"edge_{dt}", f"exp_{dt}"], {"ufunc": "power"}, lambda a, b: ufunc("power", a, b))
    add("ufunc_power_float64_c99", NUMPY, "ufunc", ["pow_x_float64", "pow_y_float64"], {"ufunc": "power"},
        lambda a, b: ufunc("power", a, b))
    add("ufunc_power_float32_c99", NUMPY, "ufunc", ["pow_x_float32", "pow_y_float32"], {"ufunc": "power"},
        lambda a, b: ufunc("power", a, b))
    # NaN handling of the min/max family
    for name in ["maximum", "minimum", "fmax", "fmin"]:
        add(f"ufunc_{name}_nan", ARRAY_API if name in ("maximum", "minimum") else NUMPY, "ufunc",
            ["nan_x_float64", "nan_y_float64"], {"ufunc": name}, lambda a, b, name=name: ufunc(name, a, b))

    # Java long and double arguments are weak scalars, as Python int and float (NEP 50)
    weak = [("uint8", 1, "long"), ("int8", -128, "long"), ("int8", 1.5, "double"), ("bool", 1, "long"),
            ("bool", 1.5, "double"), ("float16", 100000, "long"), ("float32", 0.1, "double"),
            ("uint64", 2, "long"), ("int64", 3, "long"), ("float64", -2.5, "double")]
    for dt, value, kind in weak:
        for side in ["right", "left"]:
            add(f"ufunc_add_{dt}_{kind}_{side}", NUMPY, "ufunc", [f"edge_{dt}"],
                {"ufunc": "add", "scalar": value, "scalar_type": kind, "scalar_side": side},
                lambda a, value=value, side=side: ufunc("add", a, value) if side == "right" else ufunc("add", value, a))
        add(f"ufunc_multiply_{dt}_{kind}", NUMPY, "ufunc", [f"edge_{dt}"],
            {"ufunc": "multiply", "scalar": value, "scalar_type": kind, "scalar_side": "right"},
            lambda a, value=value: ufunc("multiply", a, value))
    for dt, value in [("uint8", 300), ("uint8", -1), ("int8", 128), ("uint64", -1)]:
        add(f"ufunc_add_{dt}_out_of_bounds_{value}", NUMPY, "ufunc", [f"edge_{dt}"],
            {"ufunc": "add", "scalar": value, "scalar_type": "long", "scalar_side": "right"}, None)

    # --- type promotion (ADR-005) --------------------------------------------------------------
    for a in PROMOTION_DTYPES:
        for b in PROMOTION_DTYPES:
            add(f"result_type_{a}_{b}", promotion_tag(a, b), "result_type", [], {"dtypes": [a, b]},
                lambda a=a, b=b: np.result_type(a, b))
    for seq in [["int16", "uint16", "float32"], ["int8", "uint16", "float16"], ["int64", "uint64", "float16"],
                ["int8", "uint8", "float32", "bool"], ["uint8", "int8", "uint16"], ["bool", "bool", "bool"],
                ["int32", "float16", "complex64"], ["uint32", "int8", "float32", "float16"]]:
        add("result_type_" + "_".join(seq), NUMPY, "result_type", [], {"dtypes": seq},
            lambda seq=seq: np.result_type(*seq))
    # bfloat16 is not a NumPy dtype; NumJVM follows ml_dtypes (ADR-005)
    for a, b, r in [("bfloat16", "float16", "float32"), ("bfloat16", "int8", "bfloat16"),
                    ("bfloat16", "uint8", "bfloat16"), ("bfloat16", "int16", "float32"),
                    ("bfloat16", "float32", "float32"), ("bfloat16", "float64", "float64"),
                    ("bfloat16", "complex64", "complex64"), ("bfloat16", "bool", "bfloat16"),
                    ("bfloat16", "bfloat16", "bfloat16")]:
        add(f"result_type_{a}_{b}", NUMJVM, "result_type", [], {"dtypes": [a, b], "expected": r}, None)
    for casting in CASTINGS:
        for a in PROMOTION_DTYPES:
            add(f"can_cast_{casting}_{a}", NUMPY, "can_cast", [], {"from": a, "casting": casting},
                lambda a=a, casting=casting: {b: bool(np.can_cast(a, b, casting)) for b in PROMOTION_DTYPES})

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
        if fn is None and "expected" in params:
            entry["expect"] = {"dtype_name": params.pop("expected")}
        elif fn is None:
            entry["expect"] = {"error": True}
        else:
            try:
                result = fn(*[arr[n] for n in ins])
            except (ValueError, TypeError):
                result = None
            if result is None:
                entry["expect"] = {"error": True}
            elif isinstance(result, np.ndarray):
                if op == "ufunc" and params["ufunc"] in ULP_TOLERANCE and result.dtype.kind in "fc":
                    params["ulp"] = ULP_TOLERANCE[params["ufunc"]]
                fname = f"expected_{cid}.npy"
                np.save(os.path.join(out_dir, fname), result)
                entry["expect"] = {"file": fname, "shape": list(result.shape), "dtype": result.dtype.str,
                                   "fortran": bool(result.flags.f_contiguous and not result.flags.c_contiguous)}
            elif isinstance(result, np.dtype):
                entry["expect"] = {"dtype_name": result.name}
            elif isinstance(result, float):
                entry["expect"] = {"value": result}
            elif isinstance(result, str):
                entry["expect"] = {"repr": result}
            elif isinstance(result, list):
                entry["expect"] = {"shape": result}
            elif isinstance(result, dict) and op == "can_cast":
                entry["expect"] = {"can_cast": result}
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
        generated = generate(tmp)
        expected = sorted(os.listdir(tmp))
        actual = sorted(f for f in os.listdir(fixtures) if not f.startswith("."))
        if expected != actual:
            print("fixture file list differs:", sorted(set(expected) ^ set(actual)))
            return 1
        _, mismatch, errors = filecmp.cmpfiles(tmp, fixtures, expected, shallow=False)
        tolerant = {c["expect"]["file"]: c["params"]["ulp"] for c in generated
                    if "ulp" in c["params"] and "file" in c["expect"]}
        for f in list(mismatch):
            if f in tolerant:
                new = np.load(os.path.join(tmp, f))
                old = np.load(os.path.join(fixtures, f))
                if new.dtype == old.dtype and new.shape == old.shape \
                        and int(np.max(ulp_distance(new, old), initial=0)) <= tolerant[f]:
                    mismatch.remove(f)
        if mismatch or errors:
            print("fixtures out of date:", mismatch + errors)
            return 1
    print(f"fixtures are up to date ({len(expected)} files, numpy {np.__version__})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
