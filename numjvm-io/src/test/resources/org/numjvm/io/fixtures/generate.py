"""Generates the .npy files used by NpyReaderTest.

Run from this directory with NumPy installed; the output is committed.

    python3 generate.py

Every array is np.arange(6).reshape(2, 3) converted to the dtype (complex arrays get
an imaginary part of 10 * arange), so the Java test can recompute the expected values.
"""

import numpy as np

DTYPES = ["b1", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f2", "f4", "f8", "c8", "c16"]


def base(code):
    a = np.arange(6).reshape(2, 3)
    if code == "b1":
        return (a % 2 == 0)
    if code.startswith("c"):
        return (a + 10j * a).astype(code)
    return a.astype(code)


for code in DTYPES:
    a = base(code)
    np.save(f"{code}.npy", a)                                      # native order (little-endian), C order
    np.save(f"{code}_fortran.npy", np.asfortranarray(a))           # fortran_order: True
    if code not in ("b1", "i1", "u1"):
        np.save(f"{code}_big.npy", a.astype(a.dtype.newbyteorder(">")))   # big-endian

np.save("scalar.npy", np.array(2.5))
np.save("empty.npy", np.zeros((0, 3), dtype=np.int16))
np.save("u8_max.npy", np.array([0, 2**63, 2**64 - 1], dtype=np.uint64))
np.save("f2_values.npy", np.array([0.1, 65504.0, -np.inf, np.nan], dtype=np.float16))
