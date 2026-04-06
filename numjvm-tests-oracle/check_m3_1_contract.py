#!/usr/bin/env python3
"""Validate contraction fixtures independently of the unimplemented Java APIs."""
import json
from pathlib import Path
import numpy as np

path = Path(__file__).parent / "fixtures" / "m3_1_contractions.json"
for case in json.loads(path.read_text())["cases"]:
    a = np.asarray(case["a"], dtype=np.float64)
    b = np.asarray(case["b"], dtype=np.float64)
    if case["operation"] == "dot":
        result = np.dot(a, b)
    else:
        result = np.vecdot(a, b, axis=case["axis"])
    assert list(np.shape(result)) == case["expected_shape"], case["name"]
    np.testing.assert_array_equal(result, np.asarray(case["expected"]))
    print(case["name"], "ok")

# Float32 intermediate rounding is observable when a later node is float64.
x = np.float32(1.0)
y = np.float32(2.0 ** -24)
rounded = np.float64(np.add(x, y)) + np.float64(2.0 ** -25)
unrounded = np.float64(x) + np.float64(y) + np.float64(2.0 ** -25)
assert rounded != unrounded
assert rounded == 1.0 + 2.0 ** -25
assert np.signbit(np.negative(np.float32(0.0)))
with np.errstate(invalid="ignore"):
    assert np.isnan(np.sqrt(np.float32(-1.0)))

for operation in (
    lambda: np.dot(np.ones(2), np.ones(3)),
    lambda: np.vecdot(np.ones(2), np.ones(2), axis=2),
):
    try:
        operation()
    except (ValueError, IndexError):
        continue
    raise AssertionError("invalid contraction was accepted")
