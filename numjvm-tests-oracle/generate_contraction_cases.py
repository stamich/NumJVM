#!/usr/bin/env python3
"""Deterministic contractions with exact-sum references and NumPy shape checks."""
import argparse
import math
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--check', action='store_true')
args = parser.parse_args()
rng = np.random.default_rng(20260502)
cases = [
    ('dot', -1, (), (3,)), ('dot', -1, (3,), (3,)),
    ('dot', -1, (2, 3), (3, 4)), ('dot', -1, (2, 3, 4), (2, 4, 3)),
    ('dot', -1, (2, 0), (0, 3)), ('dot', -1, (259,), (259,)),
    ('vecdot', -1, (2, 3), (3,)), ('vecdot', -1, (2, 1, 4), (3, 4)),
    ('vecdot', 0, (3, 2), (3, 2)), ('vecdot', -1, (259,), (259,)),
]

def shape(value):
    return ','.join(map(str, value)) or '-'

def values(value):
    return ','.join(repr(float(v)) for v in np.asarray(value).flat) or '-'

def reference(op, axis, a, b, dtype, magnitude=False):
    if op == 'dot' and (a.ndim == 0 or b.ndim == 0):
        product = np.asarray(a, dtype=np.float64) * np.asarray(b, dtype=np.float64)
        return np.asarray(np.abs(product) if magnitude else product, dtype=dtype)
    if op == 'dot':
        ai = a.ndim - 1
        bi = 0 if b.ndim == 1 else b.ndim - 2
        dims = a.shape[:-1] + b.shape[:bi] + (() if b.ndim == 1 else b.shape[-1:])
        result = np.empty(dims, dtype=dtype)
        for index in np.ndindex(dims):
            left_prefix = index[:ai]
            right_prefix = index[ai:ai + bi]
            right_tail = () if b.ndim == 1 else index[-1:]
            result[index] = math.fsum((abs(float(a[left_prefix + (k,)]) * float(b[right_prefix + (k,) + right_tail])) if magnitude else float(a[left_prefix + (k,)]) * float(b[right_prefix + (k,) + right_tail])) for k in range(a.shape[ai]))
        return result
    moved_a = np.moveaxis(a, axis, -1)
    moved_b = np.moveaxis(b, axis, -1)
    batch = np.broadcast_shapes(moved_a.shape[:-1], moved_b.shape[:-1])
    aa = np.broadcast_to(moved_a, batch + moved_a.shape[-1:])
    bb = np.broadcast_to(moved_b, batch + moved_b.shape[-1:])
    result = np.empty(batch, dtype=dtype)
    for index in np.ndindex(batch):
        result[index] = math.fsum((abs(float(aa[index + (k,)]) * float(bb[index + (k,)])) if magnitude else float(aa[index + (k,)]) * float(bb[index + (k,)])) for k in range(aa.shape[-1]))
    return result

lines = ['# operation\taxis\tdtypeA\tdtypeB\tshapeA\tshapeB\tshapeOut\tA\tB\texpected\tscale']
for ta, tb in [('float32', 'float32'), ('float64', 'float64'), ('float32', 'float64')]:
    for op, axis, sa, sb in cases:
        a = np.asarray(rng.uniform(-2, 2, sa), dtype=ta)
        b = np.asarray(rng.uniform(-2, 2, sb), dtype=tb)
        expected = reference(op, axis, a, b, np.result_type(a, b))
        scale = reference(op, axis, a, b, np.float64, magnitude=True)
        numpy_result = np.dot(a, b) if op == 'dot' else np.vecdot(a, b, axis=axis)
        assert np.shape(numpy_result) == expected.shape
        np.testing.assert_allclose(numpy_result, expected, rtol=2e-5, atol=2e-5)
        lines.append('\t'.join([op, str(axis), ta, tb, shape(sa), shape(sb), shape(expected.shape),
                                values(a), values(b), values(expected), values(scale)]))
text = '\n'.join(lines) + '\n'
path = Path(__file__).resolve().parent.parent / 'numjvm-linalg/src/test/resources/contraction-oracle.tsv'
if args.check:
    if path.read_text() != text:
        raise SystemExit('contraction fixtures differ')
    print(len(lines) - 1, 'contraction cases reproduce with NumPy', np.__version__)
else:
    path.write_text(text)
