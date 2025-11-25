# NDArray model

An `NDArray` is a typed, strided view of a buffer:

| Field | Meaning |
| --- | --- |
| buffer | the memory and its owner (see [memory model](memory-model.md)) |
| dtype | element type (see [dtype system](dtype-system.md)) |
| offset | byte position of element `[0, ..., 0]` in the buffer |
| shape | length of each axis; `[]` for a scalar, a `0` for an empty array |
| strides | byte distance between neighbours on each axis; may be negative or zero |
| flags | C/F contiguity, writeable, aligned, computed once |

The byte position of element `[i, j, k]` is `offset + i*strides[0] + j*strides[1] +
k*strides[2]`. A row-major `float64` matrix of shape `[2, 3]` has strides `[24, 8]`; its
transpose is the same buffer with shape `[3, 2]` and strides `[8, 24]`.

## Invariants

Checked once, when the array or view is created:

1. every dimension is `>= 0`, rank `<= 64`;
2. shape and strides have the same rank;
3. the dtype is not null;
4. **the byte extent lies inside the buffer**: with `low` the offset plus the negative
   spans and `high` the offset plus the positive spans plus one element, `[low, high)` must
   be inside `[0, buffer size)`. An array with no elements touches no memory and may have
   any offset and strides;
5. no size, stride or offset computation overflows `long` (`Math.*Exact`);
6. the buffer's memory is alive at creation;
7. an array with a zero stride on an axis longer than 1 is read-only.

Because of (4), computing the extent once (O(rank)) replaces a range check on every
element access: accessors only check each index against its axis length.

## Element access

ADR-002 defines the accessors: `getDouble`, `getLong`, `getBoolean` and their setters, for
rank 0 to 3 with one argument per index, plus a `long[]` form for any rank. They
allocate nothing (measured: 0 B/op in JMH), use constant unaligned value layouts, and
convert values by the table in the `NDArray` Javadoc. Indices must be in `[0, dim)`;
negative indices are only interpreted by `slice`.

## Views

Views share the buffer and change only metadata. All of them run in O(rank):

| Operation | NumPy | Effect on metadata |
| --- | --- | --- |
| `transpose()` | `a.T` | reverse shape and strides |
| `permuteDims(axes)` | `np.permute_dims` | permute shape and strides |
| `swapAxes(a, b)` | `np.swapaxes` | swap two axes |
| `slice(Slice...)` | `a[...]` | see below |
| `reshapeView(shape)` | `a.reshape` when it returns a view | new strides by NumPy's no-copy rule, else `ShapeException` |
| `reshape(shape)` | `a.reshape` | view if possible, otherwise a C-contiguous copy |
| `broadcastTo(shape)` | `np.broadcast_to` | stride 0 on stretched and new axes; always read-only |

### Slicing

`slice` takes `Slice` components, mirroring NumPy's basic indexing:

| NumPy | NumJVM | Metadata |
| --- | --- | --- |
| `a[2]` | `Slice.index(2)` | offset += 2·stride, axis removed |
| `a[1:7:2]` | `Slice.range(1, 7, 2)` | offset += 1·stride, length 3, stride·2 |
| `a[::-1]` | `Slice.step(-1)` | offset += (n−1)·stride, stride negated |
| `a[None]` | `Slice.newAxis()` | new axis of length 1, stride 0 |
| `a[..., 0]` | `Slice.ellipsis(), Slice.index(0)` | ellipsis fills the remaining axes |

Range bounds follow Python's `PySlice_AdjustIndices`: negative positions count from the
end and out-of-range bounds are clipped. A NumPy-generated table checks all 864
combinations of bounds and steps on a length-7 axis.

### Reshape without copying

`reshapeView` ports NumPy's `_attempt_nocopy_reshape`. Axes of length 1 are dropped;
then groups of old axes are matched with groups of new axes that have the same product.
Within a group, each old axis must step exactly over the next one
(`stride[k] == dim[k+1] * stride[k+1]`). New strides are derived from the group's last
old stride. So a C-contiguous array can take any shape, while `a.T.reshape(-1)` must
copy. For 610 (view, shape) pairs NumJVM returns a view exactly when NumPy does, with the
same strides.

### Shared memory

`sharesMemoryWith` compares the byte extents of two arrays in the same memory
(`np.may_share_memory`): exact for blocks, conservative for interleaved views
(`a[::2]` and `a[1::2]` report `true`).

## Copies

`copy()` and `copy(Order)` produce a new contiguous, writeable array in automatic memory;
`astype(DType)` converts while copying (NumPy's unsafe casting; complex to real is
refused). When the source is already contiguous in the target order the copy is one
bulk `MemorySegment.copy`; otherwise it walks both arrays with `StridedIterator`.

## Iteration

`StridedIterator` (ADR-003) walks any number of operands over a common shape. It
advances the outer axes like an odometer and hands the innermost axis to the caller as
an offset, a stride and a length per operand, so kernels are plain counted loops:

```java
StridedIterator it = StridedIterator.of(shape, new long[][] {sa, sb, so}, new long[] {oa, ob, oo});
do {
    long pa = it.offset(0), pb = it.offset(1), po = it.offset(2);
    for (long k = 0, n = it.innerSize(); k < n; k++, pa += it.innerStride(0), ...) { ... }
} while (it.next());
```

It allocates a few small arrays when created and nothing per element or per run.

## Printing

`toString()` reproduces NumPy 2's `repr` with default print options (line width 75,
summarisation above 1000 elements, 8 significant fractional digits, shortest round-trip
digits per dtype, scientific notation by NumPy's thresholds). The oracle compares 12
reprs character for character, and the unit tests embed further NumPy reference output.
