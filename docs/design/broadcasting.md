# Broadcasting

Broadcasting lets operations combine arrays of different but compatible shapes without copying data (`ARRAY_API`).

## Rules

Shapes are aligned at their last axis. Missing leading axes count as length 1. Two lengths are compatible when they are
equal or one of them is 1; the result takes the other one. Zero-length axes broadcast only with 1 or 0.

```text
A      (4d array):  8 x 1 x 6 x 1
B      (3d array):      7 x 1 x 5
Result (4d array):  8 x 7 x 6 x 5
```

| A           | B           | Result           |
|-------------|-------------|------------------|
| `[3]`       | `[3]`       | `[3]`            |
| `[3, 1]`    | `[1, 4]`    | `[3, 4]`         |
| `[5, 1, 4]` | `[1, 3, 1]` | `[5, 3, 4]`      |
| `[]`        | `[2, 3]`    | `[2, 3]`         |
| `[0, 3]`    | `[1, 3]`    | `[0, 3]`         |
| `[2, 3]`    | `[4, 3]`    | `ShapeException` |

`Broadcasting.broadcastShapes(Shape...)` takes any number of shapes; the error message names the two arguments that
conflict, as NumPy's does.

## Broadcast views

`Broadcasting.broadcastStrides(shape, strides, target)` gives stride 0 to every new or stretched axis;
`NDArray.broadcastTo(target)` returns the corresponding view:

```java
NDArray row = NJ.array(new double[] {1, 2, 3}, 1, 3);
NDArray b = row.broadcastTo(4, 3);   // shape [4, 3], strides [0, 8], 3 elements in memory
```

Broadcast views are **always read-only**. With a zero stride, one physical element stands for many logical ones, so a
write would change a whole column at once. Writing throws
`ReadOnlyArrayException`; call `copy()` for a writeable array. The array invariants enforce this: no writeable array may
have a zero stride on an axis longer than 1.

## Use in kernels

Operations broadcast their operands before iterating. `ValidationKernels.add` computes the common shape, broadcasts both
operands' strides to it, and drives a three-operand
`StridedIterator`; operands with stride 0 simply re-read the same element. Nothing is materialised.
