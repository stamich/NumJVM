# Real dot and vecdot

The optional `numjvm-linalg` module provides experimental `Linalg.dot` and
`Linalg.vecdot` for float32 and float64 arrays. Mixed inputs produce float64.
Integer and complex inputs are not supported by these operations.

```java
var a = NJ.array(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
var weights = NJ.array(new double[]{2, 3, 4});
var dot = Linalg.dot(a, weights);              // [20, 47]
var out = NJ.zeros(DType.FLOAT64, 2);
Linalg.vecdot(a, weights, -1, out);           // [20, 47]
```

Dot uses NumPy's scalar multiplication and last/second-last-axis contraction
rules, including higher-rank inputs. Vecdot contracts the specified axis in
each operand, broadcasting the remaining axes. Its default axis is `-1`.
For example, dot of shapes `(2,3,4)` and `(5,4,6)` produces `(2,3,5,6)`;
vecdot of `(2,3)` and `(3,)` produces `(2,)`. Matmul instead applies its
documented vector-promotion and broadcast-batch rules. Empty contractions
produce zeros. Real inputs do not require conjugation.

An allocating result belongs to the first scoped operand when one exists;
otherwise it uses automatic storage. Explicit outputs must have exactly the
resolved dtype and shape, be writable and contain nonoverlapping elements.
If an output overlaps either input, the computation uses independent temporary
storage and copies back. Unlike elementwise fusion, an identical view is not
automatically safe for a contraction. Validation precedes writes; an unexpected
execution failure can leave a direct output partially written.

Java is the portable implementation. Contiguous homogeneous vector pairs may
use the optional Vector API or LP64 OpenBLAS DOT. Compatible rank-two dot can
reuse matmul/GEMM. Unsupported types/layouts, negative strides, unavailable
providers and small inputs use Java where applicable. `Linalg.nativeProvider()`
reports provider availability; `Linalg.lastBackend()` reports the calling
thread's actual selection. Reduction order differs among backends: finite values
are tested with scale-aware bounds, while NaN, infinity, signed zero and overflow
have separate expectations. Details are in [ADR-009](../adr/ADR-009-contractions.md).

Run `./gradlew :examples:contractions:run` for the portable path, or add
`-PwithVector` to enable the Vector API in the example. Reference performance
claims remain open; see [the protocol](../benchmarks/M3-1-protocol.md).
