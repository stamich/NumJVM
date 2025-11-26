# ADR-003: Iterator shape

- Status: Accepted
- Date: 2025-11-02
- Evidence: [`architecture-spike/results.md`](../../architecture-spike/results.md), sections S2, S3, S5

## Context

Every elementwise operation and reduction has to traverse arrays of any rank, with positive, negative and zero
(broadcast) strides, over several operands at once. The two common designs are:

- **Element iterator**: `next()` per element returning an offset or index. Simple, but costs a call and a multi-axis
  carry check per element.
- **External inner loop** (NumPy's `NPY_ITER_EXTERNAL_LOOP`): the iterator walks the outer axes; the caller runs a plain
  counted loop over the innermost axis with per-operand (offset, stride, count).

The spike implemented the second (`Odometer`). It traversed a transposed 256×256 array at 1.44 ns/element, allocating
120 bytes per traversal and none per element. The generic odometer `add` was within 1.0–1.6× of a dedicated contiguous
loop (S3, S5).

## Decision

1. **`StridedIterator` uses the external-inner-loop design.** It takes the broadcast shape and, per operand, the byte
   strides already broadcast to that shape and the base byte offset. It exposes:

   ```java
   long innerSize();          // length of the innermost axis (1 for 0-D)
   long innerStride(int op);  // byte stride of operand op on that axis
   long offset(int op);       // byte offset of operand op at the start of the run
   boolean next();            // advance to the next run; false after the last
   ```

   Usage:

   ```java
   var it = StridedIterator.of(shape, strides, offsets);
   if (it.hasElements()) do {
       long pa = it.offset(0), pb = it.offset(1);
       long da = it.innerStride(0), db = it.innerStride(1);
       for (long k = 0, n = it.innerSize(); k < n; k++, pa += da, pb += db) { ... }
   } while (it.next());
   ```

2. **Allocation happens once per traversal** (the counter and offset arrays), never per element or per run. Offsets are
   updated incrementally when an outer axis carries.
3. **0-D arrays produce one run of length 1; arrays with a zero-length axis produce none** (`hasElements() == false`).
4. **Kernels are written as a contiguous fast path plus the generic loop.** No dimension coalescing or loop reordering
   in M0.
5. `IndexIterator` (logical indices, reused `long[]`) exists as a simple reference for tests and for code where clarity
   matters more than speed. Kernels do not use it.

## Consequences

- Positive: the per-element body is a counted loop the JIT can unroll and, in M2, vectorise; the same iterator serves 1,
  2 or N operands.
- Negative: operands with a large inner stride (a transposed operand) run 1.5–4× slower than contiguous ones at ≥1M
  elements (S3). M1 addresses this with dimension coalescing (merge axes that are contiguous across their boundary) and
  loop reordering (choose the innermost axis by smallest stride), both of which fit behind the same interface.
- Negative: the caller writes the inner loop. That is intended: it keeps the value-layout constants and the arithmetic
  visible to the JIT.
