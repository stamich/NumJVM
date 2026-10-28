# ADR-004: Iteration plans

- Status: Accepted
- Date: 2025-11-30
- Extends: [ADR-003](ADR-003-iterator-shape.md) (the external-inner-loop iterator stays)
- Evidence: [`docs/benchmarks/M0.md`](../benchmarks/M0.md), [`docs/milestones/M0.md`](../milestones/M0.md)

## Context

M0 traverses arrays in the order of their logical axes. That is optimal for C-contiguous operands and poor for every
other layout: summing a transposed 1M-element float64 array was 13× slower than NumPy, against 2× for a contiguous one,
because the inner loop stepped through memory by a whole row. Every M1 ufunc and reduction would inherit the problem.

NumPy's `nditer` avoids it by preparing the traversal before running it: it sorts the axes by stride so that the
innermost loop walks memory sequentially, and it merges ("coalesces") axes that are contiguous with each other, so that
a C- or F-contiguous array of any rank becomes one long inner loop. `nditer` also flips negative strides and buffers
operands that need a dtype cast; those parts are large and matter less.

## Decision

1. **Every elementwise operation and reduction runs from an `IterationPlan`.** A plan is built from the common
   (broadcast) shape and, per operand, byte strides already broadcast to it (stride 0 on stretched axes) and a byte
   offset. Building it takes four steps:

   1. **Empty and 0-D shapes**: a shape with a zero-length axis gives a plan with no runs; a 0-D shape gives one run
      of length 1.
   2. **Drop length-1 axes.** They contribute nothing to any offset.
   3. **Reorder axes by stride**, with NumPy's rule (`npyiter_find_best_axis_ordering`): an insertion sort that moves
      an axis inwards when its absolute stride is smaller, for every operand whose strides on both axes are non-zero.
      Zero strides (broadcast operands, reduction outputs) do not vote. When operands disagree, C order wins, so the
      plan never reorders arrays that are all C-contiguous.
   4. **Coalesce** adjacent axes `i` (outer) and `i + 1` when, for every operand,
      `stride[i] == stride[i + 1] * dim[i + 1]`. The merged axis has length `dim[i] * dim[i + 1]` and the inner stride.

   The plan hands its shape and strides to `StridedIterator` (ADR-003), unchanged, so every caller keeps the same
   external inner loop.

2. **Inner loops have specialised paths, chosen per run** by comparing the run's strides with the element sizes:
   - all operands contiguous (stride = element size): indexed loop, the JIT's best case;
   - one input broadcast along the run (stride 0) and the others contiguous: the scalar is read once;
   - anything else: the general strided loop.

   Because coalescing turns any contiguous operand set into a single run, the contiguous path covers C- and
   F-contiguous arrays, transposed views of contiguous arrays, and reductions over all axes.

3. **Reductions use the same plan.** The output gets stride 0 on reduced axes, which then never vote in step 3: the
   input decides the order, so a reduction over a transposed array still reads memory sequentially.

4. **Not adopted from `nditer` in M1:**
   - flipping negative strides (reversed views run backwards, which hardware prefetchers handle);
   - buffering. An operand whose dtype differs from the loop's dtype is converted to a temporary contiguous array
     before the plan is built; a broadcast operand is converted at its own, unbroadcast size.

## Consequences

- Positive: the layout of an operand no longer decides its speed as long as each operand is contiguous in some axis
  order. The M1 exit criterion — transposed and broadcast operations within 2× of contiguous — becomes reachable without
  SIMD work.
- Positive: kernels shrink to inner loops over one axis; all multi-dimensional logic lives in the plan, which is tested
  once, with property tests over random shapes, strides and broadcasts.
- Positive: summation order follows memory order. Pairwise summation of a transposed array therefore adds the same
  elements in the same order as for its contiguous original; results may differ from NumPy's in the last bits for
  non-contiguous layouts (NumPy buffers in chunks of 8,192 elements), which the oracle tolerates for reductions.
- Negative: operands with mismatched dtypes cost a full temporary copy; a buffered cast (NumPy's approach) would bound
  that memory. Revisit in M2 together with the Vector API.
- Negative: building a plan allocates a few small arrays per operation. For tiny arrays this is visible next to the
  arithmetic; scopes and `out=` arrays remove the larger cost, the result allocation.
