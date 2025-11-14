package org.numjvm.iteration;

import java.util.Arrays;

import org.numjvm.Experimental;
import org.numjvm.exception.ShapeException;

/**
 * Allocation-free traversal of one or more strided operands with an external inner loop
 * (ADR-003).
 *
 * <p>The iterator walks the outer axes of a common shape like an odometer and hands the
 * innermost axis to the caller as a run: for each operand the byte offset of the run's
 * first element and the byte stride between elements, plus the run length. The caller
 * writes a plain counted loop, which the JIT can unroll and vectorise:
 *
 * <pre>{@code
 * StridedIterator it = StridedIterator.of(shape, new long[][] {stridesA, stridesB}, new long[] {offA, offB});
 * if (it.hasElements()) {
 *     do {
 *         long pa = it.offset(0), pb = it.offset(1);
 *         long da = it.innerStride(0), db = it.innerStride(1);
 *         for (long k = 0, n = it.innerSize(); k < n; k++, pa += da, pb += db) {
 *             // element k of this run: operand 0 at byte pa, operand 1 at byte pb
 *         }
 *     } while (it.next());
 * }
 * }</pre>
 *
 * <p>Every operand's strides must already match the common shape: broadcast operands
 * carry stride 0 on stretched axes (see {@code Broadcasting.broadcastStrides}). The
 * iterator allocates only when it is created; {@link #next()} updates offsets
 * incrementally. A 0-D shape produces one run of length 1; a shape with a zero-length
 * axis produces none.
 */
@Experimental
public final class StridedIterator {

    private final int nops;
    private final long[] dims;
    private final long[][] strides;
    private final long[] counter;
    private final long[] offsets;
    private final long[] innerStrides;
    private final long innerSize;
    private final boolean hasElements;

    private StridedIterator(long[] dims, long[][] strides, long[] offsets) {
        this.nops = strides.length;
        this.dims = dims;
        this.strides = strides;
        this.offsets = offsets;
        int rank = dims.length;
        this.counter = new long[Math.max(0, rank - 1)];
        this.innerSize = rank == 0 ? 1 : dims[rank - 1];
        this.innerStrides = new long[nops];
        for (int op = 0; op < nops; op++) {
            innerStrides[op] = rank == 0 ? 0 : strides[op][rank - 1];
        }
        boolean any = true;
        for (long d : dims) {
            any &= d > 0;
        }
        this.hasElements = any;
    }

    /**
     * Creates an iterator.
     *
     * @param shape   the common shape
     * @param strides byte strides per operand, each of the same rank as {@code shape}
     * @param offsets byte offset of element {@code [0, ..., 0]} per operand
     * @return a new iterator positioned at the first run
     * @throws ShapeException if ranks or operand counts disagree
     */
    public static StridedIterator of(long[] shape, long[][] strides, long[] offsets) {
        if (strides.length != offsets.length) {
            throw new ShapeException(strides.length + " stride arrays but " + offsets.length + " offsets");
        }
        long[][] st = new long[strides.length][];
        for (int op = 0; op < strides.length; op++) {
            if (strides[op].length != shape.length) {
                throw new ShapeException("operand " + op + " has strides " + Arrays.toString(strides[op])
                        + " for shape " + Arrays.toString(shape));
            }
            st[op] = strides[op].clone();
        }
        for (long d : shape) {
            if (d < 0) {
                throw new ShapeException("negative dimension in " + Arrays.toString(shape));
            }
        }
        return new StridedIterator(shape.clone(), st, offsets.clone());
    }

    /** @return {@code false} if the shape has a zero-length axis, so there is nothing to visit */
    public boolean hasElements() {
        return hasElements;
    }

    /** @return the number of operands */
    public int operands() {
        return nops;
    }

    /** @return the length of every run: the innermost dimension, or 1 for a 0-D shape */
    public long innerSize() {
        return innerSize;
    }

    /**
     * @param op operand number
     * @return the byte stride of {@code op} along the innermost axis
     */
    public long innerStride(int op) {
        return innerStrides[op];
    }

    /**
     * @param op operand number
     * @return the byte offset of {@code op}'s first element in the current run
     */
    public long offset(int op) {
        return offsets[op];
    }

    /**
     * Moves to the next run.
     *
     * @return {@code false} after the last run; the iterator is then exhausted
     */
    public boolean next() {
        for (int d = counter.length - 1; d >= 0; d--) {
            if (++counter[d] < dims[d]) {
                for (int op = 0; op < nops; op++) {
                    offsets[op] += strides[op][d];
                }
                return true;
            }
            counter[d] = 0;
            long back = dims[d] - 1;
            for (int op = 0; op < nops; op++) {
                offsets[op] -= back * strides[op][d];
            }
        }
        return false;
    }
}
