package org.numjvm.iteration;

import java.util.Arrays;

import org.numjvm.Experimental;
import org.numjvm.exception.ShapeException;

/**
 * A traversal of one or more strided operands, simplified so that the inner loop walks
 * memory as sequentially as the operands allow (ADR-004).
 *
 * <p>A plan is built from a common shape and, per operand, byte strides already broadcast
 * to that shape (stride 0 on stretched axes) plus a byte offset. Building it
 * <ol>
 *   <li>drops axes of length 1,</li>
 *   <li>reorders the remaining axes so that the axis with the smallest stride is innermost
 *       (NumPy's best-axis-ordering rule: an axis moves inwards when its absolute stride is
 *       smaller for every operand with non-zero strides on both axes; C order wins when
 *       operands disagree), and</li>
 *   <li>merges adjacent axes that are contiguous with each other for every operand.</li>
 * </ol>
 * The result visits exactly the same elements, each once, with the same per-operand byte
 * offsets as a C-order traversal of the original shape; only the order differs. A
 * C-contiguous, F-contiguous or transposed array becomes a single run.
 *
 * <pre>{@code
 * IterationPlan plan = IterationPlan.of(shape, new long[][] {stridesA, stridesOut}, new long[] {offA, 0});
 * StridedIterator it = plan.iterator();
 * if (it.hasElements()) {
 *     do { ... inner loop over it.innerSize() elements ... } while (it.next());
 * }
 * }</pre>
 *
 * <p>Plans are immutable; {@link #iterator()} returns a new iterator each time.
 */
@Experimental
public final class IterationPlan {

    private final int nops;
    private final long[] dims;
    private final long[][] strides;
    private final long[] offsets;
    private final long size;

    private IterationPlan(long[] dims, long[][] strides, long[] offsets, long size) {
        this.nops = strides.length;
        this.dims = dims;
        this.strides = strides;
        this.offsets = offsets;
        this.size = size;
    }

    /**
     * Builds a plan.
     *
     * @param shape   the common shape
     * @param strides byte strides per operand, each of the same rank as {@code shape}
     * @param offsets byte offset of element {@code [0, ..., 0]} per operand
     * @return the plan
     * @throws ShapeException if ranks or operand counts disagree, or a dimension is negative
     */
    public static IterationPlan of(long[] shape, long[][] strides, long[] offsets) {
        int nops = strides.length;
        if (offsets.length != nops) {
            throw new ShapeException(nops + " stride arrays but " + offsets.length + " offsets");
        }
        if (nops == 0) {
            throw new ShapeException("an iteration plan needs at least one operand");
        }
        int rank = shape.length;
        boolean empty = false;
        for (long d : shape) {
            if (d < 0) {
                throw new ShapeException("negative dimension in " + Arrays.toString(shape));
            }
            empty |= d == 0;
        }
        long size = 1;
        if (empty) {
            size = 0;
        } else {
            for (long d : shape) {
                try {
                    size = Math.multiplyExact(size, d);
                } catch (ArithmeticException e) {
                    throw new ShapeException("shape " + Arrays.toString(shape) + " has more than 2^63 - 1 elements");
                }
            }
        }
        for (int op = 0; op < nops; op++) {
            if (strides[op].length != rank) {
                throw new ShapeException("operand " + op + " has strides " + Arrays.toString(strides[op])
                        + " for shape " + Arrays.toString(shape));
            }
        }
        if (size == 0) {
            long[][] st = new long[nops][1];
            return new IterationPlan(new long[]{0}, st, offsets.clone(), 0);
        }

        // 1. keep axes longer than 1, in C order
        int[] kept = new int[rank];
        int n = 0;
        for (int d = 0; d < rank; d++) {
            if (shape[d] != 1) {
                kept[n++] = d;
            }
        }

        // 2. perm[0] is the innermost axis, perm[n - 1] the outermost (NumPy's convention)
        int[] perm = new int[n];
        for (int i = 0; i < n; i++) {
            perm[i] = kept[n - 1 - i];
        }
        sortAxes(perm, strides);

        // 3. coalesce, walking from the outermost axis inwards
        long[] outDims = new long[n];
        long[][] outStrides = new long[nops][n];
        int m = 0;
        for (int i = n - 1; i >= 0; i--) {
            int ax = perm[i];
            if (m > 0 && contiguous(outDims[m - 1], outStrides, m - 1, shape[ax], strides, ax)) {
                outDims[m - 1] *= shape[ax];
                for (int op = 0; op < nops; op++) {
                    outStrides[op][m - 1] = strides[op][ax];
                }
            } else {
                outDims[m] = shape[ax];
                for (int op = 0; op < nops; op++) {
                    outStrides[op][m] = strides[op][ax];
                }
                m++;
            }
        }
        long[] finalDims = Arrays.copyOf(outDims, m);
        long[][] finalStrides = new long[nops][];
        for (int op = 0; op < nops; op++) {
            finalStrides[op] = Arrays.copyOf(outStrides[op], m);
        }
        return new IterationPlan(finalDims, finalStrides, offsets.clone(), size);
    }

    /**
     * Returns the order in which the axes of {@code shape} are laid out in memory by these
     * operands, outermost first: the axis order an iteration plan would choose, before
     * dropping length-1 axes and coalescing. A new array allocated in this order (NumPy's
     * {@code order='K'}) lets every operation on these operands write sequentially.
     *
     * @param shape   the common shape
     * @param strides byte strides per operand, broadcast to {@code shape}
     * @return a permutation of {@code 0 .. shape.length - 1}; the identity for C-contiguous
     *         operands and whenever they disagree
     */
    public static int[] memoryOrder(long[] shape, long[][] strides) {
        int n = shape.length;
        int[] perm = new int[n];
        for (int i = 0; i < n; i++) {
            perm[i] = n - 1 - i;
        }
        long[][] st = new long[strides.length][];
        for (int op = 0; op < strides.length; op++) {
            if (strides[op].length != n) {
                throw new ShapeException("operand " + op + " has strides " + Arrays.toString(strides[op])
                        + " for shape " + Arrays.toString(shape));
            }
            // length-1 axes carry no layout information
            st[op] = strides[op].clone();
            for (int d = 0; d < n; d++) {
                if (shape[d] == 1) {
                    st[op][d] = 0;
                }
            }
        }
        sortAxes(perm, st);
        int[] outerFirst = new int[n];
        for (int i = 0; i < n; i++) {
            outerFirst[i] = perm[n - 1 - i];
        }
        return outerFirst;
    }

    /**
     * Whether the previous (outer) plan axis can absorb axis {@code ax} as its new inner
     * part: for every operand, outer stride == inner stride * inner length.
     */
    private static boolean contiguous(long outerDim, long[][] outStrides, int outer, long innerDim, long[][] strides,
                                      int ax) {
        for (int op = 0; op < strides.length; op++) {
            if (outStrides[op][outer] != strides[op][ax] * innerDim) {
                return false;
            }
        }
        return outerDim > 0;
    }

    /**
     * NumPy's {@code npyiter_find_best_axis_ordering}: insertion sort of the axes, innermost
     * first, by absolute stride. An axis moves inwards past another only if some operand
     * asks for it and none objects; zero strides do not vote.
     */
    private static void sortAxes(int[] perm, long[][] strides) {
        for (int i0 = 1; i0 < perm.length; i0++) {
            int ax0 = perm[i0];
            int pos = i0;
            for (int i1 = i0 - 1; i1 >= 0; i1--) {
                int ax1 = perm[i1];
                boolean ambiguous = true;
                boolean swap = false;
                for (long[] st : strides) {
                    long s0 = st[ax0];
                    long s1 = st[ax1];
                    if (s0 != 0 && s1 != 0) {
                        if (Math.abs(s1) <= Math.abs(s0)) {
                            swap = false;
                        } else if (ambiguous) {
                            swap = true;
                        }
                        ambiguous = false;
                    }
                }
                if (!ambiguous) {
                    if (swap) {
                        pos = i1;
                    } else {
                        break;
                    }
                }
            }
            if (pos != i0) {
                System.arraycopy(perm, pos, perm, pos + 1, i0 - pos);
                perm[pos] = ax0;
            }
        }
    }

    /**
     * Returns the number of operands.
     *
     * @return the number of operands
     */
    public int operands() {
        return nops;
    }

    /**
     * Returns the number of axes after simplification: 0 for a single element, 1 when every
     * operand is traversed in one run.
     *
     * @return the rank of the simplified shape
     */
    public int rank() {
        return dims.length;
    }

    /**
     * Returns the simplified shape, outermost axis first.
     *
     * @return a copy of the simplified shape
     */
    public long[] shape() {
        return dims.clone();
    }

    /**
     * Returns the byte strides of one operand on the simplified axes.
     *
     * @param op operand number
     * @return a copy of the operand's strides
     */
    public long[] strides(int op) {
        return strides[op].clone();
    }

    /**
     * Returns the byte offset of one operand's first element.
     *
     * @param op operand number
     * @return the byte offset
     */
    public long offset(int op) {
        return offsets[op];
    }

    /**
     * Returns the number of elements visited.
     *
     * @return the product of the dimensions
     */
    public long size() {
        return size;
    }

    /**
     * Returns {@code true} unless the shape has a zero-length axis.
     *
     * @return {@code true} if there is at least one element
     */
    public boolean hasElements() {
        return size > 0;
    }

    /**
     * Returns a new iterator over the plan's runs.
     *
     * @return an iterator positioned at the first run
     */
    public StridedIterator iterator() {
        return StridedIterator.of(dims, strides, offsets);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("IterationPlan[shape=").append(Arrays.toString(dims));
        for (int op = 0; op < nops; op++) {
            sb.append(", op").append(op).append(" strides=").append(Arrays.toString(strides[op]))
                    .append(" offset=").append(offsets[op]);
        }
        return sb.append(']').toString();
    }
}
