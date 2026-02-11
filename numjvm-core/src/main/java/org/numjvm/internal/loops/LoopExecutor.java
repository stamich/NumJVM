package org.numjvm.internal.loops;

import java.lang.foreign.MemorySegment;

import org.numjvm.exception.ReadOnlyArrayException;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.internal.memory.MemoryErrors;
import org.numjvm.iteration.IterationPlan;
import org.numjvm.iteration.StridedIterator;
import org.numjvm.memory.MemoryOwner;
import org.numjvm.memory.NJScope;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Shape;
import org.numjvm.shape.Broadcasting;

/**
 * Runs inner loops over arrays: broadcasts the inputs to the output's shape, builds an
 * {@link IterationPlan} for all operands and calls the loop once per run (ADR-004).
 *
 * <p>The caller has already chosen the loop for the operands' dtypes and allocated the
 * output; the executor only moves through memory. FFM lifecycle errors (closed scope,
 * wrong thread) are translated into {@link org.numjvm.exception.MemoryException}s.
 */
public final class LoopExecutor {

    private LoopExecutor() {
    }

    /**
     * Runs {@code loop} with {@code in} broadcast to the shape of {@code out}.
     *
     * @param loop the inner loop, matching both dtypes
     * @param in   input
     * @param out  writeable output
     * @throws org.numjvm.exception.ShapeException if {@code in} cannot be broadcast to {@code out}
     */
    public static void unary(UnaryLoop loop, NDArray in, NDArray out) {
        requireWriteable(out);
        Shape shape = out.shape();
        SharedSecrets.ArrayAccess access = SharedSecrets.arrayAccess();
        IterationPlan plan = IterationPlan.of(shape.dimensions(),
                new long[][]{broadcast(in, shape), out.strides().values()},
                new long[]{access.offset(in), access.offset(out)});
        if (!plan.hasElements()) {
            return;
        }
        MemorySegment mi = access.buffer(in).segment();
        MemorySegment mo = access.buffer(out).segment();
        StridedIterator it = plan.iterator();
        long n = it.innerSize();
        long si = it.innerStride(0);
        long so = it.innerStride(1);
        try {
            do {
                loop.run(mi, it.offset(0), si, mo, it.offset(1), so, n);
            } while (it.next());
        } catch (IllegalStateException | WrongThreadException e) {
            throw MemoryErrors.translate(e, failedOwner(in, out));
        }
    }

    /**
     * Runs {@code loop} with {@code a} and {@code b} broadcast to the shape of {@code out}.
     *
     * @param loop the inner loop, matching the three dtypes
     * @param a    first input
     * @param b    second input
     * @param out  writeable output
     * @throws org.numjvm.exception.ShapeException if an input cannot be broadcast to {@code out}
     */
    public static void binary(BinaryLoop loop, NDArray a, NDArray b, NDArray out) {
        requireWriteable(out);
        Shape shape = out.shape();
        SharedSecrets.ArrayAccess access = SharedSecrets.arrayAccess();
        IterationPlan plan = IterationPlan.of(shape.dimensions(),
                new long[][]{broadcast(a, shape), broadcast(b, shape), out.strides().values()},
                new long[]{access.offset(a), access.offset(b), access.offset(out)});
        if (!plan.hasElements()) {
            return;
        }
        MemorySegment ma = access.buffer(a).segment();
        MemorySegment mb = access.buffer(b).segment();
        MemorySegment mo = access.buffer(out).segment();
        StridedIterator it = plan.iterator();
        long n = it.innerSize();
        long sa = it.innerStride(0);
        long sb = it.innerStride(1);
        long so = it.innerStride(2);
        try {
            do {
                loop.run(ma, it.offset(0), sa, mb, it.offset(1), sb, mo, it.offset(2), so, n);
            } while (it.next());
        } catch (IllegalStateException | WrongThreadException e) {
            throw MemoryErrors.translate(e, failedOwner(a, b, out));
        }
    }

    /**
     * Reduces {@code in} into {@code out} with a binary loop: for every element of
     * {@code in}, {@code out[j] = op(out[j], in[i])}, where {@code j} is {@code i} with the
     * reduced axes dropped. {@code outStrides} are the output's byte strides expressed over the
     * input's shape: 0 on reduced axes. The loop sees the output as its first input and as its
     * output with the same offsets and strides; along a run over a reduced axis both strides
     * are 0, which the generated loops recognise and accumulate in a register.
     *
     * @param loop       a binary loop whose inputs and output have {@code out}'s dtype
     * @param out        writeable output, already initialised (identity or first elements)
     * @param outStrides byte strides of {@code out} over the input's shape
     * @param in         the input, of {@code out}'s dtype
     */
    public static void reduce(BinaryLoop loop, NDArray out, long[] outStrides, NDArray in) {
        reduce(loop, out, outStrides, in, null);
    }

    /**
     * Reduces with an optional virtual cast layout; its loop maps virtual offsets to source reads.
     * @param loop resolved mixed input/accumulator loop
     * @param out initialized writable output
     * @param outStrides output strides over the input shape
     * @param in physical input storage and logical shape
     * @param virtualStrides cast-layout strides, or null for physical strides
     */
    public static void reduce(BinaryLoop loop, NDArray out, long[] outStrides, NDArray in, long[] virtualStrides) {
        requireWriteable(out);
        SharedSecrets.ArrayAccess access = SharedSecrets.arrayAccess();
        long oo = access.offset(out);
        IterationPlan plan = IterationPlan.of(in.shape().dimensions(),
                new long[][]{outStrides, virtualStrides == null ? in.strides().values() : virtualStrides, outStrides},
                new long[]{oo, virtualStrides == null ? access.offset(in) : 0, oo});
        if (!plan.hasElements()) {
            return;
        }
        MemorySegment mo = access.buffer(out).segment();
        MemorySegment mi = access.buffer(in).segment();
        StridedIterator it = plan.iterator();
        long n = it.innerSize();
        long so = it.innerStride(0);
        long si = it.innerStride(1);
        try {
            do {
                loop.run(mo, it.offset(0), so, mi, it.offset(1), si, mo, it.offset(2), so, n);
            } while (it.next());
        } catch (IllegalStateException | WrongThreadException e) {
            throw MemoryErrors.translate(e, failedOwner(in, out));
        }
    }

    static long[] broadcast(NDArray a, Shape shape) {
        return a.shape().equals(shape)
                ? a.strides().values()
                : Broadcasting.broadcastStrides(a.shape(), a.strides(), shape).values();
    }

    static void requireWriteable(NDArray out) {
        if (!out.isWriteable()) {
            throw new ReadOnlyArrayException("the output array is read-only");
        }
    }

    /**
     * The owner to blame for a lifecycle error: the first operand whose memory is no
     * longer alive, or the first one when all are (a thread-confinement error).
     */
    static MemoryOwner failedOwner(NDArray... arrays) {
        for (NDArray a : arrays) {
            if (!a.isAlive()) {
                return a.memoryOwner();
            }
        }
        for (NDArray a : arrays) {
            if (a.memoryOwner() instanceof NJScope) {
                return a.memoryOwner();
            }
        }
        return arrays[0].memoryOwner();
    }
}
