package org.numjvm.ufunc;

import org.numjvm.dtype.DType;
import org.numjvm.exception.DTypeException;
import org.numjvm.exception.ShapeException;
import org.numjvm.exception.ValueException;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.internal.loops.BinaryLoop;
import org.numjvm.internal.loops.KernelRegistry;
import org.numjvm.internal.loops.LoopExecutor;
import org.numjvm.internal.loops.UnaryLoop;
import org.numjvm.internal.memory.MemoryAllocator;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Order;
import org.numjvm.ndarray.Shape;
import org.numjvm.ndarray.Slice;

/**
 * Reductions with a binary ufunc's loop (NumPy's {@code ufunc.reduce}).
 *
 * <p>The output has the input's shape with the reduced axes removed (or kept with length 1).
 * Over the input's shape its strides are 0 on reduced axes, so one iteration plan (ADR-004)
 * walks the input in memory order and accumulates into the output. A ufunc with an identity
 * starts from it; one without starts from the first element along the reduced axes and
 * fails on an empty reduction.
 */
final class Reducer {

    private Reducer() {
    }

    /**
     * Normalises an axis list: {@code null} means every axis.
     */
    static boolean[] reducedAxes(int[] axes, int ndim) {
        boolean[] reduced = new boolean[ndim];
        if (axes == null) {
            java.util.Arrays.fill(reduced, true);
            return reduced;
        }
        for (int axis : axes) {
            int a = Shape.normalizeAxis(axis, ndim);
            if (reduced[a]) {
                throw new ShapeException("duplicate value in 'axis': " + java.util.Arrays.toString(axes));
            }
            reduced[a] = true;
        }
        return reduced;
    }

    /**
     * Reduces {@code x}, already converted to the loop's dtype.
     *
     * @param name        ufunc name, for messages
     * @param loop        the binary loop, input and output of {@code x}'s dtype
     * @param x           the input
     * @param reduced     which axes to reduce
     * @param keepdims    keep reduced axes with length 1
     * @param identity    the identity, or {@code null}
     * @param reorderable whether several axes may be reduced at once
     * @param allocator   where the result's memory comes from
     */
    static NDArray reduce(String name, BinaryLoop loop, NDArray x, boolean[] reduced, boolean keepdims, Long identity,
                          boolean reorderable, MemoryAllocator allocator) {
        return reduceAs(name, loop, x, reduced, keepdims, identity, reorderable, allocator, x.dtype());
    }

    /** Reduces source values into a separately resolved accumulator dtype. */
    static NDArray reduceAs(String name, BinaryLoop loop, NDArray x, boolean[] reduced, boolean keepdims,
                            Long identity, boolean reorderable, MemoryAllocator allocator, DType target) {
        return reduceAs(name, loop, x, reduced, keepdims, identity, reorderable, allocator, target, null);
    }

    static NDArray reduceAs(String name, BinaryLoop loop, NDArray x, boolean[] reduced, boolean keepdims,
                            Long identity, boolean reorderable, MemoryAllocator allocator, DType target, NDArray direct) {
        return reduceAs(name, loop, x, reduced, keepdims, identity, reorderable, allocator, target, direct, null);
    }

    static NDArray reduceAs(String name, BinaryLoop loop, NDArray x, boolean[] reduced, boolean keepdims,
                            Long identity, boolean reorderable, MemoryAllocator allocator, DType target,
                            NDArray direct, long[] virtualStrides) {
        int nd = x.ndim();
        long[] dims = x.shape().dimensions();
        int count = 0;
        boolean emptyReduction = false;
        long[] keep = new long[nd];
        for (int d = 0; d < nd; d++) {
            if (reduced[d]) {
                count++;
                emptyReduction |= dims[d] == 0;
                keep[d] = 1;
            } else {
                keep[d] = dims[d];
            }
        }
        if (!reorderable && count > 1) {
            throw new ValueException("reduction operation '" + name + "' is not reorderable, so at most one axis may "
                    + "be specified");
        }
        DType t = target;
        NDArray out;
        if (direct == null) {
            out = SharedSecrets.arrayAccess().allocate(allocator, t, Shape.of(keep), Order.C);
        } else {
            long[] expanded = new long[nd];
            for (int d = 0, j = 0; d < nd; d++) {
                if (keepdims) { expanded[d] = direct.strides().stride(d); }
                else if (!reduced[d]) { expanded[d] = direct.strides().stride(j++); }
            }
            var access = SharedSecrets.arrayAccess();
            out = access.create(access.buffer(direct), t, access.offset(direct), Shape.of(keep),
                    org.numjvm.ndarray.Strides.of(expanded), true);
        }
        if (out.size() > 0) {
            long[] outStrides = out.strides().values();
            for (int d = 0; d < nd; d++) {
                if (reduced[d]) {
                    outStrides[d] = 0;
                }
            }
            UnaryLoop copy = KernelRegistry.unary(x.dtype() == t ? "copy" : "cast:" + t.typeName(), x.dtype());
            if (identity != null) {
                NDArray id = org.numjvm.NJ.zeros(t);
                id.setLong(identity);
                LoopExecutor.unary(KernelRegistry.unary("copy", t), id, out);
                LoopExecutor.reduce(loop, out, outStrides, x, virtualStrides);
            } else {
                if (emptyReduction) {
                    throw new ValueException("zero-size array to reduction operation " + name
                            + " which has no identity");
                }
                Slice[] first = new Slice[nd];
                Slice[] rest = new Slice[nd];
                for (int d = 0; d < nd; d++) {
                    first[d] = reduced[d] ? Slice.range(0, 1) : Slice.all();
                    rest[d] = reduced[d] && count == 1 ? Slice.from(1) : Slice.all();
                }
                LoopExecutor.unary(copy, x.slice(first), out);
                // one axis: reduce the elements after the first; several (only for idempotent
                // ufuncs such as maximum): reducing the first element again changes nothing
                LoopExecutor.reduce(loop, out, outStrides, x.slice(rest));
            }
        }
        if (keepdims) {
            return out;
        }
        long[] dropped = new long[nd - count];
        for (int d = 0, k = 0; d < nd; d++) {
            if (!reduced[d]) {
                dropped[k++] = dims[d];
            }
        }
        return out.reshapeView(dropped);
    }

    static Shape outputShape(NDArray x, boolean[] reduced, boolean keepdims) {
        var dims = new java.util.ArrayList<Long>();
        for (int d = 0; d < reduced.length; d++) {
            if (!reduced[d]) { dims.add(x.dim(d)); }
            else if (keepdims) { dims.add(1L); }
        }
        return Shape.of(dims.stream().mapToLong(Long::longValue).toArray());
    }

    static DTypeException notReducible(String name) {
        return new DTypeException("ufunc '" + name + "' cannot reduce: its output dtype differs from its inputs'");
    }
}
