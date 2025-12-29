package org.numjvm.ufunc;

import org.numjvm.NJ;
import org.numjvm.dtype.Complex;
import org.numjvm.dtype.DType;
import org.numjvm.exception.ValueException;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.internal.memory.HybridAllocator;
import org.numjvm.internal.memory.MemoryAllocator;
import org.numjvm.iteration.IterationPlan;
import org.numjvm.memory.NJScope;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Order;
import org.numjvm.ndarray.Shape;
import org.numjvm.shape.Broadcasting;

/**
 * Operand preparation shared by all ufuncs: weak scalars, conversions and output allocation.
 */
final class Operands {

    private Operands() {
    }

    /**
     * Where an operation's result and temporaries come from (T4.7): the scope of the first
     * operand that lives in an open {@link NJScope}, otherwise automatic memory. Operations on
     * scoped arrays therefore leave nothing behind when the scope closes; use
     * {@link NJScope#keep(NDArray)} to keep a result.
     */
    static MemoryAllocator allocator(NDArray... operands) {
        for (NDArray a : operands) {
            if (a.memoryOwner() instanceof NJScope scope && scope.isAlive()) {
                return SharedSecrets.scopeAccess().allocator(scope);
            }
        }
        return HybridAllocator.defaultInstance();
    }

    /**
     * A Java {@code long} as a 0-D array of the promoted dtype {@code t}. An integer dtype
     * must hold the value exactly (NumPy raises {@code OverflowError}).
     */
    static NDArray weakLong(DType t, long v) {
        if (t.isInteger() && !fits(t, v)) {
            throw new ValueException("Java integer " + v + " is out of bounds for " + t.typeName());
        }
        NDArray s = NJ.zeros(t);
        s.setLong(v);
        return s;
    }

    /**
     * A Java {@code double} as a 0-D array of the promoted dtype {@code t}.
     */
    static NDArray weakDouble(DType t, double v) {
        NDArray s = NJ.zeros(t);
        s.setDouble(v);
        return s;
    }

    /**
     * A Java {@link Complex} as a 0-D array of the promoted complex dtype {@code t}.
     */
    static NDArray weakComplex(DType t, Complex v) {
        NDArray s = NJ.zeros(t);
        s.setComplex(java.util.Objects.requireNonNull(v, "scalar"));
        return s;
    }

    /**
     * Tells whether integer dtype {@code t} holds {@code v} exactly.
     */
    static boolean fits(DType t, long v) {
        if (t == DType.UINT64) {
            return v >= 0;
        }
        int bits = t.size() * 8;
        if (bits == 64) {
            return true;
        }
        if (t.isUnsigned()) {
            return v >= 0 && v < (1L << bits);
        }
        long max = (1L << (bits - 1)) - 1;
        return v >= -max - 1 && v <= max;
    }

    /**
     * {@code a} converted to {@code t}, or {@code a} itself if it already has that dtype.
     */
    static NDArray cast(NDArray a, DType t, MemoryAllocator allocator) {
        return a.dtype() == t ? a : SharedSecrets.arrayAccess().convert(a, allocator, t, Order.C);
    }

    /**
     * A new array of {@code shape} laid out in the memory order of the inputs (NumPy's
     * {@code order='K'}): C order unless the inputs agree on another order.
     */
    static NDArray allocate(MemoryAllocator allocator, DType dtype, Shape shape, NDArray... inputs) {
        long[] dims = shape.dimensions();
        long[][] strides = new long[inputs.length][];
        for (int i = 0; i < inputs.length; i++) {
            strides[i] = Broadcasting.broadcastStrides(inputs[i].shape(), inputs[i].strides(), shape).values();
        }
        int[] order = IterationPlan.memoryOrder(dims, strides);
        boolean identity = true;
        for (int i = 0; i < order.length; i++) {
            identity &= order[i] == i;
        }
        SharedSecrets.ArrayAccess access = SharedSecrets.arrayAccess();
        if (identity) {
            return access.allocate(allocator, dtype, shape, Order.C);
        }
        long[] permuted = new long[dims.length];
        int[] inverse = new int[dims.length];
        for (int i = 0; i < order.length; i++) {
            permuted[i] = dims[order[i]];
            inverse[order[i]] = i;
        }
        return access.allocate(allocator, dtype, Shape.of(permuted), Order.C).permuteDims(inverse);
    }
}
