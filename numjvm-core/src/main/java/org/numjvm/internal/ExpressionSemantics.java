package org.numjvm.internal;

import org.numjvm.dtype.DType;
import org.numjvm.dtype.TypePromotion;
import java.util.Objects;
import org.numjvm.exception.MemoryException;
import org.numjvm.internal.memory.HybridAllocator;
import org.numjvm.internal.memory.MemoryAllocator;
import org.numjvm.memory.NJScope;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Shape;
import org.numjvm.ndarray.Order;
import org.numjvm.iteration.IterationPlan;
import org.numjvm.shape.Broadcasting;
import org.numjvm.ufunc.BinaryUfunc;
import org.numjvm.ufunc.UnaryUfunc;

/** Semantic queries shared with the optional expression compiler. */
public final class ExpressionSemantics {
    private ExpressionSemantics() { }

    /** Resolves a unary result using the same ufunc as eager execution. */
    public static DType unaryResult(UnaryUfunc operation, DType input) {
        return operation.resultType(input);
    }

    /** Resolves an array/array result using eager promotion and loop selection. */
    public static DType binaryResult(BinaryUfunc operation, DType left, DType right) {
        return operation.resultType(left, right);
    }

    /** Resolves a weak-double operand as eager scalar overloads do. */
    public static DType weakDoubleResult(BinaryUfunc operation, DType array) {
        DType common = TypePromotion.resultTypeWithDouble(array);
        return operation.resultType(common, common);
    }

    /** Uses the first live scoped operand, as eager ufuncs do. */
    public static MemoryAllocator allocator(NDArray... operands) {
        for (NDArray operand : operands) {
            if (operand.memoryOwner() instanceof NJScope scope && scope.isAlive()) {
                return SharedSecrets.scopeAccess().allocator(scope);
            }
        }
        return HybridAllocator.defaultInstance();
    }

    /** Checks accessibility before an executor reads or writes any elements. */
    public static void accessible(NDArray... arrays) {
        for (NDArray array : arrays) {
            var segment = SharedSecrets.arrayAccess().buffer(Objects.requireNonNull(array, "array")).segment();
            if (!segment.scope().isAlive()) {
                throw new MemoryException("array scope is closed");
            }
            if (!segment.isAccessibleBy(Thread.currentThread())) {
                throw new MemoryException("array is confined to another thread");
            }
        }
    }

    /** Allocates in eager ufunc order for the broadcast inputs. */
    public static NDArray allocate(MemoryAllocator allocator, DType dtype, Shape shape, NDArray... inputs) {
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
