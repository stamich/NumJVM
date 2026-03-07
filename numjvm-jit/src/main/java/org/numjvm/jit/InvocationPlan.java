package org.numjvm.jit;

import java.util.Arrays;
import java.util.Objects;
import org.numjvm.dtype.DType;
import org.numjvm.exception.DTypeException;
import org.numjvm.exception.ReadOnlyArrayException;
import org.numjvm.exception.ShapeException;
import org.numjvm.internal.ExpressionSemantics;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.iteration.IterationPlan;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Order;
import org.numjvm.ndarray.Shape;
import org.numjvm.shape.Broadcasting;

/** Validated, invocation-local metadata. Never stored in the compiled cache. */
final class InvocationPlan {
    private final NDArray[] inputs;
    private final LayoutResolver.Layout layout;
    private final DType dtype;

    InvocationPlan(Expression expression, NDArray[] arguments) {
        Objects.requireNonNull(arguments, "arguments");
        if (arguments.length != expression.arity()) throw new IllegalArgumentException("incorrect input count");
        inputs = arguments.clone();
        ExpressionSemantics.accessible(inputs);
        DType[] types = new DType[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            types[i] = inputs[i].dtype();
            if (inputs[i].shape().dimensions().length > 32) throw new ShapeException("expression rank exceeds 32");
        }
        TypeResolver resolver = new TypeResolver(types);
        dtype = resolver.resolve(expression.root());
        layout = new LayoutResolver(inputs, resolver).resolve(expression.root());
    }

    Shape shape() { return layout.shape(); }
    DType dtype() { return dtype; }

    /** A non-identical overlapping leaf needs a full result before copyback. */
    boolean needsTemporary(NDArray output) {
        var access = SharedSecrets.arrayAccess();
        for (NDArray input : inputs) {
            if (!input.sharesMemoryWith(output)) continue;
            if (input.dtype() != output.dtype()
                    || access.buffer(input).segment() != access.buffer(output).segment()
                    || access.offset(input) != access.offset(output)
                    || !Arrays.equals(Broadcasting.broadcastStrides(input.shape(), input.strides(), shape()).values(),
                                      output.strides().values())) return true;
        }
        return false;
    }

    NDArray copyBack(NDArray temporary, NDArray output) {
        checkOutput(output);
        return ExpressionSemantics.copyInto(temporary, output);
    }

    void checkOutput(NDArray output) {
        ExpressionSemantics.accessible(output);
        if (!output.shape().equals(shape())) throw new ShapeException("output shape does not match expression");
        if (output.dtype() != dtype) throw new DTypeException("output dtype does not match expression");
        if (!output.isWriteable()) throw new ReadOnlyArrayException("output array is read-only");
        if (!isNonOverlapping(output)) throw new ShapeException("output elements may overlap");
    }

    NDArray allocate() {
        return allocateWith(ExpressionSemantics.allocator(inputs));
    }

    NDArray allocateWith(org.numjvm.internal.memory.MemoryAllocator allocator) {
        long[] dims = shape().dimensions();
        int[] order = IterationPlan.memoryOrder(dims, new long[][]{layout.strides().values()});
        long[] permuted = new long[dims.length];
        int[] inverse = new int[dims.length];
        for (int i = 0; i < order.length; i++) { permuted[i] = dims[order[i]]; inverse[order[i]] = i; }
        return SharedSecrets.arrayAccess().allocate(allocator, dtype,
                Shape.of(permuted), Order.C).permuteDims(inverse);
    }

    private static boolean isNonOverlapping(NDArray array) {
        if (array.shape().size() == 0) return true;
        long[] dims = array.shape().dimensions(), strides = array.strides().values();
        Integer[] axes = new Integer[dims.length];
        for (int i = 0; i < axes.length; i++) {
            if (strides[i] == Long.MIN_VALUE) return false;
            axes[i] = i;
        }
        Arrays.sort(axes, (a, b) -> Long.compare(Math.abs(strides[a]), Math.abs(strides[b])));
        long extent = array.dtype().size();
        try {
            for (int axis : axes) {
                if (dims[axis] <= 1) continue;
                long stride = Math.abs(strides[axis]);
                if (stride < extent) return false;
                extent = Math.addExact(extent, Math.multiplyExact(dims[axis] - 1, stride));
            }
            return true;
        } catch (ArithmeticException ex) { return false; }
    }
}
