package org.numjvm.jit;

import java.util.IdentityHashMap;
import java.util.Map;
import org.numjvm.dtype.DType;
import org.numjvm.iteration.IterationPlan;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Shape;
import org.numjvm.ndarray.Strides;
import org.numjvm.shape.Broadcasting;

/** Infers each eager intermediate's shape and allocation order without allocating arrays. */
final class LayoutResolver {
    record Layout(Shape shape, Strides strides) { }
    private final NDArray[] inputs;
    private final TypeResolver types;
    private final Map<Expr, Layout> layouts = new IdentityHashMap<>();

    LayoutResolver(NDArray[] inputs, TypeResolver types) {
        this.inputs = inputs.clone();
        this.types = types;
    }

    Layout resolve(Expr node) {
        Layout cached = layouts.get(node);
        if (cached != null) return cached;
        Layout result;
        if (node.op == Expr.Op.INPUT) {
            NDArray input = inputs[node.slot];
            result = new Layout(input.shape(), input.strides());
        } else if (node.op == Expr.Op.CONSTANT) {
            result = new Layout(Shape.of(), Strides.of());
        } else {
            Layout a = resolve(node.left);
            Layout b = node.right == null ? null : resolve(node.right);
            Shape shape = b == null ? a.shape() : Broadcasting.broadcastShapes(a.shape(), b.shape());
            long[][] broadcast = new long[b == null ? 1 : 2][];
            broadcast[0] = Broadcasting.broadcastStrides(a.shape(), a.strides(), shape).values();
            if (b != null) broadcast[1] = Broadcasting.broadcastStrides(b.shape(), b.strides(), shape).values();
            result = new Layout(shape, allocatedStrides(shape, types.resolve(node), broadcast));
        }
        layouts.put(node, result);
        return result;
    }

    private static Strides allocatedStrides(Shape shape, DType dtype, long[][] inputs) {
        long[] dimensions = shape.dimensions();
        int[] order = IterationPlan.memoryOrder(dimensions, inputs);
        long[] permuted = new long[dimensions.length];
        for (int i = 0; i < order.length; i++) permuted[i] = dimensions[order[i]];
        long[] physical = Strides.cContiguous(Shape.of(permuted), dtype.size()).values();
        long[] result = new long[order.length];
        for (int i = 0; i < order.length; i++) result[order[i]] = physical[i];
        return Strides.of(result);
    }
}
