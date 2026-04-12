package org.numjvm.jit;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.IdentityHashMap;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.internal.ExpressionSemantics;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ufunc.BinaryUfunc;
import org.numjvm.ufunc.Ufuncs;
import org.numjvm.ufunc.UnaryUfunc;

/** Whole-array ufunc strategy; temporary arrays never escape the local scope. */
final class EagerExecutor {
    private EagerExecutor() { }

    /** Returns false without writing if the conservative intermediate budget is exceeded. */
    static boolean tryRun(Expression expression, NDArray[] inputs, NDArray output, long budget) {
        if (budget < 0) throw new IllegalArgumentException("negative temporary budget");
        DType[] inputTypes = new DType[inputs.length];
        for (int i = 0; i < inputs.length; i++) inputTypes[i] = inputs[i].dtype();
        var types = new TypeResolver(inputTypes);
        var layouts = new LayoutResolver(inputs, types);
        long bytes = 0;
        try {
            for (Expr node : expression.graph().nodes()) {
                if (node.op == Expr.Op.CONSTANT) bytes = Math.addExact(bytes, 8);
                else if (node.op != Expr.Op.INPUT && node != expression.root()) {
                    bytes = Math.addExact(bytes, Math.multiplyExact(layouts.resolve(node).shape().size(), types.resolve(node).size()));
                }
                if (node.left != null && node.right != null && node.left.op == Expr.Op.CONSTANT
                        && node.right.op != Expr.Op.CONSTANT) bytes = Math.addExact(bytes, 8);
            }
        } catch (ArithmeticException overflow) { return false; }
        if (bytes > budget) return false;
        try (var scope = NJ.scope()) {
            var allocator = SharedSecrets.scopeAccess().allocator(scope);
            var values = new IdentityHashMap<Expr, NDArray>();
            var remaining = new IdentityHashMap<Expr, Integer>();
            var free = new ArrayDeque<NDArray>();
            for (Expr node : expression.graph().nodes()) remaining.put(node, expression.graph().consumers(node));
            for (Expr node : expression.graph().nodes()) {
                if (node.op == Expr.Op.INPUT) {
                    values.put(node, inputs[node.slot]);
                } else if (node.op == Expr.Op.CONSTANT) {
                    var scalar = scope.zeros(DType.FLOAT64);
                    scalar.setDouble(Double.longBitsToDouble(node.bits));
                    values.put(node, scalar);
                } else {
                    NDArray left = values.get(node.left), right = node.right == null ? null : values.get(node.right);
                    NDArray result;
                    if (node == expression.root()) result = output;
                    else {
                        var layout = layouts.resolve(node);
                        result = null;
                        for (var iterator = free.iterator(); iterator.hasNext();) {
                            NDArray candidate = iterator.next();
                            if (candidate.dtype() == types.resolve(node) && candidate.shape().equals(layout.shape())
                                    && Arrays.equals(candidate.strides().values(), layout.strides().values())) {
                                result = candidate;
                                iterator.remove();
                                break;
                            }
                        }
                        if (result == null) result = node.right == null
                                ? ExpressionSemantics.allocate(allocator, types.resolve(node), layout.shape(), left)
                                : ExpressionSemantics.allocate(allocator, types.resolve(node), layout.shape(), left, right);
                    }
                    if (node.right == null) unary(node.op).apply(left, result);
                    else if (node.right.op == Expr.Op.CONSTANT && node.left.op != Expr.Op.CONSTANT)
                        binary(node.op).apply(left, Double.longBitsToDouble(node.right.bits), result);
                    else if (node.left.op == Expr.Op.CONSTANT && node.right.op != Expr.Op.CONSTANT) {
                        var weak = scope.zeros(types.resolve(node));
                        weak.setDouble(Double.longBitsToDouble(node.left.bits));
                        binary(node.op).apply(weak, right, result);
                    } else binary(node.op).apply(left, right, result);
                    values.put(node, result);
                }
                for (Expr child : new Expr[]{node.left, node.right}) {
                    if (child == null) continue;
                    int uses = remaining.get(child) - 1;
                    remaining.put(child, uses);
                    if (uses == 0 && child.op != Expr.Op.INPUT && child.op != Expr.Op.CONSTANT) {
                        free.add(values.get(child));
                        values.remove(child);
                    }
                }
            }
            if (expression.root().op == Expr.Op.INPUT || expression.root().op == Expr.Op.CONSTANT)
                ExpressionSemantics.copyInto(values.get(expression.root()), output);
        }
        return true;
    }

    private static UnaryUfunc unary(Expr.Op op) {
        return switch (op) {
            case NEGATIVE -> Ufuncs.NEGATIVE;
            case ABSOLUTE -> Ufuncs.ABSOLUTE;
            case SQUARE -> Ufuncs.SQUARE;
            case SQRT -> Ufuncs.SQRT;
            default -> throw new IllegalArgumentException("not a unary operation: " + op);
        };
    }
    private static BinaryUfunc binary(Expr.Op op) {
        return switch (op) {
            case ADD -> Ufuncs.ADD;
            case SUBTRACT -> Ufuncs.SUBTRACT;
            case MULTIPLY -> Ufuncs.MULTIPLY;
            case DIVIDE -> Ufuncs.DIVIDE;
            default -> throw new IllegalArgumentException("not a binary operation: " + op);
        };
    }
}
