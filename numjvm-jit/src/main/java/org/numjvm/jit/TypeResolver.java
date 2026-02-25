package org.numjvm.jit;

import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Objects;
import org.numjvm.dtype.DType;
import org.numjvm.internal.ExpressionSemantics;
import org.numjvm.ufunc.BinaryUfunc;
import org.numjvm.ufunc.Ufuncs;

/** Per-invocation node types; never contains the invocation's arrays. */
final class TypeResolver {
    private final DType[] inputs;
    private final Map<Expr, DType> types = new IdentityHashMap<>();

    TypeResolver(DType[] inputs) {
        this.inputs = inputs.clone();
        for (DType type : inputs) {
            Objects.requireNonNull(type, "input type");
            if (type != DType.FLOAT32 && type != DType.FLOAT64)
                throw new IllegalArgumentException("expression inputs must be float32 or float64");
        }
    }

    DType resolve(Expr node) {
        DType existing = types.get(node);
        if (existing != null) return existing;
        DType type = switch (node.op) {
            case INPUT -> inputs[node.slot];
            case CONSTANT -> DType.FLOAT64;
            case NEGATIVE -> ExpressionSemantics.unaryResult(Ufuncs.NEGATIVE, resolve(node.left));
            case ABSOLUTE -> ExpressionSemantics.unaryResult(Ufuncs.ABSOLUTE, resolve(node.left));
            case SQUARE -> ExpressionSemantics.unaryResult(Ufuncs.SQUARE, resolve(node.left));
            case SQRT -> ExpressionSemantics.unaryResult(Ufuncs.SQRT, resolve(node.left));
            case ADD -> binary(Ufuncs.ADD, node);
            case SUBTRACT -> binary(Ufuncs.SUBTRACT, node);
            case MULTIPLY -> binary(Ufuncs.MULTIPLY, node);
            case DIVIDE -> binary(Ufuncs.DIVIDE, node);
        };
        types.put(node, type);
        return type;
    }

    private DType binary(BinaryUfunc operation, Expr node) {
        DType left = resolve(node.left), right = resolve(node.right);
        if (node.left.op == Expr.Op.CONSTANT && node.right.op != Expr.Op.CONSTANT)
            return ExpressionSemantics.weakDoubleResult(operation, right);
        if (node.right.op == Expr.Op.CONSTANT && node.left.op != Expr.Op.CONSTANT)
            return ExpressionSemantics.weakDoubleResult(operation, left);
        return ExpressionSemantics.binaryResult(operation, left, right);
    }
}
