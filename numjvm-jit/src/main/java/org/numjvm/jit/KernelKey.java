package org.numjvm.jit;

import java.util.IdentityHashMap;
import java.util.List;
import org.numjvm.dtype.DType;

/** Structural identity contains no invocation arrays or memory addresses. */
record KernelKey(String graph, List<DType> types) {
    static KernelKey of(Expression expression, DType[] inputTypes) {
        var nodes = new GraphPlan(expression.root()).nodes();
        var ids = new IdentityHashMap<Expr, Integer>();
        StringBuilder graph = new StringBuilder("v1;");
        for (Expr node : nodes) {
            ids.put(node, ids.size());
            graph.append(node.op.ordinal()).append(',').append(node.slot).append(',').append(node.bits).append(',')
                    .append(node.left == null ? -1 : ids.get(node.left)).append(',')
                    .append(node.right == null ? -1 : ids.get(node.right)).append(';');
        }
        return new KernelKey(graph.toString(), List.of(inputTypes.clone()));
    }
}
