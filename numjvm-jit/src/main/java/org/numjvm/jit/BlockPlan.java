package org.numjvm.jit;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Locale;
import org.numjvm.dtype.DType;
import org.numjvm.internal.loops.BinaryLoop;
import org.numjvm.internal.loops.KernelRegistry;
import org.numjvm.internal.loops.UnaryLoop;

/** Immutable typed schedule and liveness-derived slots; contains no operand arrays. */
final class BlockPlan {
    record Step(Expr node, DType type, DType leftType, DType rightType, int destination, int left, int right,
                UnaryLoop unary, BinaryLoop binary, UnaryLoop leftCast, UnaryLoop rightCast) { }
    final List<Step> steps;
    final int scratchSlots;
    final int resultSlot;

    BlockPlan(GraphPlan graph, DType[] inputs) {
        var types = new TypeResolver(inputs);
        var slots = new IdentityHashMap<Expr, Integer>();
        var remaining = new IdentityHashMap<Expr, Integer>();
        var free = new ArrayDeque<Integer>();
        var schedule = new ArrayList<Step>();
        int nextSlot = 0;
        for (Expr node : graph.nodes()) {
            int slot = free.isEmpty() ? nextSlot++ : free.pop();
            slots.put(node, slot);
            remaining.put(node, graph.consumers(node));
            DType type = types.resolve(node);
            int left = node.left == null ? -1 : slots.get(node.left);
            int right = node.right == null ? -1 : slots.get(node.right);
            UnaryLoop unary = null, leftCast = null, rightCast = null;
            BinaryLoop binary = null;
            if (node.op != Expr.Op.INPUT && node.op != Expr.Op.CONSTANT) {
                String op = node.op.name().toLowerCase(Locale.ROOT);
                if (node.right == null) unary = KernelRegistry.unary(op, types.resolve(node.left));
                else {
                    binary = KernelRegistry.binary(op, type, type);
                    DType lt = types.resolve(node.left), rt = types.resolve(node.right);
                    if (lt != type) leftCast = KernelRegistry.unary("cast:" + type.typeName(), lt);
                    if (rt != type) rightCast = KernelRegistry.unary("cast:" + type.typeName(), rt);
                }
            }
            schedule.add(new Step(node, type, node.left == null ? null : types.resolve(node.left),
                    node.right == null ? null : types.resolve(node.right),
                    slot, left, right, unary, binary, leftCast, rightCast));
            for (Expr child : new Expr[]{node.left, node.right}) {
                if (child != null) {
                    int uses = remaining.get(child) - 1;
                    remaining.put(child, uses);
                    if (uses == 0) free.push(slots.get(child));
                }
            }
        }
        steps = List.copyOf(schedule);
        scratchSlots = nextSlot;
        resultSlot = schedule.getLast().destination();
    }
}
