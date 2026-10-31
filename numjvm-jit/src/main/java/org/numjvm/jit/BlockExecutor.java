package org.numjvm.jit;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.IdentityHashMap;
import java.util.List;
import org.numjvm.dtype.DType;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.internal.loops.KernelRegistry;
import org.numjvm.ndarray.NDArray;

/** Executes a graph block by block through the existing ufunc kernel registry. */
final class BlockExecutor {
    private BlockExecutor() { }

    static void run(Expression expression, NDArray[] inputs, NDArray output) {
        DType[] dtypes = new DType[inputs.length];
        for (int i = 0; i < inputs.length; i++) dtypes[i] = inputs[i].dtype();
        TypeResolver types = new TypeResolver(dtypes);
        GraphPlan graph = new GraphPlan(expression.root());
        List<Expr> nodes = graph.nodes();
        var slots = new IdentityHashMap<Expr, Integer>();
        var remaining = new IdentityHashMap<Expr, Integer>();
        var free = new java.util.ArrayDeque<Integer>();
        int nextSlot = 0;
        for (Expr node : nodes) {
            int slot = free.isEmpty() ? nextSlot++ : free.pop();
            slots.put(node, slot);
            remaining.put(node, graph.consumers(node));
            for (Expr child : new Expr[]{node.left, node.right}) {
                if (child != null) {
                    int uses = remaining.get(child) - 1;
                    remaining.put(child, uses);
                    if (uses == 0) free.push(slots.get(child));
                }
            }
        }
        int tempBase = nextSlot;
        var traversal = new Traversal(output.shape(), inputs, output).iterator();
        if (!traversal.hasElements()) return;
        try (var scratch = new ScratchBlocks(tempBase + 2, 8192)) {
            do {
                for (long position = 0; position < traversal.innerSize();) {
                    int count = (int) Math.min(scratch.blockSize(), traversal.innerSize() - position);
                    for (int i = 0; i < nodes.size(); i++) {
                        Expr node = nodes.get(i);
                        MemorySegment destination = scratch.slot(slots.get(node));
                        DType type = types.resolve(node);
                        if (node.op == Expr.Op.INPUT) {
                            var source = SharedSecrets.arrayAccess().buffer(inputs[node.slot]).segment();
                            long stride = traversal.innerStride(node.slot);
                            ScratchBlocks.gather(source, traversal.offset(node.slot) + position * stride, stride,
                                    destination, type.size(), count);
                        } else if (node.op == Expr.Op.CONSTANT) {
                            double value = Double.longBitsToDouble(node.bits);
                            for (int n = 0; n < count; n++) destination.set(ValueLayout.JAVA_DOUBLE_UNALIGNED, (long) n * 8, value);
                        } else if (node.right == null) {
                            var input = scratch.slot(slots.get(node.left));
                            KernelRegistry.unary(node.op.name().toLowerCase(java.util.Locale.ROOT), types.resolve(node.left))
                                    .run(input, 0, types.resolve(node.left).size(), destination, 0, type.size(), count);
                        } else {
                            MemorySegment left = converted(node.left, type, count, scratch, slots, types, tempBase);
                            MemorySegment right = converted(node.right, type, count, scratch, slots, types, tempBase + 1);
                            KernelRegistry.binary(node.op.name().toLowerCase(java.util.Locale.ROOT), type, type)
                                    .run(left, 0, type.size(), right, 0, type.size(), destination, 0, type.size(), count);
                        }
                    }
                    int outIndex = inputs.length;
                    long stride = traversal.innerStride(outIndex);
                    var target = SharedSecrets.arrayAccess().buffer(output).segment();
                    ScratchBlocks.scatter(scratch.slot(slots.get(expression.root())), target,
                            traversal.offset(outIndex) + position * stride, stride, output.dtype().size(), count);
                    position += count;
                }
            } while (traversal.next());
        }
    }

    private static MemorySegment converted(Expr node, DType target, int count, ScratchBlocks scratch,
                                            IdentityHashMap<Expr, Integer> slots, TypeResolver types, int tempSlot) {
        DType sourceType = types.resolve(node);
        MemorySegment source = scratch.slot(slots.get(node));
        if (sourceType == target) return source;
        MemorySegment temporary = scratch.slot(tempSlot);
        KernelRegistry.unary("cast:" + target.typeName(), sourceType)
                .run(source, 0, sourceType.size(), temporary, 0, target.size(), count);
        return temporary;
    }
}
