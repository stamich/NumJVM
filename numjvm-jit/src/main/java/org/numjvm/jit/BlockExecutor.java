package org.numjvm.jit;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.numjvm.dtype.DType;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.internal.loops.UnaryLoop;
import org.numjvm.ndarray.NDArray;

/** Executes a pre-resolved graph block by block through the ufunc kernels. */
final class BlockExecutor {
    private BlockExecutor() { }

    static void run(Expression expression, NDArray[] inputs, NDArray output) {
        DType[] dtypes = new DType[inputs.length];
        for (int i = 0; i < inputs.length; i++) dtypes[i] = inputs[i].dtype();
        BlockPlan plan = expression.blockPlan(dtypes);
        var traversal = new Traversal(output.shape(), inputs, output).iterator();
        if (!traversal.hasElements()) return;
        try (var scratch = new ScratchBlocks(plan.scratchSlots + 2, 8192)) {
            do {
                for (long position = 0; position < traversal.innerSize();) {
                    int count = (int) Math.min(scratch.blockSize(), traversal.innerSize() - position);
                    for (var step : plan.steps) {
                        Expr node = step.node();
                        MemorySegment destination = scratch.slot(step.destination());
                        DType type = step.type();
                        if (node.op == Expr.Op.INPUT) {
                            var source = SharedSecrets.arrayAccess().buffer(inputs[node.slot]).segment();
                            long stride = traversal.innerStride(node.slot);
                            ScratchBlocks.gather(source, traversal.offset(node.slot) + position * stride, stride,
                                    destination, type.size(), count);
                        } else if (node.op == Expr.Op.CONSTANT) {
                            double value = Double.longBitsToDouble(node.bits);
                            for (int n = 0; n < count; n++) destination.set(ValueLayout.JAVA_DOUBLE_UNALIGNED, (long) n * 8, value);
                        } else if (node.right == null) {
                            step.unary().run(scratch.slot(step.left()), 0, step.leftType().size(),
                                    destination, 0, type.size(), count);
                        } else {
                            MemorySegment left = converted(scratch.slot(step.left()), step.leftCast(), step.leftType(), type,
                                    count, scratch.slot(plan.scratchSlots));
                            MemorySegment right = converted(scratch.slot(step.right()), step.rightCast(), step.rightType(), type,
                                    count, scratch.slot(plan.scratchSlots + 1));
                            step.binary().run(left, 0, type.size(), right, 0, type.size(), destination, 0, type.size(), count);
                        }
                    }
                    int outIndex = inputs.length;
                    long stride = traversal.innerStride(outIndex);
                    var target = SharedSecrets.arrayAccess().buffer(output).segment();
                    ScratchBlocks.scatter(scratch.slot(plan.resultSlot), target,
                            traversal.offset(outIndex) + position * stride, stride, output.dtype().size(), count);
                    position += count;
                }
            } while (traversal.next());
        }
    }

    private static MemorySegment converted(MemorySegment source, UnaryLoop cast, DType sourceType,
                                            DType targetType, int count, MemorySegment temporary) {
        if (cast == null) return source;
        cast.run(source, 0, sourceType.size(), temporary, 0, targetType.size(), count);
        return temporary;
    }
}
