package org.numjvm.jit;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.numjvm.dtype.DType;
import org.numjvm.internal.SharedSecrets;
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
        var access = SharedSecrets.arrayAccess();
        MemorySegment target = access.buffer(output).segment();
        MemorySegment[] memory = new MemorySegment[plan.scratchSlots];
        long[] offsets = new long[plan.scratchSlots], strides = new long[plan.scratchSlots];
        try (var scratch = new ScratchBlocks(plan.scratchSlots + 2, 8192, expression.scratchBytes())) {
            do {
                for (long position = 0; position < traversal.innerSize();) {
                    int count = (int) Math.min(scratch.blockSize(), traversal.innerSize() - position);
                    int outIndex = inputs.length;
                    long outStride = traversal.innerStride(outIndex);
                    long outOffset = traversal.offset(outIndex) + position * outStride;
                    boolean directResult = false;
                    for (var step : plan.steps) {
                        Expr node = step.node();
                        DType type = step.type();
                        int slot = step.destination();
                        if (node.op == Expr.Op.INPUT) {
                            var source = access.buffer(inputs[node.slot]).segment();
                            long stride = traversal.innerStride(node.slot);
                            long offset = traversal.offset(node.slot) + position * stride;
                            // Root input is kept in scratch for the general scatter path.
                            if (stride == type.size() && node != expression.root()) {
                                memory[slot] = source; offsets[slot] = offset; strides[slot] = stride;
                            } else {
                                memory[slot] = scratch.slot(slot); offsets[slot] = 0; strides[slot] = type.size();
                                ScratchBlocks.gather(source, offset, stride, memory[slot], type.size(), count);
                            }
                            continue;
                        }
                        boolean direct = node == expression.root() && node.op != Expr.Op.CONSTANT
                                && outStride == type.size();
                        MemorySegment destination = direct ? target : scratch.slot(slot);
                        long destinationOffset = direct ? outOffset : 0;
                        if (node.op == Expr.Op.CONSTANT) {
                            double value = Double.longBitsToDouble(node.bits);
                            for (int n = 0; n < count; n++) destination.set(ValueLayout.JAVA_DOUBLE_UNALIGNED, (long) n * 8, value);
                        } else if (node.right == null) {
                            int left = step.left();
                            step.unary().run(memory[left], offsets[left], strides[left], destination,
                                    destinationOffset, type.size(), count);
                        } else {
                            int li = step.left(), ri = step.right();
                            MemorySegment left = memory[li], right = memory[ri];
                            long lo = offsets[li], ro = offsets[ri], ls = strides[li], rs = strides[ri];
                            if (step.leftCast() != null) {
                                var converted = scratch.slot(plan.scratchSlots);
                                step.leftCast().run(left, lo, ls, converted, 0, type.size(), count);
                                left = converted; lo = 0; ls = type.size();
                            }
                            if (step.rightCast() != null) {
                                var converted = scratch.slot(plan.scratchSlots + 1);
                                step.rightCast().run(right, ro, rs, converted, 0, type.size(), count);
                                right = converted; ro = 0; rs = type.size();
                            }
                            step.binary().run(left, lo, ls, right, ro, rs, destination, destinationOffset, type.size(), count);
                        }
                        memory[slot] = destination; offsets[slot] = destinationOffset; strides[slot] = type.size();
                        if (direct) directResult = true;
                    }
                    if (!directResult) {
                        ScratchBlocks.scatter(scratch.slot(plan.resultSlot), target, outOffset, outStride, output.dtype().size(), count);
                    }
                    position += count;
                }
            } while (traversal.next());
        }
    }
}
