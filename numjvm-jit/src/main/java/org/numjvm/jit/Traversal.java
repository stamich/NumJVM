package org.numjvm.jit;

import org.numjvm.internal.SharedSecrets;
import org.numjvm.iteration.IterationPlan;
import org.numjvm.iteration.StridedIterator;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Shape;
import org.numjvm.shape.Broadcasting;

/** Creates a joint traversal for broadcast input leaves and a result. */
final class Traversal {
    private final NDArray[] operands;
    private final IterationPlan plan;

    Traversal(Shape shape, NDArray[] inputs, NDArray output) {
        operands = new NDArray[inputs.length + 1];
        System.arraycopy(inputs, 0, operands, 0, inputs.length);
        operands[inputs.length] = output;
        long[][] strides = new long[operands.length][];
        long[] offsets = new long[operands.length];
        for (int i = 0; i < operands.length; i++) {
            NDArray array = operands[i];
            strides[i] = Broadcasting.broadcastStrides(array.shape(), array.strides(), shape).values();
            offsets[i] = SharedSecrets.arrayAccess().offset(array);
        }
        plan = IterationPlan.of(shape.dimensions(), strides, offsets);
    }

    StridedIterator iterator() { return plan.iterator(); }

    /** Generated kernels accept unit-stride or scalar input runs and unit-stride output. */
    boolean generatedEligible(StridedIterator run) {
        if (!run.hasElements()) return true;
        int last = operands.length - 1;
        for (int i = 0; i < last; i++) {
            long stride = run.innerStride(i);
            if (stride != 0 && stride != operands[i].dtype().size()) return false;
        }
        return run.innerSize() <= 1 || run.innerStride(last) == operands[last].dtype().size();
    }
}
