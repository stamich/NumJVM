package org.numjvm.linalg;

import java.util.Arrays;
import org.numjvm.NJ;
import org.numjvm.exception.DTypeException;
import org.numjvm.exception.ReadOnlyArrayException;
import org.numjvm.exception.ShapeException;
import org.numjvm.iteration.IndexIterator;
import org.numjvm.ndarray.NDArray;

/** Validates destinations and protects reductions from destructive aliases. */
final class ContractionOutput {
    private ContractionOutput() { }

    static NDArray run(NDArray a, NDArray b, NDArray out, ContractionPlan plan, boolean vector) {
        validate(out, plan);
        if (out.sharesMemoryWith(a) || out.sharesMemoryWith(b)) {
            try (var scope = NJ.scope()) {
                NDArray temp = ContractionExecutor.allocateTemporary(scope, plan);
                execute(a, b, temp, plan, vector);
                var indices = new IndexIterator(plan.output());
                while (indices.hasNext()) {
                    long[] index = indices.next();
                    out.setDouble(index, temp.getDouble(index));
                }
            }
        } else {
            execute(a, b, out, plan, vector);
        }
        return out;
    }

    private static void execute(NDArray a, NDArray b, NDArray out, ContractionPlan plan, boolean vector) {
        if (vector) ContractionExecutor.vecdot(a, b, out, plan);
        else if (a.ndim() == 2 && b.ndim() == 2) Linalg.matmul(a, b, out);
        else ContractionExecutor.dot(a, b, out, plan);
    }

    private static void validate(NDArray out, ContractionPlan plan) {
        MatmulMemory.accessible(out);
        if (!out.shape().equals(plan.output())) throw new ShapeException("output shape does not match contraction");
        if (out.dtype() != plan.dtype()) throw new DTypeException("output dtype does not match contraction");
        if (!out.isWriteable()) throw new ReadOnlyArrayException("output is read-only");
        if (!nonOverlapping(out)) throw new ShapeException("output elements may overlap");
    }

    private static boolean nonOverlapping(NDArray out) {
        if (out.shape().size() == 0) return true;
        long[] dims = out.shape().dimensions(), strides = out.strides().values();
        Integer[] axes = new Integer[dims.length];
        for (int i = 0; i < axes.length; i++) {
            if (strides[i] == Long.MIN_VALUE) return false;
            axes[i] = i;
        }
        Arrays.sort(axes, (a, b) -> Long.compare(Math.abs(strides[a]), Math.abs(strides[b])));
        long extent = out.dtype().size();
        try {
            for (int axis : axes) {
                if (dims[axis] <= 1) continue;
                long stride = Math.abs(strides[axis]);
                if (stride < extent) return false;
                extent = Math.addExact(extent, Math.multiplyExact(dims[axis] - 1, stride));
            }
            return true;
        } catch (ArithmeticException overflow) {
            return false;
        }
    }
}
