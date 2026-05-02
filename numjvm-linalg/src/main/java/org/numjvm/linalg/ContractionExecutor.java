package org.numjvm.linalg;

import org.numjvm.dtype.DType;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.iteration.IndexIterator;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Order;

/** Portable reference contraction, including arbitrary supported input strides. */
final class ContractionExecutor {
    private ContractionExecutor() { }

    static NDArray allocate(NDArray a, NDArray b, ContractionPlan plan) {
        return SharedSecrets.arrayAccess().allocate(MatmulMemory.allocator(a, b),
                plan.dtype(), plan.output(), Order.C);
    }

    static void dot(NDArray a, NDArray b, NDArray out, ContractionPlan plan) {
        long[] left = new long[a.shape().rank()];
        long[] right = new long[b.shape().rank()];
        var indices = new IndexIterator(plan.output());
        while (indices.hasNext()) {
            long[] index = indices.next();
            if (plan.scalarMultiply()) {
                double av = a.shape().rank() == 0 ? a.getDouble() : a.getDouble(index);
                double bv = b.shape().rank() == 0 ? b.getDouble() : b.getDouble(index);
                double product = plan.dtype() == DType.FLOAT32 ? (float) av * (float) bv : av * bv;
                out.setDouble(index, product);
                continue;
            }
            int pos = 0;
            for (int i = 0; i < plan.leftAxis(); i++) left[i] = index[pos++];
            for (int i = 0; i < plan.rightAxis(); i++) right[i] = index[pos++];
            if (right.length > 1) right[right.length - 1] = index[pos];
            if (plan.dtype() == DType.FLOAT32) {
                float sum = 0;
                for (long k = 0; k < plan.length(); k++) {
                    left[plan.leftAxis()] = k; right[plan.rightAxis()] = k;
                    sum += (float) a.getDouble(left) * (float) b.getDouble(right);
                }
                out.setDouble(index, sum);
            } else {
                double sum = 0;
                for (long k = 0; k < plan.length(); k++) {
                    left[plan.leftAxis()] = k; right[plan.rightAxis()] = k;
                    sum += a.getDouble(left) * b.getDouble(right);
                }
                out.setDouble(index, sum);
            }
        }
    }
}
