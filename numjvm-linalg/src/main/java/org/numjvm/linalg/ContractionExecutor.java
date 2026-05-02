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

    static NDArray allocateTemporary(org.numjvm.memory.NJScope scope, ContractionPlan plan) {
        return SharedSecrets.arrayAccess().allocate(SharedSecrets.scopeAccess().allocator(scope),
                plan.dtype(), plan.output(), Order.C);
    }

    static void dot(NDArray a, NDArray b, NDArray out, ContractionPlan plan) {
        if (NativeDot.tryRun(a, b, out, plan)) return;
        BlasDispatch.dotJava();
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

    static void vecdot(NDArray a, NDArray b, NDArray out, ContractionPlan plan) {
        if (NativeDot.tryRun(a, b, out, plan)) return;
        BlasDispatch.dotJava();
        long[] left = new long[a.shape().rank()], right = new long[b.shape().rank()];
        long[] leftDims = a.shape().dimensions(), rightDims = b.shape().dimensions();
        var indices = new IndexIterator(plan.output());
        while (indices.hasNext()) {
            long[] index = indices.next();
            broadcastIndex(index, leftDims, plan.leftAxis(), left);
            broadcastIndex(index, rightDims, plan.rightAxis(), right);
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

    /** Align each input's non-contracted axes with the broadcast result from the right. */
    private static void broadcastIndex(long[] output, long[] dims, int axis, long[] input) {
        int p = output.length - (dims.length - 1);
        for (int d = 0; d < dims.length; d++) {
            if (d != axis) {
                input[d] = dims[d] == 1 ? 0 : output[p];
                p++;
            }
        }
    }
}
