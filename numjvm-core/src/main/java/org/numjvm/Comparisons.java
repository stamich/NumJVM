package org.numjvm;

import org.numjvm.dtype.DType;
import org.numjvm.dtype.Unsigned;
import org.numjvm.exception.DTypeException;
import org.numjvm.exception.ShapeException;
import org.numjvm.ndarray.NDArray;

/**
 * Element-wise comparison helpers behind {@link NJ#arrayEquals} and {@link NJ#allClose}.
 * Written for clarity, not speed: they walk logical indices.
 */
final class Comparisons {

    private Comparisons() {
    }

    static boolean arrayEquals(NDArray a, NDArray b) {
        if (!a.shape().equals(b.shape())) {
            return false;
        }
        rejectComplex(a, b);
        boolean integral = isIntegral(a.dtype()) && isIntegral(b.dtype());
        long[] idx = new long[a.ndim()];
        for (long n = 0; n < a.size(); n++) {
            if (integral) {
                if (!sameInteger(a, b, idx)) {
                    return false;
                }
            } else if (!(a.getDouble(idx) == b.getDouble(idx))) {
                return false;
            }
            increment(idx, a.shape().dimensions());
        }
        return true;
    }

    static boolean allClose(NDArray a, NDArray b, double rtol, double atol) {
        if (!a.shape().equals(b.shape())) {
            throw new ShapeException("allClose needs equal shapes, got " + a.shape() + " and " + b.shape());
        }
        rejectComplex(a, b);
        long[] idx = new long[a.ndim()];
        long[] dims = a.shape().dimensions();
        for (long n = 0; n < a.size(); n++) {
            double x = a.getDouble(idx);
            double y = b.getDouble(idx);
            boolean close = Double.isInfinite(x) || Double.isInfinite(y)
                    ? x == y
                    : Math.abs(x - y) <= atol + rtol * Math.abs(y);
            if (!close) {
                return false;
            }
            increment(idx, dims);
        }
        return true;
    }

    private static boolean isIntegral(DType t) {
        return t.isInteger() || t.isBool();
    }

    private static boolean sameInteger(NDArray a, NDArray b, long[] idx) {
        long x = a.getLong(idx);
        long y = b.getLong(idx);
        boolean ux = a.dtype() == DType.UINT64;
        boolean uy = b.dtype() == DType.UINT64;
        if (ux == uy) {
            return x == y;
        }
        // One side is uint64: equal only if the signed side is non-negative and the bits match.
        long signed = ux ? y : x;
        long unsigned = ux ? x : y;
        return signed >= 0 && signed == unsigned && Unsigned.toDouble(unsigned) >= 0;
    }

    private static void rejectComplex(NDArray a, NDArray b) {
        if (a.dtype().isComplex() || b.dtype().isComplex()) {
            throw new DTypeException("comparisons of complex arrays are not supported in NumJVM 0.1");
        }
    }

    private static void increment(long[] idx, long[] dims) {
        for (int d = idx.length - 1; d >= 0; d--) {
            if (++idx[d] < dims[d]) {
                return;
            }
            idx[d] = 0;
        }
    }
}
