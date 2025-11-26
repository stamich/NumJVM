package org.numjvm.ndarray;

import java.util.Arrays;

import org.numjvm.exception.ShapeException;

/**
 * Shape inference and the no-copy reshape test.
 */
final class Reshaper {

    private Reshaper() {
    }

    /**
     * Resolves a single {@code -1} and checks that the element count is unchanged.
     */
    static long[] resolve(long[] requested, long size) {
        long[] dims = requested.clone();
        int unknown = -1;
        long known = 1;
        for (int i = 0; i < dims.length; i++) {
            if (dims[i] == -1) {
                if (unknown >= 0) {
                    throw new ShapeException("can only specify one unknown dimension in " + Arrays.toString(requested));
                }
                unknown = i;
            } else if (dims[i] < 0) {
                throw new ShapeException("negative dimension " + dims[i] + " in " + Arrays.toString(requested));
            } else {
                try {
                    known = Math.multiplyExact(known, dims[i]);
                } catch (ArithmeticException e) {
                    throw new ShapeException("shape " + Arrays.toString(requested) + " overflows long", e);
                }
            }
        }
        if (unknown >= 0) {
            if (known == 0 || size % known != 0) {
                throw new ShapeException("cannot reshape array of size " + size + " into shape "
                        + Arrays.toString(requested));
            }
            dims[unknown] = size / known;
            known = size;
        }
        if (known != size) {
            throw new ShapeException("cannot reshape array of size " + size + " into shape " + Arrays.toString(requested));
        }
        return dims;
    }

    /**
     * Strides that give {@code newDims} the same elements in C order without copying, or
     * {@code null} if no such strides exist. A port of NumPy's {@code _attempt_nocopy_reshape}:
     * axes of length 1 are dropped, then runs of old and new axes with equal products are
     * matched; within a run the old axes must be contiguous with each other.
     */
    static long[] noCopyStrides(long[] oldDims, long[] oldStrides, long[] newDims, long itemSize) {
        int oldNd = 0;
        long[] od = new long[oldDims.length];
        long[] os = new long[oldDims.length];
        for (int i = 0; i < oldDims.length; i++) {
            if (oldDims[i] == 0) {
                return Strides.cStrides(newDims, itemSize);   // no elements: any layout works
            }
            if (oldDims[i] != 1) {
                od[oldNd] = oldDims[i];
                os[oldNd] = oldStrides[i];
                oldNd++;
            }
        }
        int newNd = newDims.length;
        long[] ns = new long[newNd];
        int oi = 0;
        int oj = 1;
        int ni = 0;
        int nj = 1;
        while (ni < newNd && oi < oldNd) {
            long np = newDims[ni];
            long op = od[oi];
            while (np != op) {
                if (np < op) {
                    np *= newDims[nj++];
                } else {
                    op *= od[oj++];
                }
            }
            for (int ok = oi; ok < oj - 1; ok++) {
                if (os[ok] != od[ok + 1] * os[ok + 1]) {
                    return null;
                }
            }
            ns[nj - 1] = os[oj - 1];
            for (int nk = nj - 1; nk > ni; nk--) {
                ns[nk - 1] = ns[nk] * newDims[nk];
            }
            ni = nj++;
            oi = oj++;
        }
        long last = ni >= 1 ? ns[ni - 1] : itemSize;
        for (int nk = ni; nk < newNd; nk++) {
            ns[nk] = last;
        }
        return ns;
    }
}
