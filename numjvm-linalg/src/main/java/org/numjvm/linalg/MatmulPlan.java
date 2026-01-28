package org.numjvm.linalg;

import java.util.Arrays;
import org.numjvm.ndarray.Shape;
import org.numjvm.shape.Broadcasting;
import org.numjvm.exception.ShapeException;

/** Allocation-free shape decisions; dimensions stay long until native validation. */
record MatmulPlan(Shape batch, Shape output, long m, long k, long n, boolean leftVector, boolean rightVector) {
    static MatmulPlan of(Shape a, Shape b) {
        if (a.rank() == 0 || b.rank() == 0) { throw new ShapeException("matmul inputs must have rank >= 1"); }
        long[] ad = a.dimensions(), bd = b.dimensions();
        boolean av = ad.length == 1, bv = bd.length == 1;
        long m = av ? 1 : ad[ad.length - 2], k = ad[ad.length - 1];
        long bk = bv ? bd[0] : bd[bd.length - 2], n = bv ? 1 : bd[bd.length - 1];
        if (k != bk) { throw new ShapeException("matmul contraction mismatch: " + k + " != " + bk); }
        Shape batch = Broadcasting.broadcastShapes(Shape.of(Arrays.copyOf(ad, Math.max(0, ad.length - 2))),
                Shape.of(Arrays.copyOf(bd, Math.max(0, bd.length - 2))));
        long[] result = Arrays.copyOf(batch.dimensions(), batch.rank() + (av ? 0 : 1) + (bv ? 0 : 1));
        int p = batch.rank();
        if (!av) { result[p++] = m; }
        if (!bv) { result[p] = n; }
        return new MatmulPlan(batch, Shape.of(result), m, k, n, av, bv);
    }
}
