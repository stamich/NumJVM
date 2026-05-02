package org.numjvm.linalg;

import java.util.Arrays;
import org.numjvm.dtype.DType;
import org.numjvm.exception.DTypeException;
import org.numjvm.exception.ShapeException;
import org.numjvm.ndarray.Shape;
import org.numjvm.shape.Broadcasting;

/** Shape and promotion decisions independent of operand storage. */
record ContractionPlan(Shape output, DType dtype, long length, int leftAxis,
                       int rightAxis, boolean scalarMultiply) {
    private static DType dtype(DType a, DType b) {
        if ((a != DType.FLOAT32 && a != DType.FLOAT64)
                || (b != DType.FLOAT32 && b != DType.FLOAT64)) {
            throw new DTypeException("dot and vecdot require float32 or float64 inputs");
        }
        return a == DType.FLOAT64 || b == DType.FLOAT64 ? DType.FLOAT64 : DType.FLOAT32;
    }

    static ContractionPlan dot(Shape a, Shape b, DType left, DType right) {
        DType type = dtype(left, right);
        if (a.rank() == 0 || b.rank() == 0) {
            return new ContractionPlan(Broadcasting.broadcastShapes(a, b), type, 0, -1, -1, true);
        }
        long[] ad = a.dimensions(), bd = b.dimensions();
        int ai = ad.length - 1, bi = bd.length == 1 ? 0 : bd.length - 2;
        if (ad[ai] != bd[bi]) throw new ShapeException("dot contraction dimensions differ");
        long[] result = new long[ad.length + bd.length - 2];
        System.arraycopy(ad, 0, result, 0, ai);
        System.arraycopy(bd, 0, result, ai, bi);
        if (bd.length > 1) result[result.length - 1] = bd[bd.length - 1];
        return new ContractionPlan(Shape.of(result), type, ad[ai], ai, bi, false);
    }

    static ContractionPlan vecdot(Shape a, Shape b, DType left, DType right, int axis) {
        DType type = dtype(left, right);
        int ai = axis(axis, a.rank()), bi = axis(axis, b.rank());
        long[] ad = a.dimensions(), bd = b.dimensions();
        if (ad[ai] != bd[bi]) throw new ShapeException("vecdot contraction dimensions differ");
        Shape output = Broadcasting.broadcastShapes(Shape.of(without(ad, ai)), Shape.of(without(bd, bi)));
        return new ContractionPlan(output, type, ad[ai], ai, bi, false);
    }

    private static int axis(int axis, int rank) {
        long normalized = axis < 0 ? (long) axis + rank : axis;
        if (normalized < 0 || normalized >= rank) throw new ShapeException("invalid vecdot axis " + axis);
        return (int) normalized;
    }

    private static long[] without(long[] dimensions, int axis) {
        long[] result = Arrays.copyOf(dimensions, dimensions.length - 1);
        System.arraycopy(dimensions, axis + 1, result, axis, dimensions.length - axis - 1);
        return result;
    }
}
