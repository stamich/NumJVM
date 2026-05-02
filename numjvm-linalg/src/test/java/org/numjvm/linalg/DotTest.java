package org.numjvm.linalg;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.exception.ShapeException;
import org.numjvm.ndarray.Slice;
import org.numjvm.ndarray.Shape;
import static org.junit.jupiter.api.Assertions.*;

class DotTest {
    @Test void scalarsVectorsMatricesAndMixedTypes() {
        var a = NJ.array(new double[]{1, 2, 3});
        var b = NJ.array(new double[]{4, 5, 6});
        assertEquals(32, Linalg.dot(a, b).getDouble());
        assertEquals(32, Linalg.dot(a.astype(DType.FLOAT32), b).getDouble());
        assertEquals(DType.FLOAT32, Linalg.dot(a.astype(DType.FLOAT32), b.astype(DType.FLOAT32)).dtype());
        var scale = NJ.array(new double[]{2}).reshapeView();
        assertEquals(6, Linalg.dot(scale, a).getDouble(2));
        var m = NJ.array(new double[]{1, 2, 3, 4}, 2, 2);
        var n = NJ.array(new double[]{5, 6}, 2, 1);
        var result = Linalg.dot(m, n);
        assertEquals(Shape.of(2, 1), result.shape());
        assertEquals(17, result.getDouble(0, 0));
        assertEquals(39, result.getDouble(1, 0));
    }

    @Test void higherRankNegativeStridesAndEmptyContractions() {
        var a = NJ.arange(1, 25, 1).reshapeView(2, 3, 4).astype(DType.FLOAT64);
        var b = NJ.ones(DType.FLOAT64, 5, 4, 6);
        var result = Linalg.dot(a, b);
        assertEquals(Shape.of(2, 3, 5, 6), result.shape());
        assertEquals(10, result.getDouble(new long[]{0, 0, 0, 0}));
        assertEquals(90, result.getDouble(new long[]{1, 2, 4, 5}));
        var reversed = NJ.array(new double[]{1, 2, 3}).slice(Slice.step(-1));
        assertEquals(8, Linalg.dot(reversed, NJ.array(new double[]{1, 2, 1})).getDouble());
        assertEquals(0, Linalg.dot(NJ.zeros(0), NJ.zeros(0)).getDouble());
        assertThrows(ShapeException.class, () -> Linalg.dot(NJ.zeros(2), NJ.zeros(3)));
    }

    @Test void resultBelongsToFirstScope() {
        try (var scope = NJ.scope()) {
            var result = Linalg.dot(scope.ones(DType.FLOAT64, 3), NJ.ones(3));
            assertSame(scope, result.memoryOwner());
            assertEquals(3, result.getDouble());
        }
    }
}
