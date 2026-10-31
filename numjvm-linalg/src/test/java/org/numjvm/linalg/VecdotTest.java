package org.numjvm.linalg;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.exception.ShapeException;
import org.numjvm.ndarray.Shape;
import org.numjvm.ndarray.Slice;
import static org.junit.jupiter.api.Assertions.*;

class VecdotTest {
    @Test void broadcastsAndUsesSelectedAxis() {
        var a = NJ.array(new double[]{1, 2, 3, 4}, 2, 2);
        var b = NJ.array(new double[]{5, 6});
        var result = Linalg.vecdot(a, b);
        assertEquals(Shape.of(2), result.shape());
        assertEquals(17, result.getDouble(0));
        assertEquals(39, result.getDouble(1));
        var other = NJ.array(new double[]{5, 6, 7, 8}, 2, 2);
        var first = Linalg.vecdot(a, other, 0);
        assertEquals(26, first.getDouble(0));
        assertEquals(44, first.getDouble(1));
        assertEquals(39, Linalg.vecdot(a.astype(DType.FLOAT32), b).getDouble(1));
        var singleton = NJ.ones(1, 2, 3);
        var batches = NJ.ones(4, 2, 3);
        var expanded = Linalg.vecdot(singleton, batches);
        assertEquals(Shape.of(4, 2), expanded.shape());
        assertEquals(3, expanded.getDouble(3, 1));
    }

    @Test void checksAxesEmptyAndReversedInputs() {
        var reversed = NJ.array(new double[]{1, 2, 3}).slice(Slice.step(-1));
        assertEquals(8, Linalg.vecdot(reversed, NJ.array(new double[]{1, 2, 1})).getDouble());
        var empty = Linalg.vecdot(NJ.zeros(2, 0), NJ.zeros(0));
        assertEquals(Shape.of(2), empty.shape());
        assertEquals(0, empty.getDouble(1));
        assertThrows(ShapeException.class, () -> Linalg.vecdot(NJ.zeros(2, 3), NJ.zeros(3), 1));
        assertThrows(ShapeException.class, () -> Linalg.vecdot(NJ.zeros(2), NJ.zeros(3)));
    }
}
