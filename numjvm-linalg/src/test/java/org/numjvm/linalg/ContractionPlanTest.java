package org.numjvm.linalg;

import org.junit.jupiter.api.Test;
import org.numjvm.dtype.DType;
import org.numjvm.exception.DTypeException;
import org.numjvm.exception.ShapeException;
import org.numjvm.ndarray.Shape;
import static org.junit.jupiter.api.Assertions.*;

class ContractionPlanTest {
    private static final DType F = DType.FLOAT32;
    private static final DType D = DType.FLOAT64;

    @Test void dotCoversScalarVectorsAndHigherRanks() {
        var scalar = ContractionPlan.dot(Shape.scalar(), Shape.of(2, 3), F, D);
        assertTrue(scalar.scalarMultiply());
        assertEquals(Shape.of(2, 3), scalar.output());
        assertEquals(D, scalar.dtype());
        assertEquals(Shape.scalar(), ContractionPlan.dot(Shape.of(3), Shape.of(3), F, F).output());
        var higher = ContractionPlan.dot(Shape.of(2, 3, 4), Shape.of(5, 4, 6), F, D);
        assertEquals(Shape.of(2, 3, 5, 6), higher.output());
        assertEquals(4, higher.length());
        assertEquals(0, ContractionPlan.dot(Shape.of(0), Shape.of(0), F, F).length());
    }

    @Test void vecdotNormalizesAxesIndependentlyAndBroadcasts() {
        var plan = ContractionPlan.vecdot(Shape.of(2, 3), Shape.of(3), F, D, -1);
        assertEquals(Shape.of(2), plan.output());
        assertEquals(1, plan.leftAxis());
        assertEquals(0, plan.rightAxis());
        assertEquals(Shape.of(2), ContractionPlan.vecdot(Shape.of(3, 2), Shape.of(3, 2), F, F, 0).output());
        assertThrows(ShapeException.class, () -> ContractionPlan.vecdot(Shape.of(2, 3), Shape.of(3), F, F, 1));
        assertThrows(ShapeException.class, () -> ContractionPlan.vecdot(Shape.scalar(), Shape.of(3), F, F, -1));
        assertThrows(ShapeException.class, () -> ContractionPlan.vecdot(Shape.of(2, 3), Shape.of(4), F, F, -1));
        assertThrows(DTypeException.class, () -> ContractionPlan.dot(Shape.of(3), Shape.of(3), DType.INT32, F));
    }
}
