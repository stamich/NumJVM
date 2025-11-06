package org.numjvm.ndarray;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.numjvm.exception.ShapeException;

class ShapeTest {

    @Test
    void basicShape() {
        Shape s = Shape.of(3, 4, 5);
        assertEquals(3, s.rank());
        assertEquals(60, s.size());
        assertEquals(4, s.dimension(1));
        assertEquals(5, s.dimension(-1));
        assertEquals(3, s.dimension(-3));
        assertArrayEquals(new long[]{3, 4, 5}, s.dimensions());
        assertEquals("[3, 4, 5]", s.toString());
        assertFalse(s.isEmpty());
        assertFalse(s.isScalar());
    }

    @Test
    void scalarShapeHasOneElement() {
        Shape s = Shape.of();
        assertSame(Shape.scalar(), s);
        assertEquals(0, s.rank());
        assertEquals(1, s.size());
        assertTrue(s.isScalar());
        assertEquals("[]", s.toString());
    }

    @Test
    void zeroLengthDimensionMeansNoElements() {
        Shape s = Shape.of(0, 10);
        assertEquals(2, s.rank());
        assertEquals(0, s.size());
        assertTrue(s.isEmpty());
        assertEquals(0, Shape.of(5, 0, 7).size());
    }

    @Test
    void negativeDimensionIsRejected() {
        ShapeException e = assertThrows(ShapeException.class, () -> Shape.of(2, -1));
        assertTrue(e.getMessage().contains("negative dimension -1 at axis 1"), e.getMessage());
    }

    @Test
    void rankAboveMaximumIsRejected() {
        long[] ok = new long[Shape.MAX_RANK];
        java.util.Arrays.fill(ok, 1);
        assertEquals(Shape.MAX_RANK, Shape.of(ok).rank());
        assertThrows(ShapeException.class, () -> Shape.of(new long[Shape.MAX_RANK + 1]));
    }

    @Test
    void elementCountOverflowIsRejected() {
        assertThrows(ShapeException.class, () -> Shape.of(1L << 32, 1L << 32));
        assertThrows(ShapeException.class, () -> Shape.of(3, Long.MAX_VALUE / 2));
        assertEquals(Long.MAX_VALUE, Shape.of(Long.MAX_VALUE).size());
    }

    @Test
    void zeroSizeShapesStillRejectOverflowOfTheirOtherDimensions() {
        assertThrows(ShapeException.class, () -> Shape.of(0, 1L << 62, 1L << 62));
    }

    @Test
    void inputArrayIsCopied() {
        long[] dims = {2, 3};
        Shape s = Shape.of(dims);
        dims[0] = 99;
        assertEquals(2, s.dimension(0));
        s.dimensions()[1] = 99;
        assertEquals(3, s.dimension(1));
    }

    @Test
    void equalityIsByValue() {
        assertEquals(Shape.of(2, 3), Shape.of(2, 3));
        assertEquals(Shape.of(2, 3).hashCode(), Shape.of(2, 3).hashCode());
        assertNotEquals(Shape.of(2, 3), Shape.of(3, 2));
        assertNotEquals(Shape.of(1), Shape.of());
    }

    @Test
    void axisNormalisation() {
        assertEquals(0, Shape.normalizeAxis(0, 3));
        assertEquals(2, Shape.normalizeAxis(-1, 3));
        assertThrows(ShapeException.class, () -> Shape.normalizeAxis(3, 3));
        assertThrows(ShapeException.class, () -> Shape.normalizeAxis(-4, 3));
        assertThrows(ShapeException.class, () -> Shape.of().dimension(0));
    }
}
