package org.numjvm.ndarray;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;

import org.junit.jupiter.api.Test;
import org.numjvm.exception.ShapeException;

class StridesTest {

    @Test
    void cAndFOrderStrides() {
        Shape s = Shape.of(3, 4, 5);
        assertArrayEquals(new long[]{160, 40, 8}, Strides.cContiguous(s, 8).values());
        assertArrayEquals(new long[]{8, 24, 96}, Strides.fContiguous(s, 8).values());
        assertArrayEquals(new long[]{24, 8}, Strides.cContiguous(Shape.of(2, 3), 8).values());
        assertArrayEquals(new long[0], Strides.cContiguous(Shape.scalar(), 8).values());
    }

    @Test
    void zeroLengthDimensionsAreTreatedAsOneForStrides() {
        assertArrayEquals(new long[]{80, 8}, Strides.cContiguous(Shape.of(0, 10), 8).values());
        assertArrayEquals(new long[]{8, 8}, Strides.fContiguous(Shape.of(0, 10), 8).values());
    }

    @Test
    void contiguityChecks() {
        Shape s = Shape.of(2, 3);
        assertTrue(Strides.isCContiguous(s, Strides.of(24, 8), 8));
        assertFalse(Strides.isFContiguous(s, Strides.of(24, 8), 8));
        assertTrue(Strides.isFContiguous(s, Strides.of(8, 16), 8));
        assertFalse(Strides.isCContiguous(Shape.of(3, 2), Strides.of(8, 24), 8), "transposed");
        assertFalse(Strides.isCContiguous(Shape.of(2, 3), Strides.of(48, 8), 8), "every other row");
        assertFalse(Strides.isCContiguous(Shape.of(4, 3), Strides.of(0, 8), 8), "broadcast");
    }

    @Test
    void axesOfLengthOneDoNotAffectContiguity() {
        // NumPy relaxed strides: the stride of a length-1 axis is irrelevant.
        assertTrue(Strides.isCContiguous(Shape.of(1, 3), Strides.of(12345, 8), 8));
        assertTrue(Strides.isFContiguous(Shape.of(1, 3), Strides.of(-7, 8), 8));
        assertTrue(Strides.isCContiguous(Shape.of(3), Strides.of(8), 8));
        assertTrue(Strides.isFContiguous(Shape.of(3), Strides.of(8), 8), "1-D contiguous is both");
    }

    @Test
    void emptyAndScalarArraysAreContiguousInBothOrders() {
        assertTrue(Strides.isCContiguous(Shape.of(0, 3), Strides.of(-8, 999), 8));
        assertTrue(Strides.isFContiguous(Shape.of(0, 3), Strides.of(-8, 999), 8));
        assertTrue(Strides.isCContiguous(Shape.scalar(), Strides.of(), 8));
        assertTrue(Strides.isFContiguous(Shape.scalar(), Strides.of(), 8));
    }

    @Test
    void extentOfSimpleLayouts() {
        assertEquals(new ByteExtent(0, 48), Strides.extent(Shape.of(2, 3), Strides.of(24, 8), 0, 8));
        assertEquals(new ByteExtent(0, 48), Strides.extent(Shape.of(3, 2), Strides.of(8, 24), 0, 8));
        // a[::-1] of 4 elements: first element at byte 24, stride -8
        assertEquals(new ByteExtent(0, 32), Strides.extent(Shape.of(4), Strides.of(-8), 24, 8));
        // broadcast row of 3 to (4, 3)
        assertEquals(new ByteExtent(0, 24), Strides.extent(Shape.of(4, 3), Strides.of(0, 8), 0, 8));
        assertEquals(new ByteExtent(16, 24), Strides.extent(Shape.scalar(), Strides.of(), 16, 8));
    }

    @Test
    void emptyArraysTouchNoMemory() {
        ByteExtent e = Strides.extent(Shape.of(0, 5), Strides.of(-1000, 1000), 4, 8);
        assertTrue(e.isEmpty());
        assertEquals(0, e.length());
    }

    @Test
    void extentOverflowIsReported() {
        assertThrows(ShapeException.class,
                () -> Strides.extent(Shape.of(3), Strides.of(Long.MAX_VALUE / 2), 0, 8));
        assertThrows(ShapeException.class,
                () -> Strides.extent(Shape.of(2), Strides.of(8), Long.MAX_VALUE - 4, 8));
    }

    @Test
    void contiguousStridesOverflowIsReported() {
        Shape huge = Shape.of(1L << 61, 2);
        assertThrows(ShapeException.class, () -> Strides.cContiguous(huge, 8));
        assertThrows(ShapeException.class, () -> Strides.fContiguous(huge, 8));
    }

    @Test
    void rankMismatchIsReported() {
        assertThrows(ShapeException.class, () -> Strides.extent(Shape.of(2, 3), Strides.of(8), 0, 8));
    }

    @Test
    void extentMatchesBruteForceEnumeration() {
        Random rnd = new Random(20251106);
        for (int trial = 0; trial < 2000; trial++) {
            int rank = rnd.nextInt(5);
            long[] dims = new long[rank];
            long[] strides = new long[rank];
            for (int d = 0; d < rank; d++) {
                dims[d] = rnd.nextInt(5);              // includes 0 and 1
                strides[d] = (rnd.nextInt(9) - 4) * 8L; // negative, zero, positive
            }
            int itemSize = 8;
            long offset = rnd.nextInt(1000);
            ByteExtent e = Strides.extent(Shape.of(dims), Strides.of(strides), offset, itemSize);
            long[] minMax = bruteForce(dims, strides, offset);
            if (minMax == null) {
                assertTrue(e.isEmpty(), "empty shape must have empty extent");
            } else {
                assertEquals(minMax[0], e.start(), () -> "start for " + java.util.Arrays.toString(dims));
                assertEquals(minMax[1] + itemSize, e.end(), () -> "end for " + java.util.Arrays.toString(dims));
            }
        }
    }

    /**
     * Enumerates every index; returns {min offset, max offset} or null if no elements.
     */
    static long[] bruteForce(long[] dims, long[] strides, long offset) {
        long size = 1;
        for (long d : dims) {
            size *= d;
        }
        if (size == 0) {
            return null;
        }
        long min = Long.MAX_VALUE;
        long max = Long.MIN_VALUE;
        long[] idx = new long[dims.length];
        for (long n = 0; n < size; n++) {
            long off = offset;
            for (int d = 0; d < dims.length; d++) {
                off += idx[d] * strides[d];
            }
            min = Math.min(min, off);
            max = Math.max(max, off);
            for (int d = dims.length - 1; d >= 0; d--) {
                if (++idx[d] < dims[d]) {
                    break;
                }
                idx[d] = 0;
            }
        }
        return new long[]{min, max};
    }

    @Test
    void byteExtentOverlap() {
        assertTrue(new ByteExtent(0, 10).overlaps(new ByteExtent(9, 20)));
        assertFalse(new ByteExtent(0, 10).overlaps(new ByteExtent(10, 20)));
        assertFalse(new ByteExtent(5, 5).overlaps(new ByteExtent(0, 20)));
        assertThrows(IllegalArgumentException.class, () -> new ByteExtent(5, 4));
    }

    @Test
    void equalityAndCopies() {
        assertEquals(Strides.of(24, 8), Strides.of(24, 8));
        long[] raw = {24, 8};
        Strides s = Strides.of(raw);
        raw[0] = 0;
        assertEquals(24, s.stride(0));
        assertEquals(8, s.stride(-1));
        assertEquals("[24, 8]", s.toString());
    }
}
