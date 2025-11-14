package org.numjvm.shape;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import net.jqwik.api.Arbitrary;
import net.jqwik.api.ForAll;
import net.jqwik.api.Property;
import net.jqwik.api.Provide;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.numjvm.NJ;
import org.numjvm.exception.ReadOnlyArrayException;
import org.numjvm.exception.ShapeException;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Shape;
import org.numjvm.ndarray.Strides;
import org.numjvm.testing.ArrayArbitraries;

class BroadcastingTest {

    private static Shape parse(String s) {
        if (s.isBlank()) {
            return Shape.scalar();
        }
        return Shape.of(java.util.Arrays.stream(s.trim().split(" ")).mapToLong(Long::parseLong).toArray());
    }

    /** The table from the M0 specification (section 15.3), plus NumPy's documentation example. */
    @ParameterizedTest(name = "{0} with {1} -> {2}")
    @CsvSource({
            "3,       3,      3",
            "3 1,     1 4,    3 4",
            "5 1 4,   1 3 1,  5 3 4",
            "'',      2 3,    2 3",
            "0 3,     1 3,    0 3",
            "8 1 6 1, 7 1 5,  8 7 6 5",
            "1,       0,      0",
    })
    void compatibleShapes(String a, String b, String expected) {
        assertEquals(parse(expected), Broadcasting.broadcastShapes(parse(a), parse(b)));
        assertEquals(parse(expected), Broadcasting.broadcastShapes(parse(b), parse(a)), "symmetric");
    }

    @Test
    void incompatibleShapes() {
        ShapeException e = assertThrows(ShapeException.class,
                () -> Broadcasting.broadcastShapes(Shape.of(2, 3), Shape.of(4, 3)));
        assertEquals("shape mismatch: objects cannot be broadcast to a single shape. Mismatch is between arg 0 with"
                + " shape [2, 3] and arg 1 with shape [4, 3].", e.getMessage());
        assertFalse(Broadcasting.areBroadcastable(Shape.of(3), Shape.of(4)));
        assertFalse(Broadcasting.areBroadcastable(Shape.of(0), Shape.of(2)), "0 only broadcasts with 1");
    }

    @Test
    void multipleOperands() {
        assertEquals(Shape.of(2, 3, 4),
                Broadcasting.broadcastShapes(Shape.of(4), Shape.of(3, 1), Shape.of(2, 1, 1)));
        assertEquals(Shape.scalar(), Broadcasting.broadcastShapes());
        assertTrue(Broadcasting.areBroadcastable(Shape.of(1), Shape.of(5, 1), Shape.of(5)));
        ShapeException e = assertThrows(ShapeException.class,
                () -> Broadcasting.broadcastShapes(Shape.of(1), Shape.of(2), Shape.of(3)));
        assertTrue(e.getMessage().contains("arg 1 with shape [2] and arg 2 with shape [3]"), e.getMessage());
    }

    @Test
    void stridesOfBroadcastAxesAreZero() {
        assertEquals(Strides.of(0, 8), Broadcasting.broadcastStrides(Shape.of(1, 3), Strides.of(24, 8), Shape.of(4, 3)));
        assertEquals(Strides.of(0, 0, 8),
                Broadcasting.broadcastStrides(Shape.of(3), Strides.of(8), Shape.of(2, 4, 3)));
        assertThrows(ShapeException.class,
                () -> Broadcasting.broadcastStrides(Shape.of(2, 3), Strides.of(24, 8), Shape.of(3)));
        assertThrows(ShapeException.class,
                () -> Broadcasting.broadcastStrides(Shape.of(2), Strides.of(8), Shape.of(3)));
    }

    @Test
    void broadcastToIsAReadOnlyZeroCopyView() {
        NDArray row = NJ.array(new double[]{1, 2, 3}, 1, 3);
        NDArray b = row.broadcastTo(4, 3);
        assertEquals(Shape.of(4, 3), b.shape());
        assertEquals(Strides.of(0, 8), b.strides());
        assertTrue(SharedSecrets.arrayAccess().buffer(row) == SharedSecrets.arrayAccess().buffer(b));
        assertFalse(b.isWriteable());
        assertArrayEquals(new double[]{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}, ArrayArbitraries.values(b));
        assertThrows(ReadOnlyArrayException.class, () -> b.setDouble(0, 0, 9));
        row.setDouble(0, 1, 20);
        assertEquals(20.0, b.getDouble(3, 1), "the view sees writes to its source");
        assertFalse(NJ.zeros(3).broadcastTo(3).isWriteable(), "broadcastTo is always read-only, as in NumPy");
        assertThrows(ShapeException.class, () -> row.broadcastTo(4, 2));
        assertEquals(Shape.of(0, 3), row.broadcastTo(0, 3).shape());
    }

    @Provide
    Arbitrary<NDArray> arrays() {
        return ArrayArbitraries.arrays();
    }

    @Property
    void broadcastingPreservesSourceValues(@ForAll("arrays") NDArray a, @ForAll("lead") long lead) {
        long[] src = a.shape().dimensions();
        long[] target = new long[src.length + 1];
        target[0] = lead;
        for (int i = 0; i < src.length; i++) {
            target[i + 1] = src[i] == 1 ? 3 : src[i];
        }
        NDArray b = a.broadcastTo(target);
        long[] idx = new long[target.length];
        long[] srcIdx = new long[src.length];
        for (long n = 0; n < b.size(); n++) {
            for (int i = 0; i < src.length; i++) {
                srcIdx[i] = src[i] == 1 ? 0 : idx[i + 1];
            }
            assertEquals(a.getDouble(srcIdx), b.getDouble(idx));
            for (int d = target.length - 1; d >= 0; d--) {
                if (++idx[d] < target[d]) {
                    break;
                }
                idx[d] = 0;
            }
        }
    }

    @Provide
    Arbitrary<Long> lead() {
        return net.jqwik.api.Arbitraries.longs().between(0, 3);
    }
}
