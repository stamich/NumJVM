package org.numjvm.ndarray;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Map;
import java.util.function.UnaryOperator;

import net.jqwik.api.Arbitrary;
import net.jqwik.api.ForAll;
import net.jqwik.api.Property;
import net.jqwik.api.Provide;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.exception.ShapeException;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.testing.ArrayArbitraries;

class ReshapeTest {

    private static boolean sameBuffer(NDArray a, NDArray b) {
        return SharedSecrets.arrayAccess().buffer(a) == SharedSecrets.arrayAccess().buffer(b);
    }

    @Test
    void contiguousReshapeIsAView() {
        NDArray a = NJ.arange(0.0, 12.0, 1.0);
        NDArray b = a.reshape(3, 4);
        assertEquals(Shape.of(3, 4), b.shape());
        assertEquals(Strides.of(32, 8), b.strides());
        assertTrue(sameBuffer(a, b));
        b.setDouble(2, 3, 99);
        assertEquals(99.0, a.getDouble(11));
    }

    @Test
    void minusOneIsInferred() {
        NDArray a = NJ.zeros(2, 3, 4);
        assertEquals(Shape.of(6, 4), a.reshape(-1, 4).shape());
        assertEquals(Shape.of(24), a.reshape(-1).shape());
        assertEquals(Shape.of(24), a.ravel().shape());
        assertEquals(Shape.of(2, 1, 12), a.reshape(2, 1, -1).shape());
    }

    @Test
    void invalidShapesAreRejected() {
        NDArray a = NJ.zeros(2, 3);
        assertThrows(ShapeException.class, () -> a.reshape(4, 2));
        assertThrows(ShapeException.class, () -> a.reshape(-1, -1));
        assertThrows(ShapeException.class, () -> a.reshape(-1, 4));
        assertThrows(ShapeException.class, () -> a.reshape(-2, -3));
        assertThrows(ShapeException.class, () -> NJ.zeros(0).reshape(-1, 0), "size 0 cannot infer -1");
    }

    @Test
    void transposedArraysAreCopiedByReshapeAndRejectedByReshapeView() {
        NDArray t = NJ.arange(0.0, 6.0, 1.0).reshape(2, 3).transpose();
        NDArray flat = t.reshape(6);
        assertFalse(sameBuffer(t, flat), "copy");
        assertArrayEquals(new double[]{0, 3, 1, 4, 2, 5}, ArrayArbitraries.values(flat));
        ShapeException e = assertThrows(ShapeException.class, () -> t.reshapeView(6));
        assertTrue(e.getMessage().contains("without copying"), e.getMessage());
    }

    @Test
    void scalarsAndEmptyArrays() {
        NDArray s = NJ.scalar(4.0);
        assertEquals(Shape.of(1, 1), s.reshapeView(1, 1).shape());
        assertEquals(4.0, s.reshape(1).getDouble(0));
        assertEquals(Shape.scalar(), NJ.zeros(1, 1).reshapeView().shape());
        NDArray e = NJ.zeros(0, 3);
        assertEquals(Shape.of(3, 0), e.reshapeView(3, 0).shape());
        NDArray reversedEmpty = NJ.zeros(0, 4).slice(Slice.all(), Slice.step(-1));
        assertEquals(Shape.of(2, 0, 2), reversedEmpty.reshapeView(2, 0, 2).shape());
    }

    @Test
    void broadcastViewsReshapeByCopy() {
        NDArray row = NJ.arange(0.0, 3.0, 1.0);
        NDArray b = SharedSecrets.arrayAccess().create(SharedSecrets.arrayAccess().buffer(row), row.dtype(), 0,
                Shape.of(4, 3), Strides.of(0, 8), false);
        NDArray flat = b.reshape(12);
        assertTrue(flat.isWriteable(), "a copy is writeable");
        assertArrayEquals(new double[]{0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2}, ArrayArbitraries.values(flat));
        assertThrows(ShapeException.class, () -> b.reshapeView(12));
    }

    private static final Map<String, UnaryOperator<NDArray>> OPS = Map.of(
            "id", a -> a,
            "T", NDArray::transpose,
            "p102", a -> a.permuteDims(1, 0, 2),
            "s_step2_last", a -> a.slice(Slice.all(), Slice.all(), Slice.step(2)),
            "s_rev0", a -> a.slice(Slice.step(-1)),
            "s_rev_last", a -> a.slice(Slice.all(), Slice.all(), Slice.step(-1)),
            "s_col", a -> a.slice(Slice.all(), Slice.range(1, 2)),
            "s_idx_last", a -> a.slice(Slice.ellipsis(), Slice.index(0)),
            "s_mid", a -> a.slice(Slice.all(), Slice.from(1), Slice.range(1, 3)),
            "s_row", a -> a.slice(Slice.index(1)));

    /**
     * 610 (view, target shape) pairs from NumPy: reshapeView must succeed exactly where
     * NumPy's reshape returns a view, with the same strides on axes longer than 1.
     */
    @Test
    void viewOrCopyMatchesNumPy() throws IOException {
        NDArray base = NJ.arange(0.0, 24.0, 1.0).reshape(2, 3, 4);
        int views = 0;
        int rows = 0;
        try (BufferedReader r = new BufferedReader(new InputStreamReader(
                ReshapeTest.class.getResourceAsStream("reshape-cases.csv"), StandardCharsets.UTF_8))) {
            String line;
            while ((line = r.readLine()) != null) {
                if (line.startsWith("#")) {
                    continue;
                }
                String[] f = line.split(",");
                NDArray v = OPS.get(f[0]).apply(base);
                long[] target = Arrays.stream(f[1].split(";")).mapToLong(Long::parseLong).toArray();
                boolean numpyView = f[2].equals("1");
                NDArray reshaped = v.reshape(target);
                assertArrayEquals(ArrayArbitraries.values(v), ArrayArbitraries.values(reshaped), line);
                if (numpyView) {
                    NDArray rv = v.reshapeView(target);
                    String[] expected = f[3].split(";");
                    for (int d = 0; d < target.length; d++) {
                        if (!expected[d].equals("x")) {
                            assertEquals(Long.parseLong(expected[d]), rv.strides().stride(d), line);
                        }
                    }
                    assertTrue(sameBuffer(v, reshaped), line);
                    views++;
                } else {
                    assertThrows(ShapeException.class, () -> v.reshapeView(target), line);
                    assertFalse(sameBuffer(v, reshaped), line);
                }
                rows++;
            }
        }
        assertEquals(610, rows);
        assertEquals(276, views);
    }

    @Provide
    Arbitrary<NDArray> arrays() {
        return ArrayArbitraries.arrays();
    }

    @Property
    void reshapePreservesSizeAndValues(@ForAll("arrays") NDArray a) {
        NDArray flat = a.reshape(-1);
        assertEquals(a.size(), flat.size());
        assertArrayEquals(ArrayArbitraries.values(a), ArrayArbitraries.values(flat));
        NDArray back = flat.reshapeView(a.shape().dimensions());
        assertEquals(a.shape(), back.shape());
        assertArrayEquals(ArrayArbitraries.values(a), ArrayArbitraries.values(back));
    }

    @Property
    void transposedReshapeEqualsReshapeOfACopy(@ForAll("arrays") NDArray a) {
        NDArray t = a.transpose();
        assertArrayEquals(ArrayArbitraries.values(t.copy().reshape(-1)), ArrayArbitraries.values(t.reshape(-1)));
    }
}
