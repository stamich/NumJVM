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

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.exception.IndexException;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.testing.ArrayArbitraries;

class SliceTest {

    private static NDArray grid() {
        // np.arange(12.).reshape(3, 4)
        return NJ.array(new double[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 3, 4);
    }

    private static double[] v(NDArray a) {
        return ArrayArbitraries.values(a);
    }

    @Test
    void reverse() {
        NDArray r = NJ.array(new double[]{0, 1, 2, 3, 4}).slice(Slice.step(-1));   // a[::-1]
        assertArrayEquals(new double[]{4, 3, 2, 1, 0}, v(r));
        assertEquals(Strides.of(-8), r.strides());
    }

    @Test
    void negativeStepWithBounds() {
        NDArray a = NJ.arange(0.0, 10.0, 1.0);
        assertArrayEquals(new double[]{5, 3}, v(a.slice(Slice.range(5, 1, -2))));      // a[5:1:-2]
        assertArrayEquals(new double[]{9, 6, 3, 0}, v(a.slice(Slice.step(-3))));       // a[::-3]
    }

    @Test
    void indexRemovesTheAxis() {
        NDArray g = grid();
        NDArray row = g.slice(Slice.index(1));                                          // g[1]
        assertEquals(Shape.of(4), row.shape());
        assertArrayEquals(new double[]{4, 5, 6, 7}, v(row));
        NDArray col = g.slice(Slice.all(), Slice.index(-1));                            // g[:, -1]
        assertArrayEquals(new double[]{3, 7, 11}, v(col));
        NDArray element = g.slice(Slice.index(2), Slice.index(1));                      // g[2, 1]
        assertEquals(0, element.ndim());
        assertEquals(9.0, element.getDouble());
    }

    @Test
    void rangesOnSeveralAxes() {
        NDArray g = grid();
        NDArray s = g.slice(Slice.step(2), Slice.range(3, 0, -1));                      // g[::2, 3:0:-1]
        assertEquals(Shape.of(2, 3), s.shape());
        assertArrayEquals(new double[]{3, 2, 1, 11, 10, 9}, v(s));
        assertEquals(Strides.of(64, -8), s.strides());
        NDArray inner = g.slice(Slice.range(1, 3), Slice.range(1, 3));                  // g[1:3, 1:3]
        assertArrayEquals(new double[]{5, 6, 9, 10}, v(inner));
    }

    @Test
    void newAxisAndEllipsis() {
        NDArray g = grid();
        NDArray e = g.slice(Slice.ellipsis(), Slice.newAxis());                         // g[..., None]
        assertEquals(Shape.of(3, 4, 1), e.shape());
        NDArray front = g.slice(Slice.newAxis());                                       // g[None]
        assertEquals(Shape.of(1, 3, 4), front.shape());
        NDArray last = g.slice(Slice.ellipsis(), Slice.index(0));                       // g[..., 0]
        assertArrayEquals(new double[]{0, 4, 8}, v(last));
        NDArray middle = NJ.zeros(2, 3, 4, 5).slice(Slice.index(0), Slice.ellipsis(), Slice.index(1));
        assertEquals(Shape.of(3, 4), middle.shape());
        NDArray scalar = NJ.scalar(3.0).slice(Slice.newAxis(), Slice.ellipsis());       // x[None, ...]
        assertEquals(Shape.of(1), scalar.shape());
        assertEquals(3.0, scalar.getDouble(0));
        assertEquals(Shape.scalar(), NJ.scalar(3.0).slice(Slice.ellipsis()).shape());
    }

    @Test
    void emptyResults() {
        NDArray a = NJ.arange(0.0, 5.0, 1.0);
        assertEquals(0, a.slice(Slice.range(2, 2)).size());
        assertEquals(0, a.slice(Slice.from(10)).size());
        assertEquals(0, a.slice(Slice.range(1, 3, -1)).size());
        assertEquals(0, NJ.zeros(0, 3).slice(Slice.all(), Slice.index(1)).size());
    }

    @Test
    void slicesAreZeroCopyViews() {
        NDArray g = grid();
        NDArray s = g.slice(Slice.range(1, 3), Slice.step(-1));
        assertTrue(SharedSecrets.arrayAccess().buffer(g) == SharedSecrets.arrayAccess().buffer(s));
        s.setDouble(0, 0, 100);
        assertEquals(100.0, g.getDouble(1, 3));
    }

    @Test
    void errors() {
        NDArray g = grid();
        assertThrows(IndexException.class, () -> g.slice(Slice.index(0), Slice.index(0), Slice.index(0)));
        assertThrows(IndexException.class, () -> g.slice(Slice.ellipsis(), Slice.ellipsis()));
        IndexException e = assertThrows(IndexException.class, () -> g.slice(Slice.index(3)));
        assertEquals("index 3 is out of bounds for axis 0 with size 3", e.getMessage());
        assertThrows(IndexException.class, () -> g.slice(Slice.all(), Slice.index(-5)));
        assertThrows(IllegalArgumentException.class, () -> Slice.range(0, 5, 0));
        assertThrows(IndexException.class, () -> NJ.scalar(1.0).slice(Slice.index(0)));
    }

    @Test
    void readOnlyCarriesOver() {
        NDArray b = NJ.arange(0.0, 3.0, 1.0);
        NDArray ro = SharedSecrets.arrayAccess().create(SharedSecrets.arrayAccess().buffer(b),
                b.dtype(), 0, Shape.of(2, 3), Strides.of(0, 8), false);
        assertFalse(ro.slice(Slice.index(0)).isWriteable());
    }

    @Test
    void toStringOfSlices() {
        assertEquals("1:5:2", Slice.range(1, 5, 2).toString());
        assertEquals(":3", Slice.to(3).toString());
        assertEquals("::-1", Slice.step(-1).toString());
        assertEquals("...", Slice.ellipsis().toString());
    }

    /**
     * 864 start/stop/step combinations on np.arange(7), expected values produced by NumPy.
     */
    @Test
    void matchesNumPyForEveryBoundCombination() throws IOException {
        NDArray a = NJ.arange(0.0, 7.0, 1.0);
        int checked = 0;
        try (BufferedReader r = new BufferedReader(new InputStreamReader(
                SliceTest.class.getResourceAsStream("slice-cases.csv"), StandardCharsets.UTF_8))) {
            String line;
            while ((line = r.readLine()) != null) {
                if (line.startsWith("#")) {
                    continue;
                }
                String[] f = line.split(",", -1);
                Long start = f[0].equals("-") ? null : Long.valueOf(f[0]);
                Long stop = f[1].equals("-") ? null : Long.valueOf(f[1]);
                long step = Long.parseLong(f[2]);
                double[] expected = f[3].isEmpty() ? new double[0]
                        : Arrays.stream(f[3].split(";")).mapToDouble(Double::parseDouble).toArray();
                NDArray s = a.slice(new Slice.Range(start, stop, step));
                assertArrayEquals(expected, v(s), line);
                checked++;
            }
        }
        assertEquals(864, checked);
    }
}
