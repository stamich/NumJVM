package org.numjvm.ufunc;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.exception.ValueException;
import org.numjvm.ndarray.NDArray;

class MinMaxTest {

    @Test
    void minAndMax() {
        NDArray m = NJ.array(new double[]{3, -1, 7, 2, 9, -4}, 2, 3);
        assertEquals(9.0, NJ.max(m).getDouble());
        assertEquals(-4.0, m.min().getDouble());
        assertTrue(NJ.arrayEquals(NJ.array(new double[]{3, 9, 7}), m.max(0)));
        assertTrue(NJ.arrayEquals(NJ.array(new double[]{-1, -4}), NJ.min(m, 1)));
        assertEquals(DType.INT8, NJ.max(NJ.array(DType.INT8, new double[]{-5, 3})).dtype());
        assertEquals(-1L, NJ.max(NJ.array(new long[]{-1, -1})).getLong());
        NDArray u = NJ.zeros(DType.UINT64, 2);
        u.setLong(1, -1L);
        assertEquals(-1L, NJ.max(u).getLong());  // 2^64 - 1, compared unsigned
        assertEquals(0L, NJ.min(u).getLong());
        assertTrue(NJ.max(NJ.array(new boolean[]{false, true})).getBoolean());
    }

    @Test
    void nanPropagatesAndEmptyFails() {
        NDArray x = NJ.array(new double[]{1, Double.NaN, 3});
        assertTrue(Double.isNaN(NJ.max(x).getDouble()));
        assertTrue(Double.isNaN(NJ.min(x).getDouble()));
        assertThrows(ValueException.class, () -> NJ.max(NJ.zeros(0)));
        assertThrows(ValueException.class, () -> NJ.min(NJ.zeros(0, 3), 0));
        assertEquals(0, NJ.min(NJ.zeros(0, 3), 1).size());
    }
}
