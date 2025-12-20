package org.numjvm.ufunc;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.NDArray;

class MeanAnyAllTest {

    @Test
    void meanDtypes() {
        assertEquals(DType.FLOAT64, NJ.mean(NJ.array(new long[]{1, 2})).dtype());
        assertEquals(1.5, NJ.mean(NJ.array(new long[]{1, 2})).getDouble());
        assertEquals(DType.FLOAT64, NJ.mean(NJ.array(new boolean[]{true, false, true, true})).dtype());
        assertEquals(0.75, NJ.array(new boolean[]{true, false, true, true}).mean().getDouble());
        assertEquals(DType.FLOAT16, NJ.mean(NJ.zeros(DType.FLOAT16, 2)).dtype());
        assertEquals(DType.FLOAT32, NJ.mean(NJ.zeros(DType.FLOAT32, 2)).dtype());
        assertEquals(DType.BFLOAT16, NJ.mean(NJ.zeros(DType.BFLOAT16, 2)).dtype());
        // float16 means are summed in float32: 10000 * 0.1 overflows nothing and gives 0.1
        assertEquals(Float.float16ToFloat(Float.floatToFloat16(0.1f)),
                NJ.mean(NJ.full(DType.FLOAT16, 0.1, 10000)).getDouble(), 1e-4);
    }

    @Test
    void meanOverAxesAndEmpty() {
        NDArray m = NJ.arange(0, 6, 1).reshape(2, 3);
        assertTrue(NJ.arrayEquals(NJ.array(new double[]{1.5, 2.5, 3.5}), NJ.mean(m, 0)));
        assertTrue(NJ.arrayEquals(NJ.array(new double[]{1, 4}), m.mean(1)));
        assertEquals(2, m.mean(new int[]{0, 1}, true).ndim());
        assertTrue(Double.isNaN(NJ.mean(NJ.zeros(0)).getDouble()));
    }

    @Test
    void anyAndAll() {
        NDArray x = NJ.array(new double[]{0, Double.NaN, 0}, 1, 3);
        assertTrue(NJ.any(x).getBoolean());
        assertFalse(x.all().getBoolean());
        assertEquals(DType.BOOL, NJ.any(x, 0).dtype());
        assertTrue(NJ.arrayEquals(NJ.array(new boolean[]{false, true, false}), NJ.all(x, 0)));
        assertFalse(NJ.any(NJ.zeros(0)).getBoolean());
        assertTrue(NJ.all(NJ.zeros(0)).getBoolean());
        assertTrue(x.any(new int[]{1}, true).getBoolean(0, 0));
    }
}
