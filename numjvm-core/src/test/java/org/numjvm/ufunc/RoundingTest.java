package org.numjvm.ufunc;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.exception.DTypeException;
import org.numjvm.ndarray.NDArray;

/**
 * Expected values are NumPy 2.5's {@code np.round}.
 */
class RoundingTest {

    @Test
    void halvesRoundToEven() {
        NDArray r = NJ.round(NJ.array(new double[]{0.5, 1.5, 2.5, -0.5, 1.25, 2.675}));
        assertTrue(NJ.arrayEquals(NJ.array(new double[]{0, 2, 2, -0.0, 1, 3}), r));
        assertEquals(0, Double.compare(-0.0, r.getDouble(3)));
    }

    @Test
    void decimals() {
        // np.round([1.25, 2.675, 1234.5], 1) == [1.2, 2.7, 1234.5]: 2.675 * 10 is 26.75 in binary
        NDArray r = NJ.array(new double[]{1.25, 2.675, 1234.5}).round(1);
        assertEquals(1.2, r.getDouble(0));
        assertEquals(2.7, r.getDouble(1));
        assertEquals(1234.5, r.getDouble(2));
        // np.round([1234.5, 1250, 1350], -2) == [1200, 1200, 1400]
        assertTrue(NJ.arrayEquals(NJ.array(new double[]{1200, 1200, 1400}),
                NJ.round(NJ.array(new double[]{1234.5, 1250, 1350}), -2)));
    }

    @Test
    void integersKeepTheirDtype() {
        NDArray i = NJ.array(DType.INT32, new double[]{125, 135, -125, 127});
        assertEquals(DType.INT32, NJ.round(i, 2).dtype());
        NDArray r = NJ.round(i, -1);
        assertEquals(DType.INT32, r.dtype());
        assertTrue(NJ.arrayEquals(NJ.array(DType.INT32, new double[]{120, 140, -120, 130}), r));
        // np.round(np.uint8([255]), -1) == [4]: 260 wraps when converted back
        assertEquals(4, NJ.round(NJ.array(DType.UINT8, new double[]{255}), -1).getLong(0));
    }

    @Test
    void boolAndComplex() {
        NDArray b = NJ.array(new boolean[]{true, false});
        assertEquals(DType.FLOAT16, NJ.round(b).dtype());
        assertThrows(DTypeException.class, () -> NJ.round(b, 1));
        assertThrows(DTypeException.class, () -> NJ.round(NJ.zeros(DType.COMPLEX64, 1)));
    }

    @Test
    void powersOfTenFollowNumpy() {
        assertEquals(1.0, Rounding.powerOfTen(0));
        assertEquals(1e8, Rounding.powerOfTen(8));
        assertEquals(1e9, Rounding.powerOfTen(9));
        double r = 1e8;
        for (int i = 8; i < 30; i++) {
            r *= 10;
        }
        assertEquals(r, Rounding.powerOfTen(30));
    }

    @Test
    void floorCeilTruncKeepIntegers() {
        NDArray x = NJ.array(new double[]{-1.5, -0.5, 0.5, 1.5});
        assertTrue(NJ.arrayEquals(NJ.array(new double[]{-2, -1, 0, 1}), x.floor()));
        assertTrue(NJ.arrayEquals(NJ.array(new double[]{-1, -0.0, 1, 2}), x.ceil()));
        assertEquals(0, Double.compare(-0.0, NJ.trunc(x).getDouble(1)));
        assertEquals(DType.INT8, Ufuncs.FLOOR.resultType(DType.INT8));
        assertEquals(DType.BOOL, Ufuncs.CEIL.resultType(DType.BOOL));
        assertEquals(DType.FLOAT16, Ufuncs.RINT.resultType(DType.INT8));
        assertEquals(2.0, NJ.rint(NJ.array(new double[]{2.5})).getDouble(0));
    }
}
