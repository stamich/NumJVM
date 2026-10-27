package org.numjvm;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.numjvm.dtype.DType;
import org.numjvm.exception.DTypeException;
import org.numjvm.exception.ShapeException;
import org.numjvm.ndarray.NDArray;

class ComparisonsTest {

    @Test
    void arrayEqualsComparesShapeAndValues() {
        NDArray a = NJ.array(new double[]{1, 2, 3, 4}, 2, 2);
        assertTrue(NJ.arrayEquals(a, NJ.array(new double[]{1, 2, 3, 4}, 2, 2)));
        assertFalse(NJ.arrayEquals(a, NJ.array(new double[]{1, 2, 3, 4})), "different shape");
        assertFalse(NJ.arrayEquals(a, NJ.array(new double[]{1, 2, 3, 5}, 2, 2)));
        assertTrue(NJ.arrayEquals(a, NJ.array(new long[]{1, 2, 3, 4}, 2, 2)), "dtypes may differ");
        assertTrue(NJ.arrayEquals(NJ.zeros(0, 3), NJ.zeros(DType.INT8, 0, 3)));
        assertTrue(NJ.arrayEquals(NJ.scalar(1.0), NJ.scalar(1L)));
    }

    @Test
    void nanIsNeverEqual() {
        NDArray n = NJ.array(new double[]{Double.NaN});
        assertFalse(NJ.arrayEquals(n, n));
        assertFalse(NJ.allClose(n, n));
    }

    @Test
    void largeIntegersAreComparedExactly() {
        NDArray a = NJ.array(new long[]{(1L << 53) + 1});
        NDArray b = NJ.array(new long[]{1L << 53});
        assertFalse(NJ.arrayEquals(a, b), "would be equal as doubles");
        NDArray u = NJ.array(new long[]{-1}).astype(DType.UINT64);
        assertFalse(NJ.arrayEquals(u, NJ.array(new long[]{-1})), "2^64-1 is not -1");
        assertTrue(NJ.arrayEquals(NJ.array(new long[]{5}).astype(DType.UINT64), NJ.array(new long[]{5})));
    }

    @Test
    void allCloseUsesNumPyTolerances() {
        NDArray a = NJ.array(new double[]{1.0, 1e10});
        assertTrue(NJ.allClose(a, NJ.array(new double[]{1.0 + 1e-9, 1e10 + 1e4})));
        assertFalse(NJ.allClose(a, NJ.array(new double[]{1.001, 1e10})));
        assertTrue(NJ.allClose(a, NJ.array(new double[]{1.001, 1e10}), 1e-2, 0));
        NDArray inf = NJ.array(new double[]{Double.POSITIVE_INFINITY});
        assertTrue(NJ.allClose(inf, inf));
        assertFalse(NJ.allClose(inf, NJ.array(new double[]{Double.NEGATIVE_INFINITY})));
        assertThrows(ShapeException.class, () -> NJ.allClose(NJ.zeros(2), NJ.zeros(3)));
    }

    @Test
    void complexComparisonsAreNotSupportedYet() {
        NDArray c = NJ.zeros(DType.COMPLEX64, 1);
        assertThrows(DTypeException.class, () -> NJ.arrayEquals(c, c));
    }
}
