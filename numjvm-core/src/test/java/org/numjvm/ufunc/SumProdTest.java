package org.numjvm.ufunc;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.internal.loops.Pairwise;
import org.numjvm.ndarray.NDArray;

class SumProdTest {

    @Test
    void pairwiseSumIsAccurate() {
        int n = 1 << 24;
        NDArray tenths = NJ.full(DType.FLOAT64, 0.1, n);
        double exact = n * 0.1;
        double s = NJ.sum(tenths).getDouble();
        assertTrue(Math.abs(s - exact) / exact < 1e-15, () -> "relative error of " + s);
        // one pairwise sum over the whole run, as NumPy's
        assertEquals(Pairwise.float64(SharedSecrets.arrayAccess().buffer(tenths).segment(), 0, 8, n), s);
        NDArray f = NJ.full(DType.FLOAT32, 0.1, 1 << 20);
        double fs = f.sum().getDouble();
        assertTrue(Math.abs(fs - (1 << 20) * (double) 0.1f) / fs < 1e-6, () -> "float32 sum " + fs);
    }

    @Test
    void shortcutsAndDtypes() {
        NDArray m = NJ.arange(1, 7, 1).reshape(2, 3);
        assertEquals(21, NJ.sum(m).getLong());
        assertEquals(720, m.prod().getLong());
        assertTrue(NJ.arrayEquals(NJ.array(new long[]{5, 7, 9}), m.sum(0)));
        assertTrue(NJ.arrayEquals(NJ.array(new long[]{6, 120}), NJ.prod(m, -1)));
        assertEquals(DType.INT64, NJ.sum(NJ.zeros(DType.INT16, 3)).dtype());
        assertEquals(DType.UINT64, NJ.prod(NJ.zeros(DType.UINT32, 3)).dtype());
        assertEquals(DType.FLOAT32, NJ.sum(NJ.zeros(DType.FLOAT32, 3)).dtype());
        assertEquals(1, m.sum(new int[]{0, 1}, true).size());
        assertEquals(2, m.sum(new int[]{0, 1}, true).ndim());
        assertEquals(0.0, NJ.sum(NJ.zeros(0)).getDouble());
        assertEquals(1.0, NJ.prod(NJ.zeros(0)).getDouble());
        // float16 sums accumulate in float32: np.sum(np.full(10000, 0.1, np.float16)) == 1000.0
        assertEquals(1000.0, NJ.sum(NJ.full(DType.FLOAT16, 0.1, 10000)).getDouble());
    }
}
