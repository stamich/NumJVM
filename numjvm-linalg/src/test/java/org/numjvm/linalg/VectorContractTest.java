package org.numjvm.linalg;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.numjvm.BackendInfo;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.Slice;
import static org.junit.jupiter.api.Assertions.*;

class VectorContractTest {
    @Test void homogeneousVectorsUseSimdWithTailAndOtherLayoutsFallBack() {
        Assumptions.assumeTrue(BackendInfo.simdProvider().equals("jdk-vector"), "optional vector provider unavailable");
        for (var type : new DType[]{DType.FLOAT32, DType.FLOAT64}) {
            var a = NJ.ones(type, 259);
            var b = NJ.ones(type, 259);
            long before = BackendInfo.vectorRuns();
            assertEquals(259, Linalg.dot(a, b).getDouble());
            assertEquals("simd-dot", Linalg.lastBackend());
            assertTrue(BackendInfo.vectorRuns() > before);
            assertEquals(259, Linalg.vecdot(a, b).getDouble());
            assertEquals("simd-dot", Linalg.lastBackend());
            assertEquals(259, Linalg.dot(a.slice(Slice.step(-1)), b).getDouble());
            assertTrue(Linalg.lastBackend().startsWith("java:"));
        }
    }

    @Test void cancellationAndExceptionalValuesFollowContractionContract() {
        Assumptions.assumeTrue(BackendInfo.simdProvider().equals("jdk-vector"), "optional vector provider unavailable");
        double[] a = new double[259], b = new double[259];
        double expected = 0;
        for (int i = 0; i < a.length; i++) {
            a[i] = (i % 2 == 0 ? 1 : -1) * (0.25 + i / 100.0);
            b[i] = 0.5 + i / 1000.0;
            expected += a[i] * b[i];
        }
        var left = NJ.array(a); var right = NJ.array(b);
        assertEquals(expected, Linalg.dot(left, right).getDouble(), 1e-11);
        var floats = left.astype(DType.FLOAT32);
        assertEquals("simd-dot", Linalg.lastBackend());
        assertEquals(expected, Linalg.dot(floats, right.astype(DType.FLOAT32)).getDouble(), 1e-4);
        assertTrue(Double.isNaN(Linalg.dot(NJ.add(left, Double.NaN), right).getDouble()));
        Linalg.dot(floats, right);
        assertTrue(Linalg.lastBackend().startsWith("java:"));
    }
}
