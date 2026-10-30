package org.numjvm.backend;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.Slice;
import org.numjvm.internal.memory.Scratch;
import org.numjvm.ufunc.Ufuncs;
import static org.junit.jupiter.api.Assertions.*;

class BufferedConversionTest {
    @Test void chunkBoundariesAndTwoConversions() {
        for (int n : new int[]{0, 1, 4095, 4096, 4097, 16385}) {
            var a = NJ.arange(0, n, 1).astype(DType.INT8);
            var b = NJ.ones(DType.FLOAT32, n);
            var expected = NJ.add(a.astype(DType.FLOAT32), b);
            var actual = NJ.add(a, b);
            for (int i = 0; i < n; i++) { assertEquals(expected.getDouble(i), actual.getDouble(i)); }
            var out = NJ.zeros(DType.FLOAT64, n);
            Ufuncs.DIVIDE.apply(a, a.add(1L), out);
            assertEquals(0, Scratch.activeBytes());
        }
    }
    @Test void reversedBroadcastAndHazardousOverlap() {
        var a = NJ.arange(0, 20, 1).astype(DType.FLOAT64);
        Ufuncs.ADD.apply(a.slice(Slice.range(0, 19)), NJ.ones(DType.INT8, 19), a.slice(Slice.from(1)));
        for (int i = 0; i < 20; i++) { assertEquals(i, a.getDouble(i)); }
        var reversed = a.slice(Slice.step(-1));
        var sum = NJ.add(reversed, NJ.ones(DType.INT8, 1));
        assertEquals(20, sum.getDouble(0));
        assertEquals(1, sum.getDouble(19));
    }
}
