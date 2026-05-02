package org.numjvm.linalg;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.Slice;
import static org.junit.jupiter.api.Assertions.*;

class NativeDotTest {
    @Test void nativeDotAndVecdotUseBothRealBindings() {
        Assumptions.assumeTrue(OpenBlas.INSTANCE != null, "optional OpenBLAS unavailable");
        try (var scope = NJ.scope()) {
            for (DType type : new DType[]{DType.FLOAT32, DType.FLOAT64}) {
                var a = scope.ones(type, 2048);
                var b = scope.ones(type, 2048);
                long before = Linalg.nativeCalls();
                assertEquals(2048, Linalg.dot(a, b).getDouble());
                assertEquals("openblas-dot", Linalg.lastBackend());
                assertEquals(2048, Linalg.vecdot(a, b).getDouble());
                assertEquals(before + 2, Linalg.nativeCalls());
                var reversed = a.slice(Slice.step(-1));
                assertEquals(2048, Linalg.dot(reversed, b).getDouble());
                assertTrue(Linalg.lastBackend().startsWith("java:"));
                var alias = a.slice(Slice.index(0));
                assertSame(alias, Linalg.dot(a, b, alias));
                assertEquals(2048, alias.getDouble());
            }
        }
    }
}
