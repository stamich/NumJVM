package org.numjvm.jit;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import static org.junit.jupiter.api.Assertions.*;

class KernelCacheTest {
    @Test void reusesAcrossLengthsAndEvictsByCapacity() {
        try (var context = Jit.context(Jit.Mode.REQUIRED, 1)) {
            var a = context.trace(1, i -> i.input(0).add(1));
            var b = context.trace(1, i -> i.input(0).add(2));
            a.eval(NJ.ones(DType.FLOAT64, 3));
            a.eval(NJ.ones(DType.FLOAT64, 7));
            assertEquals(1, context.compilations());
            b.eval(NJ.ones(DType.FLOAT64, 3));
            a.eval(NJ.ones(DType.FLOAT64, 3));
            assertEquals(3, context.compilations());
            var statistics = context.statistics();
            assertEquals(1, statistics.cacheHits());
            assertEquals(2, statistics.evictions());
            assertEquals(4, statistics.generatedRuns());
            assertEquals(0, statistics.blockRuns());
        }
    }
}
