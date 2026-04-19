package org.numjvm.jit;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.numjvm.BackendInfo;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.Slice;
import static org.junit.jupiter.api.Assertions.*;

class FusedDispatchTest {
    @Test void autoRunsAndCachesFusedGraphWhileExplicitModesRemainStable() {
        Assumptions.assumeTrue(FusedProviders.provider() != null, "optional vector provider unavailable");
        try (var context = Jit.context(Jit.Mode.AUTO, 1)) {
            var graph = context.trace(1, i -> i.input(0).add(.25).square());
            var input = NJ.ones(DType.FLOAT64, 4099);
            long before = BackendInfo.vectorRuns();
            assertEquals(1.5625, graph.eval(input).getDouble(4098));
            assertEquals("fused-vector", context.lastDecision().strategy());
            assertTrue(BackendInfo.vectorRuns() > before);
            assertEquals(1, context.fusedStatistics().compilations());
            graph.evalInto(input, input);
            assertEquals(1.5625, input.getDouble(0));
            assertEquals("fused-vector", context.lastDecision().strategy());
            assertTrue(context.fusedStatistics().cacheHits() > 0);
            var other = context.trace(1, i -> i.input(0).add(2).square());
            other.eval(NJ.ones(4099));
            assertEquals(1, context.fusedStatistics().cachedKernels());
            assertEquals(1, context.fusedStatistics().evictions());
            var reversed = NJ.ones(4099).slice(Slice.step(-1));
            assertEquals(1.5625, graph.eval(reversed).getDouble(0));
            assertNotEquals("fused-vector", context.lastDecision().strategy());
        }
        for (var mode : new Jit.Mode[]{Jit.Mode.BLOCK, Jit.Mode.REQUIRED}) {
            try (var context = Jit.context(mode)) {
                context.trace(1, i -> i.input(0).square()).eval(NJ.ones(4099));
                assertEquals(mode == Jit.Mode.BLOCK ? "block" : "generated", context.lastDecision().strategy());
            }
        }
    }

    @Test void unsupportedGraphFallsBackWithReason() {
        Assumptions.assumeTrue(FusedProviders.provider() != null, "optional vector provider unavailable");
        try (var context = Jit.context(Jit.Mode.AUTO)) {
            var graph = context.trace(1, i -> i.input(0).add(i.constant(1).add(i.constant(2))));
            assertEquals(4, graph.eval(NJ.ones(DType.FLOAT32, 4099)).getDouble(0));
            assertNotEquals("fused-vector", context.lastDecision().strategy());
            assertTrue(context.lastDecision().reason().contains("does not support"));
        }
    }
}
