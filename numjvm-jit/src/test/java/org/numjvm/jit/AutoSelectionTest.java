package org.numjvm.jit;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.Slice;
import static org.junit.jupiter.api.Assertions.*;

class AutoSelectionTest {
    @Test void choosesSmallEagerOrFallsBackWhenOverBudget() {
        try (var context = Jit.context(Jit.Mode.AUTO)) {
            var graph = context.trace(1, i -> i.input(0).add(2).square());
            assertEquals(9, graph.eval(NJ.ones(32)).getDouble(0));
            assertEquals("eager", context.lastDecision().strategy());
            assertEquals(1, context.eagerRuns());
            context.clearThreadDecision();
            assertEquals("none", context.lastDecision().strategy());
        }
        try (var context = Jit.context(Jit.Mode.AUTO, 8, 2048, 0)) {
            var graph = context.trace(1, i -> i.input(0).add(2).square());
            assertEquals(9, graph.eval(NJ.ones(32)).getDouble(0));
            assertEquals("block", context.lastDecision().strategy());
        }
    }

    @Test void usesCachedGeneratedCodeAndRejectsIneligibleLayout() {
        try (var context = Jit.context(Jit.Mode.AUTO, 8, 2048, 0)) {
            var graph = context.trace(1, i -> i.input(0).square());
            context.kernel(graph, new DType[]{DType.FLOAT64});
            assertEquals(1, graph.eval(NJ.ones(2048)).getDouble(0));
            assertEquals("generated", context.lastDecision().strategy());
            var reversed = NJ.ones(5000).slice(Slice.step(-1));
            assertEquals(1, graph.eval(reversed).getDouble(0));
            assertEquals("block", context.lastDecision().strategy());
            assertTrue(context.lastDecision().reason().contains("unsupported"));
        }
    }
}
