package org.numjvm.jit;

import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.concurrent.Executors;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.exception.MemoryException;
import org.numjvm.internal.memory.Scratch;
import org.numjvm.ndarray.NDArray;
import static org.junit.jupiter.api.Assertions.*;

class ResourceIntegrationTest {
    @Test void closedAndConfinedScopesFailWithoutChangingOutput() throws Exception {
        var scope = NJ.scope();
        var input = scope.ones(DType.FLOAT64, 4099);
        try (var context = Jit.context(Jit.Mode.AUTO); var pool = Executors.newSingleThreadExecutor()) {
            var graph = context.trace(1, i -> i.input(0).square());
            var out = NJ.ones(4099);
            pool.submit(() -> assertThrows(MemoryException.class, () -> graph.evalInto(out, input))).get();
            assertEquals(1, out.getDouble(0));
            scope.close();
            assertThrows(MemoryException.class, () -> graph.evalInto(out, input));
            assertEquals(1, out.getDouble(0));
        }
    }

    @Test void concurrentFusedCallsShareOnlyCompiledCode() throws Exception {
        Assumptions.assumeTrue(FusedProviders.provider() != null, "optional vector provider unavailable");
        var context = Jit.context(Jit.Mode.AUTO, 2);
        try (var pool = Executors.newFixedThreadPool(4)) {
            var graph = context.trace(1, i -> i.input(0).add(.25).square());
            var tasks = new ArrayList<java.util.concurrent.Future<Double>>();
            for (int i = 0; i < 8; i++) tasks.add(pool.submit(() -> graph.eval(NJ.ones(4099)).getDouble(4098)));
            for (var task : tasks) assertEquals(1.5625, task.get());
            assertEquals(1, context.fusedStatistics().compilations());
            assertEquals(8, context.fusedStatistics().runs());
        } finally { context.close(); }
        assertEquals(0, context.fusedStatistics().cachedKernels());
    }

    @Test void blockScratchReturnsAfterRepeatedAndExceptionalCalls() {
        long before = Scratch.activeBytes();
        try (var context = Jit.context(Jit.Mode.BLOCK, 2, 2048)) {
            var graph = context.trace(1, i -> i.input(0).square());
            for (int i = 0; i < 10; i++) graph.eval(NJ.ones(4099));
            assertThrows(IllegalArgumentException.class, () -> graph.eval(new NDArray[0]));
        }
        assertEquals(before, Scratch.activeBytes());
    }

    @Test void compiledCacheDoesNotKeepInvocationInputAlive() throws Exception {
        Assumptions.assumeTrue(FusedProviders.provider() != null, "optional vector provider unavailable");
        try (var context = Jit.context(Jit.Mode.AUTO)) {
            var graph = context.trace(1, i -> i.input(0).square());
            WeakReference<NDArray> reference = executeTemporary(graph);
            WeakReference<Object> gcProbe = new WeakReference<>(new Object());
            for (int i = 0; i < 30 && (reference.get() != null || gcProbe.get() != null); i++) {
                System.gc();
                Thread.sleep(20);
            }
            Assumptions.assumeTrue(gcProbe.get() == null, "garbage collection did not run");
            assertNull(reference.get(), "compiled cache retained an invocation array");
        }
    }
    private static WeakReference<NDArray> executeTemporary(Expression graph) {
        var input = NJ.ones(4099);
        graph.eval(input);
        return new WeakReference<>(input);
    }
}
