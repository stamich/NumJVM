package org.numjvm;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ufunc.Ufuncs;
import static org.junit.jupiter.api.Assertions.*;

class ParallelStressTest {
    @Test void concurrentContextsAndRepeatedShutdown() throws Exception {
        var callers = Executors.newFixedThreadPool(4);
        try {
            var shared = NJ.ones(DType.FLOAT32, 65536);
            var futures = new java.util.ArrayList<java.util.concurrent.Future<?>>();
            for (int t = 0; t < 4; t++) futures.add(callers.submit(() -> {
                for (int repeat = 0; repeat < 4; repeat++) {
                    try (var context = NJ.parallel(2)) {
                        var output = NJ.zeros(DType.FLOAT32, 65536);
                        context.apply(Ufuncs.ADD, shared, shared, output);
                        assertEquals("parallel", context.lastDecision().strategy());
                        assertEquals(2.0, output.getDouble(65535));
                    }
                }
            }));
            for (var future : futures) future.get(30, TimeUnit.SECONDS);
        } finally { callers.shutdownNow(); }
    }

    @Test void nestedCallsAcrossContextsAreRejected() {
        try (var first = NJ.parallel(2); var second = NJ.parallel(2)) {
            var value = NJ.ones(DType.FLOAT32, 65536);
            first.runChunks(new ChunkPlan(1, 1), (i, start, end) -> {
                assertThrows(IllegalStateException.class, () -> second.sum(value));
                assertThrows(IllegalStateException.class, second::close);
            });
        }
    }

    @Test void scopeClosureDuringWorkFailsBeforeCallerReturns() throws Exception {
        var scope = NJ.scopeShared();
        try (var context = NJ.parallel(2)) {
            var input = scope.ones(DType.FLOAT64, 65536);
            var started = new CountDownLatch(1);
            var release = new CountDownLatch(1);
            var failure = new AtomicReference<Throwable>();
            var caller = new Thread(() -> {
                try {
                    context.runOrElse(new NDArray[]{input}, new ChunkPlan(1, 1), (i, a, b) -> {
                        started.countDown();
                        try { release.await(); } catch (InterruptedException e) { throw new AssertionError(e); }
                        input.getDouble(0);
                    }, () -> fail("unexpected fallback"));
                } catch (Throwable error) { failure.set(error); }
            });
            caller.start();
            assertTrue(started.await(5, TimeUnit.SECONDS));
            scope.close();
            release.countDown();
            caller.join(5000);
            assertFalse(caller.isAlive());
            assertNotNull(failure.get());
        } finally { if (scope.isAlive()) scope.close(); }
    }
}
