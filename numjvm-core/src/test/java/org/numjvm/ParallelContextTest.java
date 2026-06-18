package org.numjvm;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ParallelContextTest {
    @Test void fixedChunksCoverEveryElementOnce() {
        var counts = new java.util.concurrent.atomic.AtomicIntegerArray(103);
        try (var context = NJ.parallel(3)) {
            context.runChunks(new ChunkPlan(103, 16), (chunk, start, end) -> {
                for (long i = start; i < end; i++) counts.incrementAndGet((int) i);
            });
        }
        for (int i = 0; i < 103; i++) assertEquals(1, counts.get(i));
        var plan = new ChunkPlan(Long.MAX_VALUE, 8192);
        assertEquals(Long.MAX_VALUE, plan.end(plan.count - 1));
        assertEquals(0, new ChunkPlan(0, 3).count);
    }

    @Test void confinedMemoryFallsBackToCallingThread() {
        try (var context = NJ.parallel(2); var scope = NJ.scope(); var shared = NJ.scopeShared()) {
            var confined = scope.ones(org.numjvm.dtype.DType.FLOAT64, 4);
            var shareable = shared.ones(org.numjvm.dtype.DType.FLOAT64, 4);
            assertFalse(context.workerAccessible(confined));
            assertTrue(context.workerAccessible(shareable));
            assertTrue(context.workerAccessible(NJ.ones(4)));
            var caller = Thread.currentThread();
            context.runOrElse(new org.numjvm.ndarray.NDArray[]{confined}, new ChunkPlan(4, 2),
                    (i, start, end) -> fail("confined memory reached a worker"),
                    () -> { assertSame(caller, Thread.currentThread()); assertEquals(1.0, confined.getDouble(0)); });
            assertEquals("sequential", context.lastDecision().strategy());
            assertEquals(1, context.fallbackRuns());
            context.runOrElse(new org.numjvm.ndarray.NDArray[]{shareable}, new ChunkPlan(4, 2),
                    (i, start, end) -> assertEquals(1.0, shareable.getDouble(start)), () -> fail("unexpected fallback"));
            assertEquals("parallel", context.lastDecision().strategy());
            assertEquals(1, context.parallelRuns());
        }
    }

    @Test void nestedWorkDoesNotDeadlockAndWorkerCannotCloseContext() {
        try (var context = NJ.parallel(2)) {
            var seen = new java.util.concurrent.atomic.AtomicInteger();
            context.runChunks(new ChunkPlan(2, 1), (index, start, end) -> {
                context.runChunks(new ChunkPlan(3, 1), (j, a, b) -> seen.incrementAndGet());
                assertThrows(IllegalStateException.class, context::close);
            });
            assertEquals(6, seen.get());
        }
    }

    @Test void failureWaitsForOtherWorkersBeforeReturning() throws Exception {
        try (var context = NJ.parallel(2)) {
            var started = new java.util.concurrent.CountDownLatch(1);
            var release = new java.util.concurrent.CountDownLatch(1);
            var result = new java.util.concurrent.atomic.AtomicReference<Throwable>();
            Thread caller = new Thread(() -> {
                try {
                    context.runChunks(new ChunkPlan(2, 1), (index, start, end) -> {
                        if (index == 0) {
                            try { started.await(); } catch (InterruptedException e) { throw new AssertionError(e); }
                            throw new IllegalStateException("expected");
                        }
                        started.countDown();
                        try { release.await(); } catch (InterruptedException e) { throw new AssertionError(e); }
                    });
                } catch (Throwable ex) { result.set(ex); }
            });
            caller.start();
            assertTrue(started.await(5, java.util.concurrent.TimeUnit.SECONDS));
            caller.join(100);
            assertTrue(caller.isAlive());
            release.countDown();
            caller.join(5000);
            assertFalse(caller.isAlive());
            assertEquals("expected", result.get().getMessage());
        }
    }

    @Test void interruptedCallerWaitsForWorkerAndRestoresInterrupt() throws Exception {
        try (var context = NJ.parallel(1)) {
            var started = new java.util.concurrent.CountDownLatch(1);
            var release = new java.util.concurrent.CountDownLatch(1);
            var interrupted = new java.util.concurrent.atomic.AtomicBoolean();
            var result = new java.util.concurrent.atomic.AtomicReference<Throwable>();
            Thread caller = new Thread(() -> {
                try {
                    context.runChunks(new ChunkPlan(1, 1), (index, start, end) -> {
                        started.countDown();
                        try { release.await(); } catch (InterruptedException ex) { throw new AssertionError(ex); }
                    });
                } catch (Throwable ex) {
                    result.set(ex);
                    interrupted.set(Thread.currentThread().isInterrupted());
                }
            });
            caller.start();
            assertTrue(started.await(5, java.util.concurrent.TimeUnit.SECONDS));
            caller.interrupt();
            caller.join(100);
            assertTrue(caller.isAlive());
            release.countDown();
            caller.join(5000);
            assertFalse(caller.isAlive());
            assertInstanceOf(IllegalStateException.class, result.get());
            assertTrue(interrupted.get());
        }
    }

    @Test void appliesContiguousFloatUfuncsInParallel() {
        for (var dtype : new org.numjvm.dtype.DType[]{org.numjvm.dtype.DType.FLOAT32, org.numjvm.dtype.DType.FLOAT64}) {
            try (var context = NJ.parallel(3)) {
                var left = NJ.ones(dtype, 256, 256);
                var right = NJ.ones(dtype, 256, 256);
                var output = NJ.zeros(dtype, 256, 256);
                assertSame(output, context.apply(org.numjvm.ufunc.Ufuncs.ADD, left, right, output));
                assertEquals("parallel", context.lastDecision().strategy());
                assertEquals(2.0, output.getDouble(255, 255));
                context.apply(org.numjvm.ufunc.Ufuncs.SQUARE, output, left);
                assertEquals("parallel", context.lastDecision().strategy());
                assertEquals(4.0, left.getDouble(0, 0));
            }
        }
    }

    @Test void validatesWorkerLimitAndClosesIdempotently() {
        assertThrows(IllegalArgumentException.class, () -> NJ.parallel(0));
        assertThrows(IllegalArgumentException.class, () -> NJ.parallel(257));
        var context = NJ.parallel(2);
        assertEquals(2, context.workers());
        assertFalse(context.isClosed());
        context.close();
        context.close();
        assertTrue(context.isClosed());
    }
}
