package org.numjvm;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ParallelContextTest {
    @Test void identicalViewsAreParallelAndShiftedOverlapUsesExistingFallback() {
        try (var context = NJ.parallel(3)) {
            var input = NJ.ones(org.numjvm.dtype.DType.FLOAT32, 65536);
            var identical = input.reshapeView(65536);
            context.apply(org.numjvm.ufunc.Ufuncs.ADD, input, identical, input);
            assertEquals("parallel", context.lastDecision().strategy());
            assertEquals(2.0, input.getDouble(0));
            assertEquals(2.0, input.getDouble(65535));

            var backing = NJ.ones(org.numjvm.dtype.DType.FLOAT32, 65537);
            var left = backing.slice(org.numjvm.ndarray.Slice.range(0, 65536));
            var shifted = backing.slice(org.numjvm.ndarray.Slice.range(1, 65537));
            context.apply(org.numjvm.ufunc.Ufuncs.ADD, left, left, shifted);
            assertEquals("sequential", context.lastDecision().strategy());
            assertEquals(1.0, backing.getDouble(0));
            assertEquals(2.0, backing.getDouble(100));
        }
    }

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
            var input = NJ.ones(org.numjvm.dtype.DType.FLOAT32, 65536);
            context.runChunks(new ChunkPlan(2, 1), (index, start, end) -> {
                context.runChunks(new ChunkPlan(3, 1), (j, a, b) -> seen.incrementAndGet());
                assertEquals(65536.0, context.sum(input).getDouble());
                assertEquals("sequential", context.lastDecision().strategy());
                assertEquals("nested worker", context.lastDecision().reason());
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

    @Test void sumUsesIdenticalFixedChunksAcrossWorkerCounts() {
        float[] data = new float[100003];
        for (int i = 0; i < data.length; i++) data[i] = (i % 17 - 8) / 7.0f;
        var input = NJ.array(data);
        int expected = 0;
        for (int workers : new int[]{1, 2, 3, 7}) {
            try (var context = NJ.parallel(workers)) {
                int actual = Float.floatToRawIntBits((float) context.sum(input).getDouble());
                if (workers == 1) expected = actual;
                else assertEquals(expected, actual);
                assertEquals("parallel", context.lastDecision().strategy());
            }
        }
        var doubleInput = input.astype(org.numjvm.dtype.DType.FLOAT64);
        long expectedDouble = 0;
        for (int workers : new int[]{1, 2, 4}) {
            try (var context = NJ.parallel(workers)) {
                long actual = Double.doubleToRawLongBits(context.sum(doubleInput).getDouble());
                if (workers == 1) expectedDouble = actual;
                else assertEquals(expectedDouble, actual);
            }
        }
    }

    @Test void sumPreservesEmptyAndSpecialValueBehavior() {
        try (var context = NJ.parallel(2); var scope = NJ.scope()) {
            assertEquals(0.0, context.sum(NJ.zeros(org.numjvm.dtype.DType.FLOAT32, 0)).getDouble());
            var confined = scope.ones(org.numjvm.dtype.DType.FLOAT64, 65536);
            assertEquals(65536.0, context.sum(confined).getDouble());
            assertEquals("sequential", context.lastDecision().strategy());
            var values = new float[65536]; values[7] = Float.NaN;
            assertTrue(Double.isNaN(context.sum(NJ.array(values)).getDouble()));
        }
    }

    @Test void fallsBackForStridesSmallSizesAndRejectsReadOnlyOutput() {
        try (var context = NJ.parallel(2)) {
            var small = NJ.ones(org.numjvm.dtype.DType.FLOAT32, 10);
            var out = NJ.zeros(org.numjvm.dtype.DType.FLOAT32, 10);
            context.apply(org.numjvm.ufunc.Ufuncs.NEGATIVE, small, out);
            assertEquals("sequential", context.lastDecision().strategy());
            assertEquals(-1.0, out.getDouble(9));
            var source = NJ.ones(org.numjvm.dtype.DType.FLOAT32, 131072)
                    .slice(org.numjvm.ndarray.Slice.step(2));
            var dest = NJ.zeros(org.numjvm.dtype.DType.FLOAT32, 65536);
            context.apply(org.numjvm.ufunc.Ufuncs.SQUARE, source, dest);
            assertEquals("sequential", context.lastDecision().strategy());
            assertEquals(1.0, dest.getDouble(65535));
            var readonly = NJ.ones(org.numjvm.dtype.DType.FLOAT32, 1).broadcastTo(65536);
            assertThrows(org.numjvm.exception.ReadOnlyArrayException.class,
                    () -> context.apply(org.numjvm.ufunc.Ufuncs.SQUARE, dest, readonly));
        }
    }

    @Test void sumHandlesInfinitiesAndSignedZero() {
        try (var context = NJ.parallel(3)) {
            float[] values = new float[65536];
            java.util.Arrays.fill(values, -0.0f);
            assertEquals(Float.floatToRawIntBits(0.0f), Float.floatToRawIntBits((float) context.sum(NJ.array(values)).getDouble()));
            values[0] = Float.POSITIVE_INFINITY; values[values.length - 1] = Float.NEGATIVE_INFINITY;
            assertTrue(Double.isNaN(context.sum(NJ.array(values)).getDouble()));
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
