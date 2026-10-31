package org.numjvm.jit;

import java.util.concurrent.Executors;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.exception.MemoryException;
import static org.junit.jupiter.api.Assertions.*;

class ConcurrencyTest {
    @Test void samePlanExecutesConcurrentlyOnIndependentArrays() throws Exception {
        try (var context = Jit.context(Jit.Mode.REQUIRED); var pool = Executors.newFixedThreadPool(4)) {
            var graph = context.trace(1, i -> i.input(0).add(2).square());
            var tasks = new java.util.ArrayList<java.util.concurrent.Future<Double>>();
            for (int i = 0; i < 8; i++) tasks.add(pool.submit(() -> graph.eval(NJ.ones(DType.FLOAT64, 100)).getDouble(0)));
            for (var task : tasks) assertEquals(9.0, task.get());
            assertEquals(1, context.statistics().compilations());
        }
    }
    @Test void confinedInputCannotBeEvaluatedByAnotherThread() throws Exception {
        try (var scope = NJ.scope(); var context = Jit.context(); var pool = Executors.newSingleThreadExecutor()) {
            var input = scope.ones(DType.FLOAT64, 4);
            var graph = context.trace(1, i -> i.input(0).square());
            var failure = pool.submit(() -> assertThrows(MemoryException.class, () -> graph.eval(input)));
            failure.get();
        }
    }
    @Test void blockLeasesRemainIndependentAcrossThreads() throws Exception {
        try (var context = Jit.context(Jit.Mode.BLOCK, 8, 2048); var pool = Executors.newFixedThreadPool(4)) {
            var graph = context.trace(1, i -> i.input(0).add(2).square());
            var tasks = new java.util.ArrayList<java.util.concurrent.Future<Double>>();
            for (int i = 0; i < 8; i++) tasks.add(pool.submit(() -> graph.eval(NJ.ones(DType.FLOAT64, 10000)).getDouble(9999)));
            for (var task : tasks) assertEquals(9.0, task.get());
        }
    }
}
