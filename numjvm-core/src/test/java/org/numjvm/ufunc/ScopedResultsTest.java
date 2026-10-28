package org.numjvm.ufunc;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.concurrent.atomic.AtomicReference;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.exception.MemoryException;
import org.numjvm.memory.MemoryOwner;
import org.numjvm.memory.NJScope;
import org.numjvm.ndarray.NDArray;

/**
 * Results and temporaries of operations on scoped arrays live in the scope (T4.7).
 */
class ScopedResultsTest {

    @Test
    void resultsFollowTheirScopedOperand() {
        NDArray kept;
        NDArray inScope;
        try (NJScope scope = NJ.scope()) {
            NDArray a = scope.arange(0, 6, 1);
            NDArray auto = NJ.arange(0, 6, 1);
            NDArray sum = NJ.add(auto, a);
            assertSame(scope, sum.memoryOwner());
            assertSame(scope, NJ.sqrt(a).memoryOwner());          // includes the int64 -> float64 conversion
            assertSame(scope, NJ.sum(a).memoryOwner());
            assertSame(scope, NJ.mean(a, 0).memoryOwner());
            assertSame(scope, NJ.less(a, 3L).memoryOwner());
            assertSame(scope, NJ.round(NJ.divide(a, 3.0), 1).memoryOwner());  // astype and copy stay automatic
            assertSame(scope, NJ.add(a, NJ.array(DType.UINT64, new double[]{1})).memoryOwner());
            MemoryOwner autoOwner = NJ.add(auto, auto).memoryOwner();
            assertFalse(autoOwner instanceof NJScope);
            inScope = NJ.multiply(a, 2L);
            kept = scope.keep(inScope);
        }
        assertFalse(inScope.isAlive());
        NDArray dead = inScope;
        assertThrows(MemoryException.class, () -> dead.getLong(0));
        assertEquals(10, kept.getLong(5));
    }

    @Test
    void closedScopesAreNotUsedForResults() {
        NDArray dead;
        try (NJScope scope = NJ.scope()) {
            dead = scope.zeros(DType.FLOAT64, 3);
        }
        NDArray d = dead;
        assertThrows(MemoryException.class, () -> NJ.add(d, NJ.zeros(3)));
    }

    @Test
    void sharedScopeResultsAcrossThreads() throws InterruptedException {
        try (NJScope scope = NJ.scopeShared()) {
            NDArray a = scope.arange(0, 4, 1);
            AtomicReference<NDArray> r = new AtomicReference<>();
            Thread t = new Thread(() -> r.set(NJ.add(a, 1L)));
            t.start();
            t.join();
            assertSame(scope, r.get().memoryOwner());
            assertEquals(4, r.get().getLong(3));
        }
    }

    @Test
    void confinedScopeArraysFromAnotherThreadAreAMemoryException() throws InterruptedException {
        try (NJScope scope = NJ.scope()) {
            NDArray a = scope.arange(0, 4, 1);
            AtomicReference<Throwable> error = new AtomicReference<>();
            Thread t = new Thread(() -> {
                try {
                    NJ.add(a, 1L);
                } catch (Throwable e) {
                    error.set(e);
                }
            });
            t.start();
            t.join();
            assertTrue(error.get() instanceof MemoryException, String.valueOf(error.get()));
        }
    }
}
