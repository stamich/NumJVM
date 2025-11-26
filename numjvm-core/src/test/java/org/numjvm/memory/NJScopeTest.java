package org.numjvm.memory;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

import org.junit.jupiter.api.Test;
import org.numjvm.exception.MemoryException;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.internal.memory.Buffer;
import org.numjvm.internal.memory.MemoryErrors;

class NJScopeTest {

    private static Buffer allocate(NJScope scope, long bytes) {
        return SharedSecrets.scopeAccess().allocator(scope).allocate(bytes);
    }

    /**
     * Reads one double the way array accessors do, translating FFM errors.
     */
    private static double read(Buffer b) {
        try {
            return b.segment().get(ValueLayout.JAVA_DOUBLE_UNALIGNED, 0);
        } catch (RuntimeException e) {
            throw MemoryErrors.translate(e, b.owner());
        }
    }

    @Test
    void scopedBuffersAreNativeAndOwnedByTheScope() {
        try (NJScope scope = NJScope.confined()) {
            Buffer b = allocate(scope, 64);
            assertTrue(b.isNative());
            assertSame(scope, b.owner());
            assertEquals(0.0, read(b));
            assertEquals(NJScope.Mode.CONFINED, scope.mode());
        }
    }

    @Test
    void closingInvalidatesEveryBuffer() {
        NJScope scope = NJScope.confined();
        Buffer a = allocate(scope, 8);
        Buffer b = allocate(scope, 1 << 20);
        assertTrue(scope.isAlive());
        scope.close();
        assertFalse(scope.isAlive());
        assertFalse(a.isAlive());
        assertFalse(b.isAlive());
        MemoryException e = assertThrows(MemoryException.class, () -> read(a));
        assertTrue(e.getMessage().contains("is closed"), e.getMessage());
        assertTrue(e.getMessage().contains("NJScope#"), e.getMessage());
        assertInstanceOf(IllegalStateException.class, e.getCause());
    }

    @Test
    void closeIsIdempotent() {
        NJScope scope = NJScope.shared();
        scope.close();
        assertDoesNotThrow(scope::close);
        assertFalse(scope.isAlive());
        assertTrue(scope.toString().endsWith("[closed]"));
    }

    @Test
    void allocatingFromAClosedScopeFails() {
        NJScope scope = NJScope.confined();
        scope.close();
        assertThrows(MemoryException.class, () -> allocate(scope, 8));
        assertThrows(MemoryException.class, scope::checkAlive);
    }

    @Test
    void confinedScopeRejectsOtherThreads() throws Exception {
        try (NJScope scope = NJScope.confined()) {
            Buffer b = allocate(scope, 8);
            Throwable t = failureOnOtherThread(() -> read(b));
            assertInstanceOf(MemoryException.class, t);
            assertTrue(t.getMessage().contains("confined"), t.getMessage());
            assertInstanceOf(MemoryException.class, failureOnOtherThread(scope::close));
            assertTrue(scope.isAlive(), "a failed close from another thread leaves the scope open");
        }
    }

    @Test
    void sharedScopeAllowsOtherThreads() throws Exception {
        try (NJScope scope = NJScope.shared()) {
            Buffer b = allocate(scope, 8);
            b.segment().set(ValueLayout.JAVA_DOUBLE_UNALIGNED, 0, 42.0);
            double v = CompletableFuture.supplyAsync(() -> read(b)).get();
            assertEquals(42.0, v);
            assertEquals(NJScope.Mode.SHARED, scope.mode());
        }
    }

    @Test
    void sharedScopeCanBeClosedFromAnotherThread() throws Exception {
        NJScope scope = NJScope.shared();
        CompletableFuture.runAsync(scope::close).get();
        assertFalse(scope.isAlive());
    }

    @Test
    void describeNamesModeAndId() {
        try (NJScope c = NJScope.confined(); NJScope s = NJScope.shared()) {
            assertTrue(c.describe().startsWith("confined NJScope#"), c.describe());
            assertTrue(s.describe().startsWith("shared NJScope#"), s.describe());
        }
    }

    @Test
    void translateLeavesUnrelatedExceptionsAlone() {
        RuntimeException other = new IndexOutOfBoundsException("x");
        try (NJScope scope = NJScope.confined()) {
            assertSame(other, MemoryErrors.translate(other, scope));
            MemorySegment s = allocate(scope, 8).segment();
            assertThrows(IndexOutOfBoundsException.class, () -> s.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, 8));
        }
    }

    private static Throwable failureOnOtherThread(Runnable r) throws InterruptedException {
        try {
            CompletableFuture.runAsync(r).get();
            return null;
        } catch (ExecutionException e) {
            return e.getCause();
        }
    }
}
