package org.numjvm.memory;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.ValueLayout;

import org.junit.jupiter.api.Test;
import org.numjvm.exception.MemoryException;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.internal.memory.AutomaticOwner;
import org.numjvm.internal.memory.Buffer;

class KeepTest {

    private static final ValueLayout.OfDouble F64 = ValueLayout.JAVA_DOUBLE_UNALIGNED;

    @Test
    void keptBytesOutliveTheScopeAndTheSourceDoesNot() {
        Buffer source;
        Buffer kept;
        try (NJScope scope = NJScope.confined()) {
            source = SharedSecrets.scopeAccess().allocator(scope).allocate(10 * 8);
            for (int i = 0; i < 10; i++) {
                source.segment().setAtIndex(F64, i, i * 1.5);
            }
            kept = SharedSecrets.scopeAccess().keep(scope, source, 2 * 8, 5 * 8);
        }
        assertFalse(source.isAlive(), "source memory is freed with the scope");
        assertTrue(kept.isAlive());
        assertSame(AutomaticOwner.INSTANCE, kept.owner());
        assertEquals(40, kept.byteSize());
        for (int i = 0; i < 5; i++) {
            assertEquals((i + 2) * 1.5, kept.segment().getAtIndex(F64, i));
        }
    }

    @Test
    void keptBytesAreACopyNotAnAlias() {
        try (NJScope scope = NJScope.confined()) {
            Buffer source = SharedSecrets.scopeAccess().allocator(scope).allocate(8);
            Buffer kept = SharedSecrets.scopeAccess().keep(scope, source, 0, 8);
            source.segment().set(F64, 0, 7.0);
            assertEquals(0.0, kept.segment().get(F64, 0));
        }
    }

    @Test
    void largeKeptBuffersGoOffHeap() {
        try (NJScope scope = NJScope.confined()) {
            long big = 2L << 20;
            Buffer source = SharedSecrets.scopeAccess().allocator(scope).allocate(big);
            Buffer kept = SharedSecrets.scopeAccess().keep(scope, source, 0, big);
            assertTrue(kept.isNative());
            assertFalse(SharedSecrets.scopeAccess().keep(scope, source, 0, 64).isNative());
        }
    }

    @Test
    void keepAfterCloseFails() {
        NJScope scope = NJScope.confined();
        Buffer source = SharedSecrets.scopeAccess().allocator(scope).allocate(8);
        scope.close();
        assertThrows(MemoryException.class, () -> SharedSecrets.scopeAccess().keep(scope, source, 0, 8));
    }
}
