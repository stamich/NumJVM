package org.numjvm.internal.memory;

import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;
import org.numjvm.exception.MemoryException;
import static org.junit.jupiter.api.Assertions.*;

class ScratchTest {
    @Test void nestedLeasesAreIndependentAndAccounted() {
        long before = Scratch.activeBytes();
        try (Scratch a = Scratch.lease(128); Scratch b = Scratch.lease(128)) {
            a.memory().set(ValueLayout.JAVA_LONG, 0, 42L);
            assertEquals(0L, b.memory().get(ValueLayout.JAVA_LONG, 0));
            assertEquals(before + 256, Scratch.activeBytes());
        }
        assertEquals(before, Scratch.activeBytes());
    }
    @Test void failureAndCloseReleaseAndInvalidateMemory() {
        Scratch a = Scratch.lease(32);
        var memory = a.memory();
        a.close(); a.close();
        assertFalse(memory.scope().isAlive());
        assertThrows(IllegalStateException.class, a::memory);
        assertThrows(IllegalArgumentException.class, () -> Scratch.lease(Scratch.CHUNK_BYTES + 1));
    }
    @Test void globalBudgetIncludesNestedLeasesAndDoesNotDeadlock() {
        var leases = new java.util.ArrayList<Scratch>();
        try {
            for (int i = 0; i < Scratch.LIMIT / Scratch.CHUNK_BYTES; i++) {
                leases.add(Scratch.lease(Scratch.CHUNK_BYTES));
            }
            assertThrows(MemoryException.class, () -> Scratch.lease(1));
        } finally { leases.forEach(Scratch::close); }
        assertEquals(0, Scratch.activeBytes());
    }
}
