package org.numjvm.internal.memory;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import org.junit.jupiter.api.Test;
import org.numjvm.exception.MemoryException;
import org.numjvm.memory.MemoryOwner;

class BufferTest {

    @Test
    void heapBufferIsAliveAndOwnedAutomatically() {
        Buffer b = Buffer.of(MemorySegment.ofArray(new long[4]), AutomaticOwner.INSTANCE, 8);
        assertEquals(32, b.byteSize());
        assertFalse(b.isNative());
        assertTrue(b.isAlive());
        assertSame(AutomaticOwner.INSTANCE, b.owner());
        assertEquals(8, b.baseAlignment());
        assertDoesNotThrow(b::checkAlive);
    }

    @Test
    void bufferDiesWithItsArena() {
        Arena arena = Arena.ofConfined();
        Buffer b = Buffer.of(arena.allocate(64, 64), AutomaticOwner.INSTANCE, 64);
        assertTrue(b.isNative());
        assertTrue(b.isAlive());
        arena.close();
        assertFalse(b.isAlive());
        MemoryException e = assertThrows(MemoryException.class, b::checkAlive);
        assertTrue(e.getMessage().contains("closed"), e.getMessage());
    }

    @Test
    void bufferDiesWithItsOwner() {
        boolean[] alive = {true};
        MemoryOwner owner = new MemoryOwner() {
            @Override
            public boolean isAlive() {
                return alive[0];
            }

            @Override
            public String describe() {
                return "test owner";
            }
        };
        Buffer b = Buffer.of(MemorySegment.ofArray(new long[1]), owner, 8);
        assertTrue(b.isAlive());
        alive[0] = false;
        assertFalse(b.isAlive());
        assertThrows(MemoryException.class, owner::checkAlive);
        assertTrue(assertThrows(MemoryException.class, b::checkAlive).getMessage().startsWith("test owner"));
    }

    @Test
    void alignmentMustBeAPowerOfTwo() {
        MemorySegment s = MemorySegment.ofArray(new long[1]);
        assertThrows(IllegalArgumentException.class, () -> Buffer.of(s, AutomaticOwner.INSTANCE, 12));
        assertThrows(IllegalArgumentException.class, () -> Buffer.of(s, AutomaticOwner.INSTANCE, 0));
    }

    @Test
    void automaticOwnerIsAlwaysAlive() {
        assertTrue(AutomaticOwner.INSTANCE.isAlive());
        assertDoesNotThrow(AutomaticOwner.INSTANCE::checkAlive);
        assertEquals("automatic memory", AutomaticOwner.INSTANCE.describe());
    }
}
