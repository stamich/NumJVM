package org.numjvm.internal.memory;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.numjvm.dtype.DType;
import org.numjvm.exception.MemoryException;

class AllocatorTest {

    @ParameterizedTest
    @EnumSource(DType.class)
    void heapBuffersAreAlignedForEveryDType(DType dtype) {
        Buffer b = HeapAllocator.INSTANCE.allocate(dtype.size() * 10L);
        assertEquals(dtype.size() * 10L, b.byteSize());
        assertFalse(b.isNative());
        assertTrue(b.segment().maxByteAlignment() >= dtype.alignment());
        assertAlignedAccessWorks(b.segment(), dtype);
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void nativeBuffersAre64ByteAligned(DType dtype) {
        Buffer b = NativeAllocator.automatic().allocate(dtype.size() * 10L);
        assertTrue(b.isNative());
        assertEquals(0, b.segment().address() % NativeAllocator.ALIGNMENT);
        assertEquals(NativeAllocator.ALIGNMENT, b.baseAlignment());
        assertAlignedAccessWorks(b.segment(), dtype);
    }

    /** Reads every element with the dtype's aligned layout; would throw on misalignment. */
    private static void assertAlignedAccessWorks(MemorySegment s, DType dtype) {
        ValueLayout layout = switch (dtype.alignment()) {
            case 1 -> ValueLayout.JAVA_BYTE;
            case 2 -> ValueLayout.JAVA_SHORT;
            case 4 -> ValueLayout.JAVA_INT;
            case 8 -> ValueLayout.JAVA_LONG;
            default -> throw new AssertionError(dtype);
        };
        for (long off = 0; off + layout.byteSize() <= s.byteSize(); off += layout.byteSize()) {
            switch (layout) {
                case ValueLayout.OfByte l -> s.get(l, off);
                case ValueLayout.OfShort l -> s.get(l, off);
                case ValueLayout.OfInt l -> s.get(l, off);
                case ValueLayout.OfLong l -> s.get(l, off);
                default -> throw new AssertionError(layout);
            }
        }
    }

    @Test
    void heapSizeIsRoundedUpInternallyButReportedExactly() {
        Buffer b = HeapAllocator.INSTANCE.allocate(13);
        assertEquals(13, b.byteSize());
        assertEquals(8, b.baseAlignment());
    }

    @Test
    void buffersAreZeroFilled() {
        for (MemoryAllocator a : new MemoryAllocator[]{HeapAllocator.INSTANCE, NativeAllocator.automatic()}) {
            MemorySegment s = a.allocate(4096).segment();
            for (long i = 0; i < s.byteSize(); i++) {
                assertEquals(0, s.get(ValueLayout.JAVA_BYTE, i));
            }
        }
    }

    @Test
    void zeroLengthAllocationsWork() {
        assertEquals(0, HeapAllocator.INSTANCE.allocate(0).byteSize());
        assertEquals(0, NativeAllocator.automatic().allocate(0).byteSize());
        assertEquals(0, HybridAllocator.withThreshold(0).allocate(0).byteSize());
    }

    @Test
    void negativeSizesAreRejected() {
        assertThrows(MemoryException.class, () -> HeapAllocator.INSTANCE.allocate(-1));
        assertThrows(MemoryException.class, () -> NativeAllocator.automatic().allocate(-1));
        assertThrows(MemoryException.class, () -> HybridAllocator.withThreshold(1024).allocate(-1));
    }

    @Test
    void heapRejectsSizesBeyondTheLargestLongArray() {
        assertThrows(MemoryException.class, () -> HeapAllocator.INSTANCE.allocate(HeapAllocator.MAX_BYTES + 1));
    }

    @Test
    void nativeAllocationBeyondDirectMemoryLimitIsAMemoryException() {
        // Far above any MaxDirectMemorySize used in tests; the JDK refuses before touching memory.
        MemoryException e = assertThrows(MemoryException.class,
                () -> NativeAllocator.automatic().allocate(1L << 50));
        assertTrue(e.getMessage().contains("MaxDirectMemorySize"), e.getMessage());
    }

    @Test
    void hybridRoutesByThreshold() {
        HybridAllocator h = HybridAllocator.withThreshold(1024);
        assertFalse(h.allocate(1).isNative());
        assertFalse(h.allocate(1024).isNative());
        assertTrue(h.allocate(1025).isNative());
        HybridAllocator allNative = HybridAllocator.withThreshold(0);
        assertTrue(allNative.allocate(1).isNative());
    }

    @Test
    void defaultThresholdIsOneMebibyte() {
        assertEquals(1L << 20, HybridAllocator.parseThreshold(null));
        assertEquals(1L << 20, HybridAllocator.parseThreshold(" "));
        assertSame(HybridAllocator.defaultInstance(), HybridAllocator.defaultInstance());
    }

    @ParameterizedTest
    @ValueSource(strings = {"-1", "abc", "1.5", "1MiB"})
    void invalidThresholdPropertyIsRejected(String value) {
        assertThrows(MemoryException.class, () -> HybridAllocator.parseThreshold(value));
    }

    @Test
    void thresholdPropertyIsParsed() {
        assertEquals(0, HybridAllocator.parseThreshold("0"));
        assertEquals(32768, HybridAllocator.parseThreshold(" 32768 "));
        assertThrows(MemoryException.class, () -> HybridAllocator.withThreshold(-5));
    }

    @Test
    void arenaAllocatorUsesTheArenaAndOwner() {
        Arena arena = Arena.ofConfined();
        NativeAllocator a = NativeAllocator.inArena(arena, AutomaticOwner.INSTANCE);
        Buffer b = a.allocate(128);
        assertTrue(b.isAlive());
        arena.close();
        assertFalse(b.isAlive());
        assertThrows(MemoryException.class, () -> a.allocate(8));
    }
}
