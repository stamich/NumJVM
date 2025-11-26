package org.numjvm.internal.memory;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.exception.MemoryException;
import org.numjvm.memory.MemoryStats;
import org.numjvm.memory.NJScope;

class NativeMemoryTrackerTest {

    @Test
    void countsAllocationsAndPeak() {
        NativeMemoryTracker t = new NativeMemoryTracker();
        t.allocated(100);
        t.allocated(50);
        t.released(100);
        assertEquals(50, t.liveBytes());
        assertEquals(150, t.peakBytes());
        assertEquals(2, t.allocationCount());
    }

    @Test
    void arenaAllocationsAreReleasedWhenTheArenaCloses() {
        NativeMemoryTracker t = new NativeMemoryTracker();
        Arena arena = Arena.ofConfined();
        NativeAllocator a = NativeAllocator.inArena(arena, AutomaticOwner.INSTANCE, t);
        a.allocate(1000);
        a.allocate(24);
        a.allocate(0);
        assertEquals(1024, t.liveBytes());
        assertEquals(2, t.allocationCount(), "zero-byte allocations are not counted");
        arena.close();
        a.arenaClosed();
        assertEquals(0, t.liveBytes());
    }

    @Test
    void automaticAllocationsAreReleasedAfterGc() throws InterruptedException {
        NativeMemoryTracker t = new NativeMemoryTracker();
        NativeAllocator a = NativeAllocator.automatic(t);
        for (int i = 0; i < 1000; i++) {
            a.allocate(4096);
        }
        assertEquals(1000 * 4096L, t.peakBytes());
        assertTrue(awaitLiveBytes(t, 0), "live native bytes return to 0 once the buffers are collected");
    }

    @Test
    void scopeCloseReturnsTheGlobalCounterToBaseline() {
        long before = NativeMemoryTracker.GLOBAL.liveBytes();
        NJScope scope = NJ.scope();
        org.numjvm.internal.SharedSecrets.scopeAccess().allocator(scope).allocate(1 << 16);
        assertTrue(NativeMemoryTracker.GLOBAL.liveBytes() >= before + (1 << 16));
        scope.close();
        // Other tests may run concurrently and allocate automatic memory; only require
        // that this scope's 64 KiB are gone again.
        assertTrue(NativeMemoryTracker.GLOBAL.liveBytes() < before + (1 << 16));
    }

    @Test
    void softLimitSendsLargeRequestsToTheHeap() {
        NativeMemoryTracker t = new NativeMemoryTracker();
        HybridAllocator h = new HybridAllocator(1024, 10_000, HeapAllocator.INSTANCE,
                NativeAllocator.automatic(t), t);
        Buffer first = h.allocate(8_000);
        assertTrue(first.isNative());
        Buffer second = h.allocate(8_000);
        assertFalse(second.isNative(), "8,000 more native bytes would exceed the 10,000-byte soft limit");
        assertEquals(10_000, h.nativeSoftLimit());
    }

    @Test
    void softLimitPropertyParsing() {
        assertEquals(HybridAllocator.NO_LIMIT, HybridAllocator.parseSoftLimit(null));
        assertEquals(1 << 20, HybridAllocator.parseSoftLimit("1048576"));
        assertThrows(MemoryException.class, () -> HybridAllocator.parseSoftLimit("-2"));
    }

    @Test
    void memoryStatsReflectsSettings() {
        MemoryStats s = NJ.memoryStats();
        assertEquals(HybridAllocator.defaultInstance().heapThreshold(), s.heapThreshold());
        assertFalse(s.hasNativeSoftLimit());
        assertTrue(s.peakNativeBytes() >= s.liveNativeBytes());
    }

    /**
     * Runs GC until the tracker reaches {@code target} or 10 seconds pass.
     */
    static boolean awaitLiveBytes(NativeMemoryTracker t, long target) throws InterruptedException {
        long deadline = System.nanoTime() + 10_000_000_000L;
        while (System.nanoTime() < deadline) {
            if (t.liveBytes() <= target) {
                return true;
            }
            System.gc();
            Thread.sleep(20);
        }
        return t.liveBytes() <= target;
    }
}
