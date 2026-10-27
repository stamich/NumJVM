package org.numjvm.internal.memory;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.ref.WeakReference;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.exception.ShapeException;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.memory.NJScope;

/**
 * Memory behaviour under load and at the edges (M0.2 definition of done).
 */
class MemorySoakTest {

    private static final ValueLayout.OfDouble F64 = ValueLayout.JAVA_DOUBLE_UNALIGNED;

    @Test
    void hundredThousandAutomaticNativeArraysDoNotGrowWithoutBound() throws InterruptedException {
        NativeMemoryTracker t = new NativeMemoryTracker();
        NativeAllocator a = NativeAllocator.automatic(t);
        long bytes = 32 * 1024;
        Buffer[] window = new Buffer[8];
        long maxLive = 0;
        double checksum = 0;
        for (int i = 0; i < 100_000; i++) {
            Buffer b = a.allocate(bytes);
            b.segment().set(F64, 0, i);
            checksum += b.segment().get(F64, 0);
            window[i & 7] = b;
            if (i % 1000 == 0) {
                maxLive = Math.max(maxLive, t.liveBytes());
            }
        }
        assertEquals(99_999L * 100_000 / 2, checksum);
        assertEquals(100_000, t.allocationCount());
        // 100,000 x 32 KiB = 3.1 GiB were allocated in total; at no sampled point was more
        // than the JDK's direct-memory budget alive (it triggers GC when reached).
        assertTrue(maxLive < Runtime.getRuntime().maxMemory() + (64L << 20),
                "max live native bytes " + maxLive);
        java.util.Arrays.fill(window, null);
        assertTrue(NativeMemoryTrackerTest.awaitLiveBytes(t, 0), "live bytes return to 0, are " + t.liveBytes());
    }

    @Test
    void hundredThousandScopedArraysReturnToBaseline() {
        long before = NativeMemoryTracker.GLOBAL.liveBytes();
        long seen = 0;
        for (int s = 0; s < 1000; s++) {
            try (NJScope scope = NJ.scope()) {
                MemoryAllocator a = SharedSecrets.scopeAccess().allocator(scope);
                for (int i = 0; i < 100; i++) {
                    a.allocate(4096).segment().set(F64, 0, 1.0);
                    seen++;
                }
            }
        }
        assertEquals(100_000, seen);
        assertTrue(NativeMemoryTracker.GLOBAL.liveBytes() <= before + (1 << 20),
                "scoped memory is released at close");
    }

    @Test
    void repeatedLargeScopedBuffersAreReleased() {
        for (int i = 0; i < 20; i++) {
            try (NJScope scope = NJ.scope()) {
                MemorySegment s = SharedSecrets.scopeAccess().allocator(scope).allocate(64L << 20).segment();
                s.set(F64, s.byteSize() - 8, 1.0);
            }
        }
        // Without release, 20 x 64 MiB would exceed the 1 GiB direct-memory budget of the test JVM.
    }

    @Test
    void derivedSegmentsKeepAutomaticMemoryAlive() throws InterruptedException {
        Buffer b = NativeAllocator.automatic().allocate(1024);
        b.segment().set(F64, 512, 42.0);
        MemorySegment slice = b.segment().asSlice(512);
        WeakReference<Buffer> ref = new WeakReference<>(b);
        b = null;
        for (int i = 0; i < 10 && ref.get() != null; i++) {
            System.gc();
            Thread.sleep(10);
        }
        // The buffer object may be gone, but memory reachable through a derived segment
        // stays valid: the slice keeps the arena reachable.
        assertTrue(slice.scope().isAlive());
        assertEquals(42.0, slice.get(F64, 0));
    }

    @Test
    void zeroLengthBuffersInEveryAllocator() {
        try (NJScope scope = NJ.scope()) {
            assertEquals(0, SharedSecrets.scopeAccess().allocator(scope).allocate(0).byteSize());
        }
        assertEquals(0, HybridAllocator.defaultInstance().allocate(0).byteSize());
        assertFalse(HybridAllocator.defaultInstance().allocate(0).isNative());
    }

    @Test
    void allocationSizesCannotSilentlyOverflow() {
        assertEquals(800, Sizes.bytesFor(100, 8));
        assertEquals(0, Sizes.bytesFor(0, 16));
        assertThrows(ShapeException.class, () -> Sizes.bytesFor(Long.MAX_VALUE / 4, 8));
        assertThrows(ShapeException.class, () -> Sizes.bytesFor(-1, 8));
        assertThrows(ShapeException.class, () -> Sizes.bytesFor(1, 0));
    }

    @Test
    void doubleCloseOfManyScopes() {
        for (int i = 0; i < 100; i++) {
            NJScope scope = NJ.scopeShared();
            SharedSecrets.scopeAccess().allocator(scope).allocate(64);
            scope.close();
            scope.close();
            assertFalse(scope.isAlive());
        }
    }
}
