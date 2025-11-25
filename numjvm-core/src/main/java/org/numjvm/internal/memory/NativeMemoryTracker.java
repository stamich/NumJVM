package org.numjvm.internal.memory;

import java.lang.ref.Cleaner;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAccumulator;

/**
 * Counts the native memory NumJVM has allocated and not yet released.
 *
 * <p>The JDK bounds native memory ({@code -XX:MaxDirectMemorySize}) but does not say how
 * much of it belongs to NumJVM; this counter does (ADR-001, spike S4). Buffers in
 * automatic arenas are counted as released when a {@link Cleaner} observes that the
 * buffer became unreachable, which happens at about the same time as the GC frees the
 * arena. Buffers in scopes are released when the scope closes.
 */
public final class NativeMemoryTracker {

    /** Process-wide tracker used by all native allocators. */
    public static final NativeMemoryTracker GLOBAL = new NativeMemoryTracker();

    private static final Cleaner CLEANER = Cleaner.create(r -> {
        Thread t = new Thread(r, "numjvm-native-memory-cleaner");
        t.setDaemon(true);
        return t;
    });

    private final AtomicLong live = new AtomicLong();
    private final LongAccumulator peak = new LongAccumulator(Math::max, 0);
    private final AtomicLong allocations = new AtomicLong();

    NativeMemoryTracker() {
    }

    /**
     * Records an allocation.
     *
     * @param bytes size of the allocation
     */
    public void allocated(long bytes) {
        long now = live.addAndGet(bytes);
        peak.accumulate(now);
        allocations.incrementAndGet();
    }

    /**
     * Records a release.
     *
     * @param bytes size being released
     */
    public void released(long bytes) {
        live.addAndGet(-bytes);
    }

    /**
     * Arranges for {@code bytes} to be released when {@code anchor} becomes unreachable.
     *
     * @param anchor object whose reachability tracks the memory (the buffer)
     * @param bytes  size of the memory
     */
    public void releaseWhenUnreachable(Object anchor, long bytes) {
        CLEANER.register(anchor, () -> released(bytes));
    }

    /**
     * Returns native bytes allocated and not yet released.
     *
     * @return native bytes allocated and not yet released
     */
    public long liveBytes() {
        return live.get();
    }

    /**
     * Returns the largest value {@link #liveBytes()} has had.
     *
     * @return the largest value {@link #liveBytes()} has had
     */
    public long peakBytes() {
        return peak.get();
    }

    /**
     * Returns the number of native allocations since start-up.
     *
     * @return the number of native allocations since start-up
     */
    public long allocationCount() {
        return allocations.get();
    }
}
