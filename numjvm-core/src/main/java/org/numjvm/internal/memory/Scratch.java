package org.numjvm.internal.memory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicLong;
import org.numjvm.exception.MemoryException;

/**
 * An operation-local scratch lease. No thread-local storage is retained. A single
 * reservation covers all operands so nested/concurrent calls cannot deadlock.
 */
public final class Scratch implements AutoCloseable {
    /** Maximum bytes leased by one operation. */
    public static final int CHUNK_BYTES = 64 * 1024;
    /** Process-wide active conversion budget, including nested leases. */
    public static final long LIMIT = 16L * 1024 * 1024;
    private static final AtomicLong ACTIVE = new AtomicLong();
    private final Arena arena;
    private final MemorySegment memory;
    private final long bytes;
    private boolean closed;

    private Scratch(long bytes) {
        this.bytes = bytes;
        long old;
        do {
            old = ACTIVE.get();
            if (bytes > LIMIT - old) {
                throw new MemoryException("conversion scratch budget exhausted (16 MiB); retry with fewer concurrent operations");
            }
        } while (!ACTIVE.compareAndSet(old, old + bytes));
        Arena a = Arena.ofConfined();
        try {
            memory = a.allocate(bytes, 64);
            arena = a;
            NativeMemoryTracker.GLOBAL.allocated(bytes);
        } catch (RuntimeException | Error e) {
            a.close();
            ACTIVE.addAndGet(-bytes);
            throw e;
        }
    }

    /**
     * Returns independent bounded lease; caller must close it.
     * @param bytes requested bytes, from 1 through CHUNK_BYTES
     * @return independent bounded lease; caller must close it
     */
    public static Scratch lease(int bytes) {
        if (bytes < 1 || bytes > CHUNK_BYTES) { throw new IllegalArgumentException("invalid scratch size: " + bytes); }
        return new Scratch(bytes);
    }

    /** Returns the live memory; it must never escape into a returned array.
     * @return the live memory; it must never escape into a returned array
     */
    public MemorySegment memory() {
        if (closed) { throw new IllegalStateException("closed scratch lease"); }
        return memory;
    }

    /** Returns total active conversion scratch bytes, across callers.
     * @return total active conversion scratch bytes, across callers
     */
    public static long activeBytes() { return ACTIVE.get(); }

    /** Releases native bytes deterministically; closing twice is harmless. */
    @Override public void close() {
        if (!closed) {
            arena.close();
            closed = true;
            NativeMemoryTracker.GLOBAL.released(bytes);
            ACTIVE.addAndGet(-bytes);
        }
    }
}
