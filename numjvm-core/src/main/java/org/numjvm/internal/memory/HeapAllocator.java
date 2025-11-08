package org.numjvm.internal.memory;

import java.lang.foreign.MemorySegment;

import org.numjvm.exception.MemoryException;

/**
 * Allocates GC-managed buffers backed by {@code long[]}.
 *
 * <p>{@code long[]} rather than {@code byte[]} because heap segments are only as aligned
 * as their array elements: a {@code byte[]} segment cannot be read with aligned
 * {@code double} layouts (ADR-001, spike S1.1). {@code long[]} gives 8-byte alignment,
 * enough for every dtype. The segment is sliced to the requested size.
 */
public final class HeapAllocator implements MemoryAllocator {

    /** The single instance. */
    public static final HeapAllocator INSTANCE = new HeapAllocator();

    /** Largest heap buffer: the largest {@code long[]} the JVM allows, in bytes (about 16 GiB). */
    public static final long MAX_BYTES = (long) (Integer.MAX_VALUE - 8) * Long.BYTES;

    /** Alignment of byte 0 of every heap buffer. */
    public static final int ALIGNMENT = Long.BYTES;

    private HeapAllocator() {
    }

    @Override
    public Buffer allocate(long byteSize) {
        if (byteSize < 0) {
            throw new MemoryException("negative allocation size " + byteSize);
        }
        if (byteSize > MAX_BYTES) {
            throw new MemoryException("cannot allocate " + byteSize + " bytes on the heap; the maximum is " + MAX_BYTES);
        }
        long[] backing = new long[(int) ((byteSize + 7) >>> 3)];
        MemorySegment segment = MemorySegment.ofArray(backing).asSlice(0, byteSize);
        return Buffer.of(segment, AutomaticOwner.INSTANCE, ALIGNMENT);
    }

    @Override
    public String toString() {
        return "HeapAllocator";
    }
}
