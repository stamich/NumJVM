package org.numjvm.internal.memory;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

import org.numjvm.exception.MemoryException;
import org.numjvm.memory.MemoryOwner;

/**
 * A block of memory that arrays interpret: a {@link MemorySegment} plus the owner that
 * decides its lifetime.
 *
 * <p>Internal: never exposed through the public API, because the full writable segment
 * would let callers bypass the read-only flag of views. Arrays and views share one
 * buffer and keep it reachable; the buffer keeps the segment, and through it the
 * arena, reachable.
 */
public final class Buffer {

    private final MemorySegment segment;
    private final MemoryOwner owner;
    private final int baseAlignment;

    private Buffer(MemorySegment segment, MemoryOwner owner, int baseAlignment) {
        this.segment = Objects.requireNonNull(segment, "segment");
        this.owner = Objects.requireNonNull(owner, "owner");
        this.baseAlignment = baseAlignment;
    }

    /**
     * Wraps a segment.
     *
     * @param segment       the memory
     * @param owner         who decides its lifetime
     * @param baseAlignment guaranteed alignment of the segment's first byte, in bytes
     * @return the buffer
     */
    public static Buffer of(MemorySegment segment, MemoryOwner owner, int baseAlignment) {
        if (baseAlignment <= 0 || Integer.bitCount(baseAlignment) != 1) {
            throw new IllegalArgumentException("alignment must be a power of two: " + baseAlignment);
        }
        return new Buffer(segment, owner, baseAlignment);
    }

    /**
     * Returns the memory segment.
     *
     * @return the memory segment
     */
    public MemorySegment segment() {
        return segment;
    }

    /**
     * Returns the owner that decides this buffer's lifetime.
     *
     * @return the owner that decides this buffer's lifetime
     */
    public MemoryOwner owner() {
        return owner;
    }

    /**
     * Returns the size in bytes.
     *
     * @return the size in bytes
     */
    public long byteSize() {
        return segment.byteSize();
    }

    /**
     * Returns {@code true} if the memory is off-heap.
     *
     * @return {@code true} if the memory is off-heap
     */
    public boolean isNative() {
        return segment.isNative();
    }

    /**
     * Returns the guaranteed alignment of byte 0, e.g. 8 for heap and 64 for native buffers.
     *
     * @return the guaranteed alignment of byte 0, e.g. 8 for heap and 64 for native buffers
     */
    public int baseAlignment() {
        return baseAlignment;
    }

    /**
     * Returns {@code true} if both the owner and the segment's arena are still open.
     *
     * @return {@code true} if both the owner and the segment's arena are still open
     */
    public boolean isAlive() {
        return owner.isAlive() && segment.scope().isAlive();
    }

    /**
     * Throws if the memory may no longer be accessed.
     *
     * @throws MemoryException if the owner or arena has been closed
     */
    public void checkAlive() {
        if (!isAlive()) {
            throw new MemoryException(owner.describe() + " is closed; its arrays can no longer be accessed");
        }
    }

    @Override
    public String toString() {
        return "Buffer[" + byteSize() + " bytes, " + (isNative() ? "native" : "heap") + ", " + owner.describe() + "]";
    }
}
