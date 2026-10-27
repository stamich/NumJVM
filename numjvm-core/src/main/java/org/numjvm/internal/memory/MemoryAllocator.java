package org.numjvm.internal.memory;

/**
 * Source of buffers. Implementations decide where memory lives and who owns it.
 */
@FunctionalInterface
public interface MemoryAllocator {

    /**
     * Allocates a zero-filled buffer.
     *
     * @param byteSize size in bytes, {@code >= 0}
     * @return the buffer
     * @throws org.numjvm.exception.MemoryException if the size is negative or the memory
     *                                              cannot be allocated
     */
    Buffer allocate(long byteSize);
}
