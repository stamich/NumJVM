package org.numjvm.internal.memory;

import org.numjvm.exception.MemoryException;

/**
 * Allocator for arrays that are not created in a scope: small buffers on the heap,
 * large ones in automatic native arenas.
 *
 * <p>The threshold defaults to {@link #DEFAULT_HEAP_THRESHOLD} (1 MiB, ADR-001) and can
 * be set with the system property {@value #THRESHOLD_PROPERTY} (bytes; {@code 0} sends
 * every non-empty buffer off-heap). Requests too large for a {@code long[]} always go
 * off-heap.
 */
public final class HybridAllocator implements MemoryAllocator {

    /** System property holding the heap threshold in bytes. */
    public static final String THRESHOLD_PROPERTY = "numjvm.memory.heapThreshold";

    /** Default heap threshold: 1 MiB. */
    public static final long DEFAULT_HEAP_THRESHOLD = 1L << 20;

    private static volatile HybridAllocator defaultInstance;

    private final long heapThreshold;
    private final MemoryAllocator heap;
    private final MemoryAllocator offHeap;

    HybridAllocator(long heapThreshold, MemoryAllocator heap, MemoryAllocator offHeap) {
        if (heapThreshold < 0) {
            throw new MemoryException("heap threshold must be >= 0, was " + heapThreshold);
        }
        this.heapThreshold = heapThreshold;
        this.heap = heap;
        this.offHeap = offHeap;
    }

    /**
     * Creates an allocator with an explicit threshold.
     *
     * @param heapThreshold largest request, in bytes, served from the heap
     * @return the allocator
     */
    public static HybridAllocator withThreshold(long heapThreshold) {
        return new HybridAllocator(heapThreshold, HeapAllocator.INSTANCE, NativeAllocator.automatic());
    }

    /**
     * Returns the process-wide allocator configured from system properties.
     *
     * @return the default allocator
     * @throws MemoryException if {@value #THRESHOLD_PROPERTY} is not a non-negative number
     */
    public static HybridAllocator defaultInstance() {
        HybridAllocator a = defaultInstance;
        if (a == null) {
            synchronized (HybridAllocator.class) {
                a = defaultInstance;
                if (a == null) {
                    a = withThreshold(parseThreshold(System.getProperty(THRESHOLD_PROPERTY)));
                    defaultInstance = a;
                }
            }
        }
        return a;
    }

    static long parseThreshold(String value) {
        if (value == null || value.isBlank()) {
            return DEFAULT_HEAP_THRESHOLD;
        }
        try {
            long v = Long.parseLong(value.strip());
            if (v < 0) {
                throw new MemoryException(THRESHOLD_PROPERTY + " must be >= 0, was " + v);
            }
            return v;
        } catch (NumberFormatException e) {
            throw new MemoryException(THRESHOLD_PROPERTY + " must be a number of bytes, was '" + value + "'", e);
        }
    }

    /** @return the largest request, in bytes, served from the heap */
    public long heapThreshold() {
        return heapThreshold;
    }

    @Override
    public Buffer allocate(long byteSize) {
        if (byteSize <= heapThreshold || byteSize == 0) {
            if (byteSize <= HeapAllocator.MAX_BYTES) {
                return heap.allocate(byteSize);
            }
        }
        return offHeap.allocate(byteSize);
    }

    @Override
    public String toString() {
        return "HybridAllocator[heapThreshold=" + heapThreshold + "]";
    }
}
