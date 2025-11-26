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
 *
 * <p>An optional soft limit on live native bytes ({@value #SOFT_LIMIT_PROPERTY}, unset by
 * default) sends requests that would exceed it to the heap instead, as long as they fit
 * in a {@code long[]}. The hard bound on native memory remains the JDK's
 * {@code -XX:MaxDirectMemorySize}.
 */
public final class HybridAllocator implements MemoryAllocator {

    /**
     * System property holding the heap threshold in bytes.
     */
    public static final String THRESHOLD_PROPERTY = "numjvm.memory.heapThreshold";

    /**
     * Default heap threshold: 1 MiB.
     */
    public static final long DEFAULT_HEAP_THRESHOLD = 1L << 20;

    /**
     * System property holding the optional soft limit on live native bytes.
     */
    public static final String SOFT_LIMIT_PROPERTY = "numjvm.memory.nativeSoftLimit";

    /**
     * Value of {@link #nativeSoftLimit()} when no soft limit is set.
     */
    public static final long NO_LIMIT = -1;

    private static volatile HybridAllocator defaultInstance;

    private final long heapThreshold;
    private final long nativeSoftLimit;
    private final MemoryAllocator heap;
    private final MemoryAllocator offHeap;
    private final NativeMemoryTracker tracker;

    HybridAllocator(long heapThreshold, long nativeSoftLimit, MemoryAllocator heap, MemoryAllocator offHeap,
                    NativeMemoryTracker tracker) {
        if (heapThreshold < 0) {
            throw new MemoryException("heap threshold must be >= 0, was " + heapThreshold);
        }
        if (nativeSoftLimit < NO_LIMIT) {
            throw new MemoryException("native soft limit must be >= 0, was " + nativeSoftLimit);
        }
        this.heapThreshold = heapThreshold;
        this.nativeSoftLimit = nativeSoftLimit;
        this.heap = heap;
        this.offHeap = offHeap;
        this.tracker = tracker;
    }

    /**
     * Creates an allocator with an explicit threshold.
     *
     * @param heapThreshold largest request, in bytes, served from the heap
     * @return the allocator
     */
    public static HybridAllocator withThreshold(long heapThreshold) {
        return new HybridAllocator(heapThreshold, NO_LIMIT, HeapAllocator.INSTANCE, NativeAllocator.automatic(),
                NativeMemoryTracker.GLOBAL);
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
                    a = new HybridAllocator(
                            parseThreshold(System.getProperty(THRESHOLD_PROPERTY)),
                            parseSoftLimit(System.getProperty(SOFT_LIMIT_PROPERTY)),
                            HeapAllocator.INSTANCE, NativeAllocator.automatic(), NativeMemoryTracker.GLOBAL);
                    defaultInstance = a;
                }
            }
        }
        return a;
    }

    static long parseThreshold(String value) {
        return value == null || value.isBlank() ? DEFAULT_HEAP_THRESHOLD : parseBytes(THRESHOLD_PROPERTY, value);
    }

    static long parseSoftLimit(String value) {
        return value == null || value.isBlank() ? NO_LIMIT : parseBytes(SOFT_LIMIT_PROPERTY, value);
    }

    private static long parseBytes(String property, String value) {
        try {
            long v = Long.parseLong(value.strip());
            if (v < 0) {
                throw new MemoryException(property + " must be >= 0, was " + v);
            }
            return v;
        } catch (NumberFormatException e) {
            throw new MemoryException(property + " must be a number of bytes, was '" + value + "'", e);
        }
    }

    /**
     * Returns the soft limit on live native bytes, or {@link #NO_LIMIT}.
     *
     * @return the soft limit on live native bytes, or {@link #NO_LIMIT}
     */
    public long nativeSoftLimit() {
        return nativeSoftLimit;
    }

    /**
     * Returns the largest request, in bytes, served from the heap.
     *
     * @return the largest request, in bytes, served from the heap
     */
    public long heapThreshold() {
        return heapThreshold;
    }

    @Override
    public Buffer allocate(long byteSize) {
        boolean fitsHeap = byteSize <= HeapAllocator.MAX_BYTES;
        if (fitsHeap && (byteSize <= heapThreshold || overSoftLimit(byteSize))) {
            return heap.allocate(byteSize);
        }
        return offHeap.allocate(byteSize);
    }

    private boolean overSoftLimit(long byteSize) {
        return nativeSoftLimit != NO_LIMIT && tracker.liveBytes() + byteSize > nativeSoftLimit;
    }

    @Override
    public String toString() {
        return "HybridAllocator[heapThreshold=" + heapThreshold
                + (nativeSoftLimit == NO_LIMIT ? "" : ", nativeSoftLimit=" + nativeSoftLimit) + "]";
    }
}
