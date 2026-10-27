package org.numjvm.memory;

/**
 * Snapshot of NumJVM's memory usage and configuration, from {@code NJ.memoryStats()}.
 *
 * <p>Only native (off-heap) memory is counted; heap arrays are ordinary Java objects and
 * show up in the usual heap metrics. Native memory of arrays that are no longer
 * reachable is counted as live until the garbage collector has processed them.
 *
 * @param liveNativeBytes   native bytes allocated by NumJVM and not yet released
 * @param peakNativeBytes   highest value of {@code liveNativeBytes} since start-up
 * @param nativeAllocations number of native allocations since start-up
 * @param heapThreshold     largest automatic allocation, in bytes, placed on the heap
 * @param nativeSoftLimit   soft limit on live native bytes, or {@code -1} if none is set
 */
public record MemoryStats(
        long liveNativeBytes,
        long peakNativeBytes,
        long nativeAllocations,
        long heapThreshold,
        long nativeSoftLimit) {

    /**
     * Tells whether a soft limit is configured.
     *
     * @return {@code true} if {@code -Dnumjvm.memory.nativeSoftLimit} is set
     */
    public boolean hasNativeSoftLimit() {
        return nativeSoftLimit >= 0;
    }
}
