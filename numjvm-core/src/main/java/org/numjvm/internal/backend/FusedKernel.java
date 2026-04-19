package org.numjvm.internal.backend;

import java.lang.foreign.MemorySegment;

/** A compiled expression for eligible memory runs. */
@FunctionalInterface
public interface FusedKernel {
    /** Runs one validated logical run.
     * @param segments input segments followed by output segment
     * @param offsets byte offsets for each segment
     * @param strides byte strides for each segment
     * @param count number of elements
     */
    void run(MemorySegment[] segments, long[] offsets, long[] strides, long count);
}
