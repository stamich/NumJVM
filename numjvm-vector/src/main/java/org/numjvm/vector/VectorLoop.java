package org.numjvm.vector;

import java.lang.foreign.MemorySegment;

/** A generated loop for the whole vector portion of a contiguous run. */
interface VectorLoop {
    void runVectors(MemorySegment[] segments, long[] offsets, long[] strides, long count, int lanes);
}
