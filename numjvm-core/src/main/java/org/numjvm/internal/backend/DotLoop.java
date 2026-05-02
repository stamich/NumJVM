package org.numjvm.internal.backend;

import java.lang.foreign.MemorySegment;

/** Internal contiguous real-vector contraction; result is written to a scalar output. */
@FunctionalInterface
public interface DotLoop {
    /** Runs a dot product over contiguous elements.
     * @param a left memory
     * @param ao left byte offset
     * @param b right memory
     * @param bo right byte offset
     * @param out destination memory
     * @param oo destination byte offset
     * @param count element count
     */
    void run(MemorySegment a, long ao, MemorySegment b, long bo, MemorySegment out, long oo, long count);
}
