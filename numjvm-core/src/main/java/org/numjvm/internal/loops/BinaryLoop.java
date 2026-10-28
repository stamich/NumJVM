package org.numjvm.internal.loops;

import java.lang.foreign.MemorySegment;

/**
 * An inner loop with two inputs and one output: {@code out[k] = f(a[k], b[k])} for
 * {@code k} in {@code [0, n)}, where element {@code k} of an operand is at byte
 * {@code offset + k * stride}.
 */
@FunctionalInterface
public interface BinaryLoop {

    /**
     * Processes one run.
     *
     * @param a   first input memory
     * @param pa  byte offset of the first input's first element
     * @param sa  first input byte stride (0 for a broadcast input)
     * @param b   second input memory
     * @param pb  byte offset of the second input's first element
     * @param sb  second input byte stride (0 for a broadcast input)
     * @param out output memory
     * @param po  byte offset of the output's first element
     * @param so  output byte stride
     * @param n   number of elements
     */
    void run(MemorySegment a, long pa, long sa, MemorySegment b, long pb, long sb, MemorySegment out, long po,
             long so, long n);
}
