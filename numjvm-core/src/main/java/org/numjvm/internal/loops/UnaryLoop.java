package org.numjvm.internal.loops;

import java.lang.foreign.MemorySegment;

/**
 * An inner loop with one input and one output: {@code out[k] = f(in[k])} for {@code k} in
 * {@code [0, n)}, where element {@code k} of an operand is at byte {@code offset + k * stride}.
 */
@FunctionalInterface
public interface UnaryLoop {

    /**
     * Processes one run.
     *
     * @param in  input memory
     * @param pi  byte offset of the input's first element
     * @param si  input byte stride (0 for a broadcast input)
     * @param out output memory
     * @param po  byte offset of the output's first element
     * @param so  output byte stride
     * @param n   number of elements
     */
    void run(MemorySegment in, long pi, long si, MemorySegment out, long po, long so, long n);
}
