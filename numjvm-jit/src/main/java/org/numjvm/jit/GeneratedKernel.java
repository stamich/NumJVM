package org.numjvm.jit;

/** Internal interface implemented by hidden generated expression classes. */
interface GeneratedKernel {
    int version();
    double evaluate(double[] inputs);
    void run(java.lang.foreign.MemorySegment[] segments, long[] offsets, long[] strides, long size);
}
