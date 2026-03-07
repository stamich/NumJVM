package org.numjvm.jit;

/** Internal interface implemented by hidden generated expression classes. */
interface GeneratedKernel {
    int version();
    double evaluate(double[] inputs);
}
