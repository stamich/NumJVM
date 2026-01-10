package org.numjvm.internal.backend;

import org.numjvm.dtype.DType;
import org.numjvm.internal.loops.BinaryLoop;
import org.numjvm.internal.loops.UnaryLoop;

/** Trusted internal CPU extension; never exposed as a public array memory accessor. */
public interface LoopProvider {
    /** @return stable diagnostic name */
    String name();

    /** @return accelerated wrapper, or the original loop when unsupported */
    default UnaryLoop unary(String operation, DType type, UnaryLoop scalar) { return scalar; }

    /** @return accelerated wrapper, or the original loop when unsupported */
    default BinaryLoop binary(String operation, DType type, BinaryLoop scalar) { return scalar; }
}
