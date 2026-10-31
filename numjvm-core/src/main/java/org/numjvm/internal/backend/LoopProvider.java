package org.numjvm.internal.backend;

import org.numjvm.dtype.DType;
import org.numjvm.internal.loops.BinaryLoop;
import org.numjvm.internal.loops.UnaryLoop;

/** Trusted internal CPU extension; never exposed as a public array memory accessor. */
public interface LoopProvider {
    /** Returns stable diagnostic name.
     * @return stable diagnostic name
     */
    String name();

    /**
     * Wraps an already-resolved operation; unsupported runs keep the scalar path.
     * @param operation registry operation name
     * @param type resolved input dtype
     * @param scalar original scalar implementation
     * @return accelerated wrapper, or scalar when unsupported
     */
    default UnaryLoop unary(String operation, DType type, UnaryLoop scalar) { return scalar; }

    /**
     * Wraps an already-resolved operation; unsupported runs keep the scalar path.
     * @param operation registry operation name
     * @param type resolved input dtype
     * @param scalar original scalar implementation
     * @return accelerated wrapper, or scalar when unsupported
     */
    default BinaryLoop binary(String operation, DType type, BinaryLoop scalar) { return scalar; }

    /** Returns a supported contiguous real dot implementation, or null.
     * @param type resolved common input and output type
     * @return vector contraction kernel or null
     */
    default DotLoop dot(DType type) { return null; }
}
