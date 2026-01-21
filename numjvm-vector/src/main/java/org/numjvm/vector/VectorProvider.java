package org.numjvm.vector;

import org.numjvm.dtype.DType;
import org.numjvm.internal.backend.LoopProvider;
import org.numjvm.internal.loops.BinaryLoop;

/** Optional service implementation; every unsupported run delegates to its scalar loop. */
public final class VectorProvider implements LoopProvider {
    /** Constructs the service provider. */
    public VectorProvider() { }
    @Override public String name() { return "jdk-vector"; }
    @Override public BinaryLoop binary(String op, DType type, BinaryLoop scalar) {
        return GeneratedVectorKernels.binary(op, type, scalar);
    }
}
