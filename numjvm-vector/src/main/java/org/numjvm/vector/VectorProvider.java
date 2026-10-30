package org.numjvm.vector;

import org.numjvm.dtype.DType;
import org.numjvm.internal.backend.LoopProvider;
import org.numjvm.internal.loops.BinaryLoop;
import org.numjvm.internal.loops.UnaryLoop;

/** Optional service implementation; every unsupported run delegates to its scalar loop. */
public final class VectorProvider implements LoopProvider {
    /** Constructs the service provider. */
    public VectorProvider() {
        // Force incubator linkage during discovery so auto mode can diagnose an
        // installed but unresolved provider before selecting any numerical loop.
        jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED.length();
    }
    @Override public String name() { return "jdk-vector"; }
    @Override public UnaryLoop unary(String op, DType type, UnaryLoop scalar) {
        return GeneratedVectorKernels.unary(op, type, scalar);
    }
    @Override public BinaryLoop binary(String op, DType type, BinaryLoop scalar) {
        BinaryLoop vector = GeneratedVectorKernels.binary(op, type, scalar);
        if (type == DType.BOOL && (op.equals("logical_and") || op.equals("logical_or"))) {
            return VectorReductions.truth(op.equals("logical_and"), vector);
        }
        if ((op.equals("maximum") || op.equals("minimum")) && (type == DType.FLOAT32 || type == DType.FLOAT64)) {
            return type == DType.FLOAT32 ? VectorReductions.extremaFLOAT(op.equals("maximum"), vector)
                    : VectorReductions.extremaDOUBLE(op.equals("maximum"), vector);
        }
        if (op.equals("add") && (type == DType.INT32 || type == DType.INT64)) {
            return VectorReductions.integerSum(type, vector);
        }
        return op.equals("add") && (type == DType.FLOAT32 || type == DType.FLOAT64)
                ? VectorPairwise.wrap(type, scalar, vector) : vector;
    }
}
