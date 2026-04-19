package org.numjvm.vector;

import org.numjvm.internal.backend.FusedGraph;
import org.numjvm.internal.backend.FusedKernel;
import org.numjvm.internal.backend.FusedProvider;

/** Optional fused-vector service, independently discoverable from scalar JIT. */
public final class FusedVectorProvider implements FusedProvider {
    /** Verifies that the incubating Vector API is resolved. */
    public FusedVectorProvider() { jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED.length(); }
    @Override public String name() { return "jdk-vector"; }
    @Override public FusedKernel compile(FusedGraph graph) { return null; }
}
