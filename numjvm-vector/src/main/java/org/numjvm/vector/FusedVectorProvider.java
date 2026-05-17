package org.numjvm.vector;

import org.numjvm.internal.backend.FusedGraph;
import org.numjvm.internal.backend.FusedKernel;
import org.numjvm.internal.backend.FusedProvider;

/** Optional fused-vector service, independently discoverable from scalar JIT. */
public final class FusedVectorProvider implements FusedProvider {
    /** Verifies that the incubating Vector API is resolved. */
    public FusedVectorProvider() { jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED.length(); }
    @Override public String name() { return "jdk-vector"; }
    @Override public FusedKernel compile(FusedGraph graph) {
        if (graph.inputTypes().stream().allMatch(t -> t == org.numjvm.dtype.DType.FLOAT32)) {
            var expression = (FloatExpression) VectorExpressionCompiler.compile(graph, org.numjvm.dtype.DType.FLOAT32);
            return expression == null ? null : new FloatFusedKernel(expression);
        }
        var expression = (DoubleExpression) VectorExpressionCompiler.compile(graph, org.numjvm.dtype.DType.FLOAT64);
        if (expression != null) return new DoubleFusedKernel(expression);
        var mixed = MixedExpressionCompiler.compile(graph);
        return mixed == null ? null : new MixedFusedKernel(mixed, graph.inputTypes());
    }
}
