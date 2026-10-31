package org.numjvm.internal.backend;

/** Optional trusted compiler for typed fused expressions. */
public interface FusedProvider {
    /** @return stable diagnostic provider name */
    String name();
    /** Compiles a supported graph; returns null for unsupported graphs.
     * @param graph immutable graph with resolved node types
     * @return kernel or null when unsupported
     */
    FusedKernel compile(FusedGraph graph);
}
