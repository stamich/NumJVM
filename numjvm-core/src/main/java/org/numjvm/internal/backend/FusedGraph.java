package org.numjvm.internal.backend;

import java.util.List;
import java.util.Objects;
import org.numjvm.dtype.DType;

/** Immutable typed expression description with no array or owner references. */
public record FusedGraph(List<Node> nodes, List<DType> inputTypes) {
    /** Supported graph operations. */
    public enum Op { INPUT, CONSTANT, ADD, SUBTRACT, MULTIPLY, DIVIDE, NEGATIVE, ABSOLUTE, SQUARE, SQRT }
    /** A node's references are indices in topological order, or -1 when absent.
     * @param op operation
     * @param type resolved result dtype
     * @param slot input slot, or -1
     * @param bits raw constant bits
     * @param left left predecessor, or -1
     * @param right right predecessor, or -1
     */
    public record Node(Op op, DType type, int slot, long bits, int left, int right) {
        public Node { Objects.requireNonNull(op); Objects.requireNonNull(type); }
    }
    /** Copies caller-owned lists.
     * @param nodes nodes in topological order
     * @param inputTypes invocation input dtypes
     */
    public FusedGraph {
        nodes = List.copyOf(nodes);
        inputTypes = List.copyOf(inputTypes);
        if (nodes.isEmpty()) throw new IllegalArgumentException("empty fused graph");
    }
}
