package org.numjvm.internal.loops;

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import org.numjvm.dtype.DType;

/**
 * The dtypes an inner loop is written for: one operation, its input dtypes and its output
 * dtype, e.g. {@code add(float64, float64) -> float64} or
 * {@code less(int8, int8) -> bool}.
 *
 * @param operation the operation name, as in NumPy ({@code "add"}, {@code "sqrt"}, ...)
 * @param inputs    input dtypes, one per input operand
 * @param output    output dtype
 */
public record LoopSignature(String operation, List<DType> inputs, DType output) {

    /**
     * Validates and copies the components.
     *
     * @param operation the operation name
     * @param inputs    input dtypes
     * @param output    output dtype
     */
    public LoopSignature {
        Objects.requireNonNull(operation, "operation");
        inputs = List.copyOf(inputs);
        Objects.requireNonNull(output, "output");
    }

    /**
     * Returns the number of inputs.
     *
     * @return the number of inputs
     */
    public int arity() {
        return inputs.size();
    }

    @Override
    public String toString() {
        return operation + inputs.stream().map(DType::typeName).collect(Collectors.joining(", ", "(", ")"))
                + " -> " + output.typeName();
    }
}
