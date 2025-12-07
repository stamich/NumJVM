package org.numjvm.ufunc;

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import org.numjvm.dtype.DType;
import org.numjvm.dtype.TypePromotion;
import org.numjvm.exception.DTypeException;
import org.numjvm.internal.loops.KernelRegistry;
import org.numjvm.internal.loops.LoopSignature;

/**
 * An element-wise operation over arrays: NumPy's {@code ufunc}.
 *
 * <p>Instances are the constants of {@link Ufuncs}; they are immutable and thread-safe.
 */
public abstract sealed class Ufunc permits UnaryUfunc, BinaryUfunc {

    private final String name;
    private final String loopName;

    Ufunc(String name, String loopName) {
        this.name = Objects.requireNonNull(name, "name");
        this.loopName = Objects.requireNonNull(loopName, "loopName");
    }

    /**
     * Returns the NumPy name of this ufunc, e.g. {@code "add"}.
     *
     * @return the name
     */
    public final String name() {
        return name;
    }

    /**
     * Returns the number of inputs.
     *
     * @return 1 or 2
     */
    public abstract int nin();

    /**
     * Returns the loop signatures this ufunc can run, in the order type resolution
     * searches them, e.g. {@code "int8,int8->int8"} (NumPy's {@code ufunc.types}, with dtype
     * names).
     *
     * @return the signatures
     */
    public final List<String> types() {
        return KernelRegistry.signatures(loopName).stream()
                .map(s -> s.inputs().stream().map(DType::typeName).collect(Collectors.joining(","))
                        + "->" + s.output().typeName())
                .toList();
    }

    final String loopName() {
        return loopName;
    }

    /**
     * The first registered loop whose input dtype {@code common} casts to safely (ADR-005).
     */
    final LoopSignature firstSafeLoop(DType common) {
        LoopSignature exact = nin() == 1 ? KernelRegistry.find(loopName, common)
                : KernelRegistry.find(loopName, common, common);
        if (exact != null) {
            return exact;
        }
        for (LoopSignature s : KernelRegistry.signatures(loopName)) {
            if (TypePromotion.canCast(common, s.inputs().getFirst())) {
                return s;
            }
        }
        return null;
    }

    final DTypeException noLoop(DType... inputs) {
        String types = java.util.Arrays.stream(inputs).map(DType::typeName).collect(Collectors.joining(", "));
        return new DTypeException("ufunc '" + name + "' has no loop for (" + types + ")");
    }

    @Override
    public String toString() {
        return "<ufunc '" + name + "'>";
    }
}
