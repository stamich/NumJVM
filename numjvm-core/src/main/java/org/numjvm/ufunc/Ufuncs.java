package org.numjvm.ufunc;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.numjvm.dtype.DType;

/**
 * All of NumJVM's ufuncs.
 *
 * <p>Each constant follows NumPy's ufunc of the same name; the dtype rules are those of
 * ADR-005 and the edge cases (overflow, division by zero, NaN) those of decision D5. Integer
 * arithmetic wraps on overflow.
 */
public final class Ufuncs {

    private Ufuncs() {
    }

    /**
     * {@code x + y}; logical or for bool ({@code ARRAY_API}: {@code add}).
     */
    public static final BinaryUfunc ADD = BinaryUfunc.builder("add").build();

    /**
     * {@code x - y} ({@code ARRAY_API}: {@code subtract}). Not defined for two bool operands,
     * as in NumPy; use a logical or bitwise xor instead.
     */
    public static final BinaryUfunc SUBTRACT = BinaryUfunc.builder("subtract")
            .unsupported(DType.BOOL, "boolean subtract is not supported; use LOGICAL_XOR or BITWISE_XOR instead")
            .build();

    /**
     * {@code x * y}; logical and for bool ({@code ARRAY_API}: {@code multiply}).
     */
    public static final BinaryUfunc MULTIPLY = BinaryUfunc.builder("multiply").build();

    /**
     * Returns every ufunc, in declaration order.
     *
     * @return the ufuncs
     */
    public static List<Ufunc> all() {
        return List.copyOf(ByName.MAP.values());
    }

    /**
     * Looks a ufunc up by its NumPy name, e.g. {@code "add"}.
     *
     * @param name the name
     * @return the ufunc, or empty if NumJVM has no ufunc of that name
     */
    public static Optional<Ufunc> byName(String name) {
        return Optional.ofNullable(ByName.MAP.get(name));
    }

    /**
     * Built on first use, after all constants are initialised. Every public constant must be
     * listed here (UfuncsTest checks it).
     */
    private static final class ByName {
        static final Map<String, Ufunc> MAP = build(List.of(
                ADD, SUBTRACT, MULTIPLY));

        private static Map<String, Ufunc> build(List<Ufunc> all) {
            Map<String, Ufunc> m = new LinkedHashMap<>();
            for (Ufunc u : all) {
                m.put(u.name(), u);
            }
            return java.util.Collections.unmodifiableMap(m);
        }
    }
}
