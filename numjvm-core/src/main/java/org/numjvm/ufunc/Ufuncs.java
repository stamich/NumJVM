package org.numjvm.ufunc;

/**
 * All of NumJVM's ufuncs.
 *
 * <p>Each constant follows NumPy's ufunc of the same name; the dtype rules are those of
 * ADR-005 and the edge cases (overflow, division by zero, NaN) those of decision D5.
 */
public final class Ufuncs {

    private Ufuncs() {
    }

    /**
     * {@code x + y}; logical or for bool ({@code ARRAY_API}: {@code add}).
     */
    public static final BinaryUfunc ADD = BinaryUfunc.builder("add").build();
}
