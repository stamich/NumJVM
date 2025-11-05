package org.numjvm.dtype;

/**
 * Category of a {@link DType}, matching NumPy's {@code dtype.kind}.
 */
public enum DTypeKind {

    /** Boolean, NumPy kind {@code 'b'}. */
    BOOL('b'),

    /** Signed two's-complement integer, NumPy kind {@code 'i'}. */
    SIGNED_INTEGER('i'),

    /** Unsigned integer, NumPy kind {@code 'u'}. */
    UNSIGNED_INTEGER('u'),

    /** IEEE 754 (or bfloat16) floating point, NumPy kind {@code 'f'}. */
    FLOATING('f'),

    /** Complex number stored as two floating-point components, NumPy kind {@code 'c'}. */
    COMPLEX('c');

    private final char numpyKind;

    DTypeKind(char numpyKind) {
        this.numpyKind = numpyKind;
    }

    /**
     * Returns NumPy's single-character kind code.
     *
     * @return {@code 'b'}, {@code 'i'}, {@code 'u'}, {@code 'f'} or {@code 'c'}
     */
    public char numpyKind() {
        return numpyKind;
    }
}
