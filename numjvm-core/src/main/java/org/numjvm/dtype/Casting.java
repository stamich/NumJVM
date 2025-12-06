package org.numjvm.dtype;

/**
 * How permissive a dtype conversion may be, as NumPy's {@code casting} argument
 * ({@code NUMPY_COMPATIBILITY}; see ADR-005).
 */
public enum Casting {

    /** Only the same dtype. */
    NO,

    /** Only the same dtype; NumJVM dtypes have no byte order, so this equals {@link #NO}. */
    EQUIV,

    /** Only conversions that preserve every value: {@code resultType(from, to) == to}. */
    SAFE,

    /**
     * Safe conversions, or conversions within a kind or to a higher kind in the order
     * bool, unsigned, signed, float, complex: float64 to float32 and int64 to float16 are
     * allowed, float to integer and complex to real are not.
     */
    SAME_KIND,

    /** Any conversion. */
    UNSAFE
}
