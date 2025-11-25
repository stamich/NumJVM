package org.numjvm.ndarray;

/**
 * Memory-layout and access properties of an array, computed when the array is created.
 *
 * <p>The flags follow NumPy's {@code ndarray.flags} ({@code NUMPY_COMPATIBILITY}):
 * <ul>
 *   <li>{@code C_CONTIGUOUS}: elements are adjacent in row-major order. Axes of length 1
 *       are ignored, and an empty array is contiguous in both orders.</li>
 *   <li>{@code F_CONTIGUOUS}: the same for column-major order.</li>
 *   <li>{@code WRITEABLE}: elements may be written through this array. Broadcast views
 *       are never writeable.</li>
 *   <li>{@code ALIGNED}: every element address is a multiple of the dtype's alignment.
 *       NumJVM's accessors do not depend on it (ADR-002); it matters for SIMD and native
 *       code.</li>
 * </ul>
 */
public final class ArrayFlags {

    /** Bit set when the array is C-contiguous. */
    public static final int C_CONTIGUOUS = 1;
    /** Bit set when the array is Fortran-contiguous. */
    public static final int F_CONTIGUOUS = 1 << 1;
    /** Bit set when elements may be written. */
    public static final int WRITEABLE = 1 << 2;
    /** Bit set when every element is aligned to the dtype's alignment. */
    public static final int ALIGNED = 1 << 3;

    private final int bits;

    private ArrayFlags(int bits) {
        this.bits = bits;
    }

    static ArrayFlags of(int bits) {
        return new ArrayFlags(bits);
    }

    /**
     * Computes the flags of an array.
     *
     * @param dims          the shape
     * @param strides       byte strides
     * @param offset        byte offset of element {@code [0, ..., 0]}
     * @param itemSize      element size
     * @param alignment     the dtype's alignment
     * @param baseAlignment guaranteed alignment of byte 0 of the buffer
     * @param writeable     whether writes are allowed
     * @return the flag bits
     */
    static int compute(long[] dims, long[] strides, long offset, int itemSize, int alignment,
                       int baseAlignment, boolean writeable) {
        int bits = 0;
        if (Strides.isCContiguous(dims, strides, itemSize)) {
            bits |= C_CONTIGUOUS;
        }
        if (Strides.isFContiguous(dims, strides, itemSize)) {
            bits |= F_CONTIGUOUS;
        }
        if (writeable) {
            bits |= WRITEABLE;
        }
        if (isAligned(dims, strides, offset, alignment, baseAlignment)) {
            bits |= ALIGNED;
        }
        return bits;
    }

    /** NumPy's rule: the offset and every stride of an axis longer than 1 are multiples of the alignment. */
    private static boolean isAligned(long[] dims, long[] strides, long offset, int alignment, int baseAlignment) {
        if (baseAlignment < alignment) {
            return false;
        }
        long combined = offset;
        for (int i = 0; i < dims.length; i++) {
            if (dims[i] == 0) {
                return true;
            }
            if (dims[i] > 1) {
                combined |= strides[i];
            }
        }
        return combined % alignment == 0;
    }

    /**
     * Returns {@code true} if the array is C-contiguous.
     *
     * @return {@code true} if the array is C-contiguous
     */
    public boolean isCContiguous() {
        return (bits & C_CONTIGUOUS) != 0;
    }

    /**
     * Returns {@code true} if the array is Fortran-contiguous.
     *
     * @return {@code true} if the array is Fortran-contiguous
     */
    public boolean isFContiguous() {
        return (bits & F_CONTIGUOUS) != 0;
    }

    /**
     * Returns {@code true} if the array is C- or Fortran-contiguous.
     *
     * @return {@code true} if the array is C- or Fortran-contiguous
     */
    public boolean isContiguous() {
        return (bits & (C_CONTIGUOUS | F_CONTIGUOUS)) != 0;
    }

    /**
     * Returns {@code true} if elements may be written.
     *
     * @return {@code true} if elements may be written
     */
    public boolean isWriteable() {
        return (bits & WRITEABLE) != 0;
    }

    /**
     * Returns {@code true} if every element is aligned.
     *
     * @return {@code true} if every element is aligned
     */
    public boolean isAligned() {
        return (bits & ALIGNED) != 0;
    }

    /**
     * Returns the raw flag bits, a combination of the constants of this class.
     *
     * @return the raw flag bits, a combination of the constants of this class
     */
    public int bits() {
        return bits;
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof ArrayFlags f && f.bits == bits;
    }

    @Override
    public int hashCode() {
        return bits;
    }

    /**
     * Formats the flags like NumPy's {@code ndarray.flags}.
     */
    @Override
    public String toString() {
        return "  C_CONTIGUOUS : " + py(isCContiguous()) + "\n"
                + "  F_CONTIGUOUS : " + py(isFContiguous()) + "\n"
                + "  WRITEABLE : " + py(isWriteable()) + "\n"
                + "  ALIGNED : " + py(isAligned());
    }

    private static String py(boolean b) {
        return b ? "True" : "False";
    }
}
