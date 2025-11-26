package org.numjvm.ndarray;

import java.util.Arrays;

import org.numjvm.exception.ShapeException;

/**
 * Immutable byte strides: the distance in bytes between neighbouring elements along
 * each axis.
 *
 * <p>Strides may be positive, zero (every index along the axis maps to the same
 * element, as in a broadcast view) or negative (the axis runs backwards through
 * memory, as in {@code a[::-1]}).
 *
 * <p>This class also holds the layout calculations that depend on strides:
 * C and Fortran order, contiguity tests that follow NumPy's rules, and the byte
 * extent an array may touch.
 */
public final class Strides {

    private static final Strides SCALAR = new Strides(new long[0]);

    final long[] values;

    private Strides(long[] values) {
        this.values = values;
    }

    /**
     * Creates strides from byte distances.
     *
     * @param bytes one stride per axis; copied
     * @return the strides
     */
    public static Strides of(long... bytes) {
        return bytes.length == 0 ? SCALAR : new Strides(bytes.clone());
    }

    /**
     * Wraps an array the caller will not modify afterwards.
     */
    static Strides ofOwned(long[] bytes) {
        return bytes.length == 0 ? SCALAR : new Strides(bytes);
    }

    /**
     * Strides of a C-contiguous (row-major) array: the last axis varies fastest.
     *
     * <p>Zero-length dimensions are treated as length 1, as NumPy does, so the strides
     * of an empty array are still those of a non-empty array of similar shape.
     *
     * @param shape    the shape
     * @param itemSize bytes per element
     * @return the strides, e.g. {@code [160, 40, 8]} for shape {@code [3, 4, 5]} and 8-byte items
     */
    public static Strides cContiguous(Shape shape, int itemSize) {
        return ofOwned(cStrides(shape.dims, itemSize));
    }

    private static ShapeException tooLarge(long[] dims, long itemSize, ArithmeticException e) {
        return new ShapeException("byte size of shape " + Arrays.toString(dims) + " with " + itemSize
                + "-byte items overflows long", e);
    }

    /**
     * Strides of a Fortran-contiguous (column-major) array: the first axis varies
     * fastest.
     *
     * @param shape    the shape
     * @param itemSize bytes per element
     * @return the strides, e.g. {@code [8, 24, 96]} for shape {@code [3, 4, 5]} and 8-byte items
     */
    public static Strides fContiguous(Shape shape, int itemSize) {
        return ofOwned(fStrides(shape.dims, itemSize));
    }

    static long[] cStrides(long[] dims, long itemSize) {
        long[] s = new long[dims.length];
        long acc = itemSize;
        try {
            for (int i = dims.length - 1; i >= 0; i--) {
                s[i] = acc;
                acc = Math.multiplyExact(acc, Math.max(1, dims[i]));
            }
        } catch (ArithmeticException e) {
            throw tooLarge(dims, itemSize, e);
        }
        return s;
    }

    static long[] fStrides(long[] dims, long itemSize) {
        long[] s = new long[dims.length];
        long acc = itemSize;
        try {
            for (int i = 0; i < dims.length; i++) {
                s[i] = acc;
                acc = Math.multiplyExact(acc, Math.max(1, dims[i]));
            }
        } catch (ArithmeticException e) {
            throw tooLarge(dims, itemSize, e);
        }
        return s;
    }

    /**
     * Returns the number of axes.
     *
     * @return the rank
     */
    public int rank() {
        return values.length;
    }

    /**
     * Returns the stride of one axis.
     *
     * @param axis the axis; negative values count from the end
     * @return the stride in bytes
     */
    public long stride(int axis) {
        return values[Shape.normalizeAxis(axis, values.length)];
    }

    /**
     * Returns a copy of the strides.
     *
     * @return a new array of byte strides
     */
    public long[] values() {
        return values.clone();
    }

    /**
     * Tells whether an array with these strides is C-contiguous.
     *
     * <p>NumPy's rules ({@code NUMPY_COMPATIBILITY}): axes of length 1 are ignored, and an
     * array with no elements is contiguous in both orders.
     *
     * @param shape    the array shape
     * @param strides  the array strides
     * @param itemSize bytes per element
     * @return {@code true} if elements are adjacent in row-major order
     */
    public static boolean isCContiguous(Shape shape, Strides strides, int itemSize) {
        return isCContiguous(shape.dims, strides.values, itemSize);
    }

    /**
     * Tells whether an array with these strides is Fortran-contiguous, by the same
     * rules as {@link #isCContiguous(Shape, Strides, int)}.
     *
     * @param shape    the array shape
     * @param strides  the array strides
     * @param itemSize bytes per element
     * @return {@code true} if elements are adjacent in column-major order
     */
    public static boolean isFContiguous(Shape shape, Strides strides, int itemSize) {
        return isFContiguous(shape.dims, strides.values, itemSize);
    }

    static boolean isCContiguous(long[] dims, long[] strides, long itemSize) {
        if (hasZero(dims)) {
            return true;
        }
        long expected = itemSize;
        for (int i = dims.length - 1; i >= 0; i--) {
            if (dims[i] != 1) {
                if (strides[i] != expected) {
                    return false;
                }
                expected *= dims[i];
            }
        }
        return true;
    }

    static boolean isFContiguous(long[] dims, long[] strides, long itemSize) {
        if (hasZero(dims)) {
            return true;
        }
        long expected = itemSize;
        for (int i = 0; i < dims.length; i++) {
            if (dims[i] != 1) {
                if (strides[i] != expected) {
                    return false;
                }
                expected *= dims[i];
            }
        }
        return true;
    }

    private static boolean hasZero(long[] dims) {
        for (long d : dims) {
            if (d == 0) {
                return true;
            }
        }
        return false;
    }

    /**
     * Computes the bytes an array may touch, relative to the start of its buffer.
     *
     * <p>With {@code offset} the byte position of element {@code [0, ..., 0]}, the
     * lowest byte is reached by taking index {@code dim - 1} on every axis with a
     * negative stride, and the highest by taking it on every axis with a positive
     * stride. The cost is O(rank); this is what lets an array validate all its element
     * addresses once, at construction (ADR-002).
     *
     * @param shape    the array shape
     * @param strides  the array strides; same rank as the shape
     * @param offset   byte offset of the first logical element
     * @param itemSize bytes per element
     * @return the extent; empty if the shape has no elements
     * @throws ShapeException if ranks differ or the computation overflows {@code long}
     */
    public static ByteExtent extent(Shape shape, Strides strides, long offset, int itemSize) {
        return extent(shape.dims, strides.values, offset, itemSize);
    }

    static ByteExtent extent(long[] dims, long[] strides, long offset, long itemSize) {
        if (dims.length != strides.length) {
            throw new ShapeException("shape has rank " + dims.length + " but strides have rank " + strides.length);
        }
        if (hasZero(dims)) {
            return new ByteExtent(offset, offset);
        }
        try {
            long low = offset;
            long high = offset;
            for (int i = 0; i < dims.length; i++) {
                long span = Math.multiplyExact(dims[i] - 1, strides[i]);
                if (span < 0) {
                    low = Math.addExact(low, span);
                } else {
                    high = Math.addExact(high, span);
                }
            }
            return new ByteExtent(low, Math.addExact(high, itemSize));
        } catch (ArithmeticException e) {
            throw new ShapeException("byte extent of shape " + Arrays.toString(dims) + " with strides "
                    + Arrays.toString(strides) + " overflows long", e);
        }
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof Strides s && Arrays.equals(values, s.values);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(values);
    }

    /**
     * Formats the strides like a list of byte distances, e.g. {@code [24, 8]}.
     */
    @Override
    public String toString() {
        return Arrays.toString(values);
    }
}
