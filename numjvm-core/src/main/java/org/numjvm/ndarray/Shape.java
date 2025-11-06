package org.numjvm.ndarray;

import java.util.Arrays;

import org.numjvm.exception.ShapeException;

/**
 * Immutable list of array dimensions.
 *
 * <p>A shape of rank 0 ({@code []}) describes a scalar array with one element. A shape
 * containing a zero ({@code [0, 10]}) describes an array with no elements.
 *
 * <p>Validation: every dimension is {@code >= 0}, the rank is at most {@link #MAX_RANK},
 * and the product of the dimensions, with zeros counted as ones, fits in a
 * {@code long}. The last rule makes stride computation for every shape overflow-free,
 * including zero-size shapes such as {@code [0, 2^62, 2^62]}, which are rejected.
 */
public final class Shape {

    /** Maximum number of dimensions, matching NumPy 2's {@code NPY_MAXDIMS}. */
    public static final int MAX_RANK = 64;

    private static final Shape SCALAR = new Shape(new long[0], 1);

    final long[] dims;
    private final long size;

    private Shape(long[] dims, long size) {
        this.dims = dims;
        this.size = size;
    }

    /**
     * Creates a shape.
     *
     * @param dimensions the dimensions; copied
     * @return the shape
     * @throws ShapeException if a dimension is negative, the rank exceeds
     *                        {@link #MAX_RANK}, or the element count overflows
     */
    public static Shape of(long... dimensions) {
        if (dimensions.length == 0) {
            return SCALAR;
        }
        return validated(dimensions.clone());
    }

    /**
     * Returns the rank-0 shape {@code []}.
     *
     * @return the scalar shape
     */
    public static Shape scalar() {
        return SCALAR;
    }

    /** Wraps an array the caller will not modify afterwards. */
    static Shape ofOwned(long[] dims) {
        return dims.length == 0 ? SCALAR : validated(dims);
    }

    private static Shape validated(long[] dims) {
        if (dims.length > MAX_RANK) {
            throw new ShapeException("rank " + dims.length + " exceeds the maximum of " + MAX_RANK);
        }
        long product = 1;
        boolean empty = false;
        for (int i = 0; i < dims.length; i++) {
            long d = dims[i];
            if (d < 0) {
                throw new ShapeException("negative dimension " + d + " at axis " + i + " in " + Arrays.toString(dims));
            }
            if (d == 0) {
                empty = true;
                continue;
            }
            try {
                product = Math.multiplyExact(product, d);
            } catch (ArithmeticException e) {
                throw new ShapeException("element count of shape " + Arrays.toString(dims) + " overflows long", e);
            }
        }
        return new Shape(dims, empty ? 0 : product);
    }

    /**
     * Returns the number of dimensions.
     *
     * @return the rank, 0 for a scalar
     */
    public int rank() {
        return dims.length;
    }

    /**
     * Returns the number of elements: the product of the dimensions.
     *
     * @return 1 for a scalar shape, 0 if any dimension is 0
     */
    public long size() {
        return size;
    }

    /**
     * Returns one dimension.
     *
     * @param axis the axis; negative values count from the end ({@code -1} is the last)
     * @return the dimension along {@code axis}
     * @throws ShapeException if the axis is out of range
     */
    public long dimension(int axis) {
        return dims[normalizeAxis(axis, dims.length)];
    }

    /**
     * Returns a copy of the dimensions.
     *
     * @return a new array
     */
    public long[] dimensions() {
        return dims.clone();
    }

    /**
     * Tells whether the shape has no elements.
     *
     * @return {@code true} if any dimension is 0
     */
    public boolean isEmpty() {
        return size == 0;
    }

    /**
     * Tells whether this is the rank-0 shape.
     *
     * @return {@code true} for {@code []}
     */
    public boolean isScalar() {
        return dims.length == 0;
    }

    /**
     * Converts a possibly negative axis into the range {@code [0, rank)}.
     *
     * @param axis an axis in {@code [-rank, rank)}
     * @param rank the number of dimensions
     * @return the normalised axis
     * @throws ShapeException if the axis is out of range
     */
    public static int normalizeAxis(int axis, int rank) {
        int a = axis < 0 ? axis + rank : axis;
        if (a < 0 || a >= rank) {
            throw new ShapeException("axis " + axis + " is out of bounds for rank " + rank);
        }
        return a;
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof Shape s && Arrays.equals(dims, s.dims);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(dims);
    }

    /**
     * Formats the shape like a list, e.g. {@code [2, 3]} or {@code []}.
     */
    @Override
    public String toString() {
        return Arrays.toString(dims);
    }
}
