package org.numjvm.ufunc;

import org.numjvm.ndarray.NDArray;

/**
 * Reductions over axes ({@code ARRAY_API}: {@code sum}, {@code prod}, ...), each the
 * {@code reduce} of a ufunc with NumPy's dtype rules.
 *
 * <p>Every reduction has three forms: over all axes (a 0-D result), over one axis, and over
 * a list of axes with {@code keepdims}. In the last form {@code null} means all axes and an
 * empty array none. Negative axes count from the end.
 */
public final class Reductions {

    private Reductions() {
    }

    // ------------------------------------------------------------------ sum

    /**
     * Sums all elements ({@code ARRAY_API}: {@code sum}).
     *
     * <p>Bool and integers narrower than 64 bits are summed in int64 (unsigned: uint64), with
     * wrap-around; floating-point runs are summed with NumPy's pairwise algorithm, float16 and
     * bfloat16 in float32 precision. An empty sum is 0.
     *
     * @param x the array
     * @return a 0-D array
     */
    public static NDArray sum(NDArray x) {
        return Ufuncs.ADD.reduce(x);
    }

    /**
     * Sums over one axis; see {@link #sum(NDArray)}.
     *
     * @param x    the array
     * @param axis the axis
     * @return the sums, with that axis removed
     */
    public static NDArray sum(NDArray x, int axis) {
        return Ufuncs.ADD.reduce(x, axis);
    }

    /**
     * Sums over several axes; see {@link #sum(NDArray)}.
     *
     * @param x        the array
     * @param axes     the axes; {@code null} for all, an empty array for none
     * @param keepdims keep reduced axes with length 1
     * @return the sums
     */
    public static NDArray sum(NDArray x, int[] axes, boolean keepdims) {
        return Ufuncs.ADD.reduce(x, axes, keepdims);
    }

    // ------------------------------------------------------------------ prod

    /**
     * Multiplies all elements ({@code ARRAY_API}: {@code prod}).
     *
     * <p>Bool and integers narrower than 64 bits are multiplied in int64 (unsigned: uint64),
     * with wrap-around. An empty product is 1.
     *
     * @param x the array
     * @return a 0-D array
     */
    public static NDArray prod(NDArray x) {
        return Ufuncs.MULTIPLY.reduce(x);
    }

    /**
     * Multiplies over one axis; see {@link #prod(NDArray)}.
     *
     * @param x    the array
     * @param axis the axis
     * @return the products, with that axis removed
     */
    public static NDArray prod(NDArray x, int axis) {
        return Ufuncs.MULTIPLY.reduce(x, axis);
    }

    /**
     * Multiplies over several axes; see {@link #prod(NDArray)}.
     *
     * @param x        the array
     * @param axes     the axes; {@code null} for all, an empty array for none
     * @param keepdims keep reduced axes with length 1
     * @return the products
     */
    public static NDArray prod(NDArray x, int[] axes, boolean keepdims) {
        return Ufuncs.MULTIPLY.reduce(x, axes, keepdims);
    }

    // ------------------------------------------------------------------ max

    /**
     * The largest element ({@code ARRAY_API}: {@code max}). NaN propagates: any NaN
     * gives NaN. The result keeps the input's dtype.
     *
     * @param x the array
     * @return a 0-D array
     * @throws org.numjvm.exception.ValueException if {@code x} is empty (there is no identity)
     */
    public static NDArray max(NDArray x) {
        return Ufuncs.MAXIMUM.reduce(x);
    }

    /**
     * The largest element along one axis; see {@link #max(NDArray)}.
     *
     * @param x    the array
     * @param axis the axis
     * @return the results, with that axis removed
     * @throws org.numjvm.exception.ValueException if the axis has length 0 and the result
     *                                             is not empty
     */
    public static NDArray max(NDArray x, int axis) {
        return Ufuncs.MAXIMUM.reduce(x, axis);
    }

    /**
     * The largest element over several axes; see {@link #max(NDArray)}.
     *
     * @param x        the array
     * @param axes     the axes; {@code null} for all, an empty array for none
     * @param keepdims keep reduced axes with length 1
     * @return the results
     */
    public static NDArray max(NDArray x, int[] axes, boolean keepdims) {
        return Ufuncs.MAXIMUM.reduce(x, axes, keepdims);
    }

    // ------------------------------------------------------------------ min

    /**
     * The smallest element ({@code ARRAY_API}: {@code min}). NaN propagates: any NaN
     * gives NaN. The result keeps the input's dtype.
     *
     * @param x the array
     * @return a 0-D array
     * @throws org.numjvm.exception.ValueException if {@code x} is empty (there is no identity)
     */
    public static NDArray min(NDArray x) {
        return Ufuncs.MINIMUM.reduce(x);
    }

    /**
     * The smallest element along one axis; see {@link #min(NDArray)}.
     *
     * @param x    the array
     * @param axis the axis
     * @return the results, with that axis removed
     * @throws org.numjvm.exception.ValueException if the axis has length 0 and the result
     *                                             is not empty
     */
    public static NDArray min(NDArray x, int axis) {
        return Ufuncs.MINIMUM.reduce(x, axis);
    }

    /**
     * The smallest element over several axes; see {@link #min(NDArray)}.
     *
     * @param x        the array
     * @param axes     the axes; {@code null} for all, an empty array for none
     * @param keepdims keep reduced axes with length 1
     * @return the results
     */
    public static NDArray min(NDArray x, int[] axes, boolean keepdims) {
        return Ufuncs.MINIMUM.reduce(x, axes, keepdims);
    }
}
