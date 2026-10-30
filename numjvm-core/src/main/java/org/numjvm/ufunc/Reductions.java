package org.numjvm.ufunc;

import org.numjvm.dtype.DType;
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

    // ------------------------------------------------------------------ mean

    /**
     * The arithmetic mean ({@code ARRAY_API}: {@code mean}), computed as NumPy does: the sum
     * divided by the number of elements reduced. Bool and integer arrays are summed in
     * float64 and give float64; float16 and bfloat16 are summed in float32; other dtypes are
     * summed in their own. The count is an int64, as NumPy's intp, so float32 sums are divided
     * in float64 and complex64 sums in complex128, and the result is rounded back to the
     * array's dtype. The mean of an empty array is NaN.
     *
     * @param x the array
     * @return a 0-D array
     */
    public static NDArray mean(NDArray x) {
        return mean(x, null, false);
    }

    /**
     * The mean along one axis; see {@link #mean(NDArray)}.
     *
     * @param x    the array
     * @param axis the axis
     * @return the means, with that axis removed
     */
    public static NDArray mean(NDArray x, int axis) {
        return mean(x, new int[]{axis}, false);
    }

    /**
     * The mean over several axes; see {@link #mean(NDArray)}.
     *
     * @param x        the array
     * @param axes     the axes; {@code null} for all, an empty array for none
     * @param keepdims keep reduced axes with length 1
     * @return the means
     */
    public static NDArray mean(NDArray x, int[] axes, boolean keepdims) {
        DType t = x.dtype();
        DType sumType = t.isBool() || t.isInteger() ? DType.FLOAT64
                : t == DType.FLOAT16 || t == DType.BFLOAT16 ? DType.FLOAT32 : t;
        boolean[] reduced = Reducer.reducedAxes(axes, x.ndim());
        long count = 1;
        for (int d = 0; d < reduced.length; d++) {
            if (reduced[d]) {
                count *= x.dim(d);
            }
        }
        org.numjvm.internal.memory.MemoryAllocator allocator = Operands.allocator(x);
        NDArray sum = t.isBool() || t.isInteger()
                ? IntegerMean.sum(x, reduced, keepdims, allocator, null)
                : Ufuncs.ADD.reduce(Operands.cast(x, sumType, allocator), axes, keepdims);
        // NumPy divides by the count as an intp, which is not a weak scalar: float32 sums are
        // divided in float64 and complex64 sums in complex128, then converted back
        NDArray n = Operands.weakLong(DType.INT64, count);
        NDArray mean = Ufuncs.DIVIDE.apply(sum, n);
        DType target = t.isBool() || t.isInteger() ? DType.FLOAT64 : t;
        if (sum.ndim() == 0) {
            // a scalar result is converted once, to the array's dtype
            return Operands.cast(mean, target, allocator);
        }
        // an array result is divided into the sum array (casting='unsafe'), then float16
        // and bfloat16 results are converted from it
        return Operands.cast(Operands.cast(mean, sumType, allocator), target, allocator);
    }

    // ------------------------------------------------------------------ any / all

    /**
     * Whether any element is true, that is non-zero ({@code ARRAY_API}: {@code any}). NaN
     * counts as true; an empty array gives {@code false}.
     *
     * @param x the array
     * @return a 0-D bool array
     */
    public static NDArray any(NDArray x) {
        return Ufuncs.LOGICAL_OR.reduce(x);
    }

    /**
     * Whether any element along one axis is true; see {@link #any(NDArray)}.
     *
     * @param x    the array
     * @param axis the axis
     * @return a bool array with that axis removed
     */
    public static NDArray any(NDArray x, int axis) {
        return Ufuncs.LOGICAL_OR.reduce(x, axis);
    }

    /**
     * Whether any element over several axes is true; see {@link #any(NDArray)}.
     *
     * @param x        the array
     * @param axes     the axes; {@code null} for all, an empty array for none
     * @param keepdims keep reduced axes with length 1
     * @return a bool array
     */
    public static NDArray any(NDArray x, int[] axes, boolean keepdims) {
        return Ufuncs.LOGICAL_OR.reduce(x, axes, keepdims);
    }

    /**
     * Whether every element is true, that is non-zero ({@code ARRAY_API}: {@code all}). NaN
     * counts as true; an empty array gives {@code true}.
     *
     * @param x the array
     * @return a 0-D bool array
     */
    public static NDArray all(NDArray x) {
        return Ufuncs.LOGICAL_AND.reduce(x);
    }

    /**
     * Whether every element along one axis is true; see {@link #all(NDArray)}.
     *
     * @param x    the array
     * @param axis the axis
     * @return a bool array with that axis removed
     */
    public static NDArray all(NDArray x, int axis) {
        return Ufuncs.LOGICAL_AND.reduce(x, axis);
    }

    /**
     * Whether every element over several axes is true; see {@link #all(NDArray)}.
     *
     * @param x        the array
     * @param axes     the axes; {@code null} for all, an empty array for none
     * @param keepdims keep reduced axes with length 1
     * @return a bool array
     */
    public static NDArray all(NDArray x, int[] axes, boolean keepdims) {
        return Ufuncs.LOGICAL_AND.reduce(x, axes, keepdims);
    }

    // ------------------------------------------------------------------ into an output

    /**
     * Sums into {@code out}, which must have the reduced shape; the result dtype must convert to
     * {@code out}'s with {@code SAME_KIND} casting.
     *
     * @param x        the array
     * @param axes     the axes; {@code null} for all, an empty array for none
     * @param keepdims keep reduced axes with length 1
     * @param out      the output
     * @return {@code out}
     */
    public static NDArray sum(NDArray x, int[] axes, boolean keepdims, NDArray out) {
        return Ufuncs.ADD.reduce(x, axes, keepdims, out);
    }

    /**
     * Multiplies into {@code out}; see {@link #sum(NDArray, int[], boolean, NDArray)}.
     *
     * @param x        the array
     * @param axes     the axes; {@code null} for all, an empty array for none
     * @param keepdims keep reduced axes with length 1
     * @param out      the output
     * @return {@code out}
     */
    public static NDArray prod(NDArray x, int[] axes, boolean keepdims, NDArray out) {
        return Ufuncs.MULTIPLY.reduce(x, axes, keepdims, out);
    }

    /**
     * The largest elements into {@code out}; see {@link #sum(NDArray, int[], boolean, NDArray)}.
     *
     * @param x        the array
     * @param axes     the axes; {@code null} for all, an empty array for none
     * @param keepdims keep reduced axes with length 1
     * @param out      the output
     * @return {@code out}
     */
    public static NDArray max(NDArray x, int[] axes, boolean keepdims, NDArray out) {
        return Ufuncs.MAXIMUM.reduce(x, axes, keepdims, out);
    }

    /**
     * The smallest elements into {@code out}; see {@link #sum(NDArray, int[], boolean, NDArray)}.
     *
     * @param x        the array
     * @param axes     the axes; {@code null} for all, an empty array for none
     * @param keepdims keep reduced axes with length 1
     * @param out      the output
     * @return {@code out}
     */
    public static NDArray min(NDArray x, int[] axes, boolean keepdims, NDArray out) {
        return Ufuncs.MINIMUM.reduce(x, axes, keepdims, out);
    }

    /**
     * The means into {@code out}; see {@link #sum(NDArray, int[], boolean, NDArray)}.
     *
     * @param x        the array
     * @param axes     the axes; {@code null} for all, an empty array for none
     * @param keepdims keep reduced axes with length 1
     * @param out      the output
     * @return {@code out}
     */
    public static NDArray mean(NDArray x, int[] axes, boolean keepdims, NDArray out) {
        Outputs.accessible(x, out);
        boolean[] reduced = Reducer.reducedAxes(axes, x.ndim());
        DType target = x.dtype().isBool() || x.dtype().isInteger() ? DType.FLOAT64 : x.dtype();
        Outputs.check("mean", out, Reducer.outputShape(x, reduced, keepdims), target);
        if ((x.dtype().isBool() || x.dtype().isInteger())
                && out.dtype() == DType.FLOAT64 && out.isCContiguous() && !out.sharesMemoryWith(x)) {
            IntegerMean.sum(x, reduced, keepdims, Operands.allocator(out, x), out);
            long count = 1;
            for (int d = 0; d < reduced.length; d++) { if (reduced[d]) { count *= x.dim(d); } }
            Ufuncs.DIVIDE.apply(out, Operands.weakLong(DType.INT64, count), out);
            return out;
        }
        if ((x.dtype() == DType.FLOAT32 || x.dtype() == DType.FLOAT64) && out.dtype() == x.dtype()
                && out.isCContiguous() && !out.sharesMemoryWith(x)) {
            Ufuncs.ADD.reduce(x, axes, keepdims, out);
            long count = 1;
            for (int d = 0; d < reduced.length; d++) { if (reduced[d]) { count *= x.dim(d); } }
            final double divisor = count;
            org.numjvm.internal.loops.UnaryLoop divide = (in, pi, si, result, po, so, n) -> {
                for (long i = 0; i < n; i++) {
                    if (x.dtype() == DType.FLOAT32) {
                        float value = in.get(java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED, pi + i * si);
                        result.set(java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED, po + i * so, (float) (value / divisor));
                    } else {
                        double value = in.get(java.lang.foreign.ValueLayout.JAVA_DOUBLE_UNALIGNED, pi + i * si);
                        result.set(java.lang.foreign.ValueLayout.JAVA_DOUBLE_UNALIGNED, po + i * so, value / divisor);
                    }
                }
            };
            org.numjvm.internal.loops.LoopExecutor.unary(divide, out, out);
            return out;
        }
        return Outputs.store("mean", mean(x, axes, keepdims), out);
    }
}
