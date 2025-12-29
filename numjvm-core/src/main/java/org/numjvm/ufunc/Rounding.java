package org.numjvm.ufunc;

import java.util.Objects;

import org.numjvm.dtype.DType;
import org.numjvm.exception.DTypeException;
import org.numjvm.internal.memory.MemoryAllocator;
import org.numjvm.ndarray.NDArray;

/**
 * Rounding to a number of decimals ({@code ARRAY_API}: {@code round}; NumPy's
 * {@code np.round}).
 *
 * <p>The algorithm is NumPy's, so results match it including its rounding artefacts:
 * <ul>
 *   <li>integer arrays with {@code decimals >= 0} are returned as a copy;</li>
 *   <li>{@code decimals == 0} is {@link Ufuncs#RINT} (halves to even; bool gives float16);</li>
 *   <li>{@code decimals > 0}: {@code rint(x * 10^d) / 10^d}, each step in the array's dtype,
 *       so float16 rounds at float16 precision;</li>
 *   <li>{@code decimals < 0}: {@code rint(x / 10^-d) * 10^-d}; integer arrays are computed in
 *       float64 and converted back with {@code astype}, so values wrap at the dtype's width
 *       ({@code round(uint8 255, -1) == 4}) and int64 values above 2<sup>53</sup> lose
 *       precision, as in NumPy;</li>
 *   <li>complex arrays round the real and imaginary parts separately, as in NumPy;</li>
 *   <li>bool with {@code decimals != 0} is a {@link DTypeException}, as in NumPy.</li>
 * </ul>
 */
public final class Rounding {

    private static final double[] POWERS = {1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8};

    private Rounding() {
    }

    /**
     * Rounds to the nearest integer, halves to even: {@code round(x, 0)}.
     *
     * @param x the array
     * @return a new array; integer arrays keep their dtype, bool gives float16
     */
    public static NDArray round(NDArray x) {
        return round(x, 0);
    }

    /**
     * Rounds to {@code decimals} decimal places, halves to even.
     *
     * @param x        the array
     * @param decimals number of decimal places; negative values round to tens, hundreds, ...
     * @return a new array of the same dtype (bool with {@code decimals == 0} gives float16)
     * @throws DTypeException for bool with {@code decimals != 0}
     */
    public static NDArray round(NDArray x, int decimals) {
        Objects.requireNonNull(x, "x");
        DType t = x.dtype();
        if (t.isComplex()) {
            // as NumPy's PyArray_Round: the real and imaginary parts are rounded separately
            MemoryAllocator allocator = Operands.allocator(x);
            NDArray r = Operands.allocate(allocator, t, x.shape(), x);
            Outputs.copyInto(round(x.real(), decimals), r.real());
            Outputs.copyInto(round(x.imag(), decimals), r.imag());
            return r;
        }
        if (t.isInteger() && decimals >= 0) {
            return Ufuncs.POSITIVE.apply(x);
        }
        if (decimals == 0) {
            return Ufuncs.RINT.apply(x);
        }
        if (t.isBool()) {
            throw new DTypeException("cannot round bool to " + decimals + " decimals: the result of multiply is "
                    + "float64, which cannot be stored as bool");
        }
        double f = powerOfTen(Math.abs((long) decimals));
        if (decimals > 0) {
            return Ufuncs.DIVIDE.apply(Ufuncs.RINT.apply(Ufuncs.MULTIPLY.apply(x, f)), f);
        }
        NDArray r = Ufuncs.MULTIPLY.apply(Ufuncs.RINT.apply(Ufuncs.DIVIDE.apply(x, f)), f);
        return Operands.cast(r, t, Operands.allocator(x));
    }

    /**
     * NumPy's {@code power_of_ten}: exact up to 10^8, then repeated multiplication.
     */
    static double powerOfTen(long n) {
        if (n < POWERS.length) {
            return POWERS[(int) n];
        }
        double r = POWERS[8];
        for (long i = 8; i < n; i++) {
            r *= 10.0;
        }
        return r;
    }
}
