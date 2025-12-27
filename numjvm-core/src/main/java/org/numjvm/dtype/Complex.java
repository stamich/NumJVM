package org.numjvm.dtype;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;

/**
 * An immutable complex number with {@code double} real and imaginary parts: the Java value
 * of one element of a {@link DType#COMPLEX64} or {@link DType#COMPLEX128} array.
 *
 * <p>complex64 elements read as a {@code Complex} exactly (each {@code float} part widens
 * to {@code double}); writing a {@code Complex} into a complex64 array rounds each part to
 * {@code float} independently.
 *
 * <p>Equality is the record's: parts are compared with {@link Double#compare}, so a NaN
 * part equals a NaN part and {@code 0.0} differs from {@code -0.0}. That suits use as a
 * map key and in tests; the elementwise {@code equal} ufunc follows IEEE 754 instead.
 *
 * <p>{@link #toString()} prints Python's {@code repr} of a complex number, which NumPy also
 * uses for complex scalars: {@code (1+2j)}, {@code (1.5-0.25j)}, {@code 2j}, {@code (nan+infj)}.
 *
 * @param re real part
 * @param im imaginary part
 */
public record Complex(double re, double im) {

    /** {@code 0 + 0j}. */
    public static final Complex ZERO = new Complex(0, 0);

    /** {@code 1 + 0j}. */
    public static final Complex ONE = new Complex(1, 0);

    /** The imaginary unit {@code 0 + 1j}. */
    public static final Complex I = new Complex(0, 1);

    /**
     * Returns {@code re + im·j}.
     *
     * @param re real part
     * @param im imaginary part
     * @return the complex number
     */
    public static Complex of(double re, double im) {
        return new Complex(re, im);
    }

    /**
     * Returns {@code re + 0j}.
     *
     * @param re real part
     * @return the complex number with a zero imaginary part
     */
    public static Complex ofReal(double re) {
        return new Complex(re, 0);
    }

    /**
     * Returns {@code true} if either part is NaN.
     *
     * @return whether either part is NaN
     */
    public boolean isNaN() {
        return Double.isNaN(re) || Double.isNaN(im);
    }

    /**
     * Returns {@code true} if either part is infinite, as C99 Annex G defines it, even when
     * the other part is NaN.
     *
     * @return whether either part is infinite
     */
    public boolean isInfinite() {
        return Double.isInfinite(re) || Double.isInfinite(im);
    }

    /**
     * Returns {@code true} if both parts are finite.
     *
     * @return whether both parts are finite
     */
    public boolean isFinite() {
        return Double.isFinite(re) && Double.isFinite(im);
    }

    /**
     * Returns Python's {@code repr} of this number: {@code (re±imj)}, or just {@code imj}
     * when the real part is {@code +0.0}. Each part is the shortest decimal that reads back
     * as the same {@code double}, without a trailing {@code .0} and in scientific notation
     * below {@code 1e-4} or from {@code 1e16} on.
     *
     * @return the string form
     */
    @Override
    public String toString() {
        String imag = repr(im);
        if (re == 0 && Double.doubleToRawLongBits(re) == 0) {
            return imag + "j";
        }
        String sign = imag.startsWith("-") ? "" : "+";
        return "(" + repr(re) + sign + imag + "j)";
    }

    /**
     * Python's {@code float.__repr__} without the trailing {@code .0}, as used inside
     * {@code complex.__repr__}.
     */
    static String repr(double v) {
        if (Double.isNaN(v)) {
            return "nan";
        }
        if (Double.isInfinite(v)) {
            return v > 0 ? "inf" : "-inf";
        }
        String sign = v < 0 || v == 0 && Double.doubleToRawLongBits(v) < 0 ? "-" : "";
        if (v == 0) {
            return sign + "0";
        }
        double a = Math.abs(v);
        BigDecimal d = new BigDecimal(Double.toString(a)).stripTrailingZeros();
        if (d.precision() == 2) {
            // Double.toString never prints fewer than two digits; Python prints one when one
            // digit reads back (4.9e-324 is 5e-324 in Python)
            BigDecimal down = d.round(new MathContext(1, RoundingMode.DOWN));
            BigDecimal up = d.round(new MathContext(1, RoundingMode.UP));
            boolean downOk = Double.parseDouble(down.toString()) == a;
            boolean upOk = Double.parseDouble(up.toString()) == a;
            if (downOk && upOk) {
                d = d.round(new MathContext(1, RoundingMode.HALF_EVEN));
            } else if (downOk) {
                d = down;
            } else if (upOk) {
                d = up;
            }
        }
        int exponent = d.precision() - d.scale() - 1;
        if (exponent < -4 || exponent >= 16) {
            String digits = d.unscaledValue().toString();
            String mantissa = digits.length() == 1 ? digits : digits.charAt(0) + "." + digits.substring(1);
            String e = Integer.toString(Math.abs(exponent));
            return sign + mantissa + "e" + (exponent < 0 ? "-" : "+") + (e.length() < 2 ? "0" + e : e);
        }
        return sign + d.toPlainString();
    }
}
