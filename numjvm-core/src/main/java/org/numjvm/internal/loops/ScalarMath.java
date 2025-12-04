package org.numjvm.internal.loops;

import org.numjvm.exception.ValueException;

/**
 * Scalar operations whose edge cases need more than one Java operator (decision D5). The
 * generated loops call these; everything else is a plain Java expression.
 *
 * <p>The rules, all {@code NUMPY_COMPATIBILITY} unless stated:
 * <ul>
 *   <li><b>Integer overflow wraps</b> to the dtype's width (two's complement), including
 *       {@code abs(MIN)} and {@code MIN // -1}. Java's {@code int} and {@code long}
 *       arithmetic wraps the same way; narrower dtypes are computed in {@code int} and
 *       truncated when written.</li>
 *   <li><b>Integer division and remainder by zero return 0</b>, without an exception or
 *       warning ({@code NUMJVM_SPECIFIC} in part: NumPy also returns 0 but issues a
 *       {@code RuntimeWarning}; Java has no floating-point status flags to mirror).</li>
 *   <li><b>Integer floor division and remainder round toward negative infinity</b>, so
 *       the remainder has the sign of the divisor, as in Python.</li>
 *   <li><b>Floating-point floor division and remainder</b> follow NumPy's
 *       {@code npy_divmod}: the remainder has the sign of the divisor (zero results take
 *       the divisor's sign), the quotient is floored and corrected so that
 *       {@code a == b * q + r} as closely as rounding allows; division by zero gives
 *       {@code a / b} (an infinity or NaN) and the remainder NaN.</li>
 *   <li><b>{@code maximum} and {@code minimum} propagate NaN</b>; {@code fmax} and
 *       {@code fmin} return the other operand when one is NaN. When the operands compare
 *       equal (for example {@code -0.0} and {@code 0.0}), {@code maximum}/{@code minimum}
 *       return the second and {@code fmax}/{@code fmin} the first, as NumPy's SIMD loops do
 *       (the Array API leaves signed zeros unspecified).</li>
 *   <li><b>An integer raised to a negative integer power</b> is a {@link ValueException}
 *       ("Integers to negative integer powers are not allowed.").</li>
 * </ul>
 */
public final class ScalarMath {

    private ScalarMath() {
    }

    // ------------------------------------------------------------------ integer division

    /**
     * Floor division of {@code int} values; 0 when {@code y == 0}.
     *
     * @param x dividend
     * @param y divisor
     * @return {@code floor(x / y)}, wrapped on overflow
     */
    public static int floorDivide(int x, int y) {
        return y == 0 ? 0 : Math.floorDiv(x, y);
    }

    /**
     * Floor division of {@code long} values; 0 when {@code y == 0}.
     *
     * @param x dividend
     * @param y divisor
     * @return {@code floor(x / y)}, wrapped on overflow
     */
    public static long floorDivide(long x, long y) {
        return y == 0 ? 0 : Math.floorDiv(x, y);
    }

    /**
     * Division of uint64 bit patterns; 0 when {@code y == 0}.
     *
     * @param x dividend, unsigned
     * @param y divisor, unsigned
     * @return the unsigned quotient
     */
    public static long floorDivideUnsigned(long x, long y) {
        return y == 0 ? 0 : Long.divideUnsigned(x, y);
    }

    /**
     * Remainder of floor division of {@code int} values (sign of the divisor); 0 when
     * {@code y == 0}.
     *
     * @param x dividend
     * @param y divisor
     * @return {@code x - floor(x / y) * y}
     */
    public static int remainder(int x, int y) {
        return y == 0 ? 0 : Math.floorMod(x, y);
    }

    /**
     * Remainder of floor division of {@code long} values (sign of the divisor); 0 when
     * {@code y == 0}.
     *
     * @param x dividend
     * @param y divisor
     * @return {@code x - floor(x / y) * y}
     */
    public static long remainder(long x, long y) {
        return y == 0 ? 0 : Math.floorMod(x, y);
    }

    /**
     * Remainder of uint64 bit patterns; 0 when {@code y == 0}.
     *
     * @param x dividend, unsigned
     * @param y divisor, unsigned
     * @return the unsigned remainder
     */
    public static long remainderUnsigned(long x, long y) {
        return y == 0 ? 0 : Long.remainderUnsigned(x, y);
    }

    // ------------------------------------------------------------------ floating-point division

    /**
     * NumPy's {@code npy_floor_divide} for {@code double}.
     *
     * @param a dividend
     * @param b divisor
     * @return the floored quotient
     */
    public static double floorDivide(double a, double b) {
        if (b == 0) {
            return a / b;
        }
        double mod = a % b;
        double div = (a - mod) / b;
        if (mod != 0) {
            if ((b < 0) != (mod < 0)) {
                div -= 1.0;
            }
        }
        if (div != 0) {
            double floordiv = Math.floor(div);
            if (div - floordiv > 0.5) {
                floordiv += 1.0;
            }
            return floordiv;
        }
        return Math.copySign(0.0, a / b);
    }

    /**
     * NumPy's {@code npy_remainder} for {@code double}: the result has the sign of
     * {@code b}.
     *
     * @param a dividend
     * @param b divisor
     * @return {@code a - floor(a / b) * b}; NaN when {@code b == 0}
     */
    public static double remainder(double a, double b) {
        if (b == 0) {
            return Double.NaN;
        }
        double mod = a % b;
        if (mod != 0) {
            if ((b < 0) != (mod < 0)) {
                mod += b;
            }
            return mod;
        }
        return Math.copySign(0.0, b);
    }

    /**
     * NumPy's {@code npy_floor_divide} for {@code float}.
     *
     * @param a dividend
     * @param b divisor
     * @return the floored quotient
     */
    public static float floorDivide(float a, float b) {
        if (b == 0) {
            return a / b;
        }
        float mod = a % b;
        float div = (a - mod) / b;
        if (mod != 0) {
            if ((b < 0) != (mod < 0)) {
                div -= 1.0f;
            }
        }
        if (div != 0) {
            float floordiv = (float) Math.floor(div);
            if (div - floordiv > 0.5f) {
                floordiv += 1.0f;
            }
            return floordiv;
        }
        return Math.copySign(0.0f, a / b);
    }

    /**
     * NumPy's {@code npy_remainder} for {@code float}.
     *
     * @param a dividend
     * @param b divisor
     * @return {@code a - floor(a / b) * b}; NaN when {@code b == 0}
     */
    public static float remainder(float a, float b) {
        if (b == 0) {
            return Float.NaN;
        }
        float mod = a % b;
        if (mod != 0) {
            if ((b < 0) != (mod < 0)) {
                mod += b;
            }
            return mod;
        }
        return Math.copySign(0.0f, b);
    }

    // ------------------------------------------------------------------ min / max

    /**
     * NumPy's {@code maximum} for {@code double}: NaN propagates; equal operands give {@code b}.
     *
     * @param a first operand
     * @param b second operand
     * @return the larger operand, or NaN
     */
    public static double maximum(double a, double b) {
        return a != a || a > b ? a : b;
    }

    /**
     * NumPy's {@code minimum} for {@code double}: NaN propagates; equal operands give {@code b}.
     *
     * @param a first operand
     * @param b second operand
     * @return the smaller operand, or NaN
     */
    public static double minimum(double a, double b) {
        return a != a || a < b ? a : b;
    }

    /**
     * NumPy's {@code fmax} for {@code double}: a NaN operand is ignored; equal operands give {@code a}.
     *
     * @param a first operand
     * @param b second operand
     * @return the larger operand, NaN only if both are NaN
     */
    public static double fmax(double a, double b) {
        return b != b || a >= b ? a : b;
    }

    /**
     * NumPy's {@code fmin} for {@code double}: a NaN operand is ignored; equal operands give {@code a}.
     *
     * @param a first operand
     * @param b second operand
     * @return the smaller operand, NaN only if both are NaN
     */
    public static double fmin(double a, double b) {
        return b != b || a <= b ? a : b;
    }

    /**
     * {@link #maximum(double, double)} for {@code float}.
     *
     * @param a first operand
     * @param b second operand
     * @return the larger operand, or NaN
     */
    public static float maximum(float a, float b) {
        return a != a || a > b ? a : b;
    }

    /**
     * {@link #minimum(double, double)} for {@code float}.
     *
     * @param a first operand
     * @param b second operand
     * @return the smaller operand, or NaN
     */
    public static float minimum(float a, float b) {
        return a != a || a < b ? a : b;
    }

    /**
     * {@link #fmax(double, double)} for {@code float}.
     *
     * @param a first operand
     * @param b second operand
     * @return the larger operand, NaN only if both are NaN
     */
    public static float fmax(float a, float b) {
        return b != b || a >= b ? a : b;
    }

    /**
     * {@link #fmin(double, double)} for {@code float}.
     *
     * @param a first operand
     * @param b second operand
     * @return the smaller operand, NaN only if both are NaN
     */
    public static float fmin(float a, float b) {
        return b != b || a <= b ? a : b;
    }

    // ------------------------------------------------------------------ integer power

    /**
     * {@code x} to the power {@code y} for {@code int} values, wrapping on overflow.
     *
     * @param x base
     * @param y exponent, {@code >= 0}
     * @return {@code x^y} modulo 2^32 ({@code 0^0 == 1})
     * @throws ValueException if {@code y < 0}
     */
    public static int power(int x, int y) {
        if (y < 0) {
            throw negativePower();
        }
        int result = 1;
        int base = x;
        for (int e = y; e != 0; e >>>= 1) {
            if ((e & 1) != 0) {
                result *= base;
            }
            base *= base;
        }
        return result;
    }

    /**
     * {@code x} to the power {@code y} for {@code long} values, wrapping on overflow.
     *
     * @param x base
     * @param y exponent, {@code >= 0}
     * @return {@code x^y} modulo 2^64
     * @throws ValueException if {@code y < 0}
     */
    public static long power(long x, long y) {
        if (y < 0) {
            throw negativePower();
        }
        return powerUnsigned(x, y);
    }

    /**
     * {@code x} to the power {@code y} for uint64 bit patterns, wrapping on overflow.
     *
     * @param x base, unsigned
     * @param y exponent, unsigned
     * @return {@code x^y} modulo 2^64
     */
    public static long powerUnsigned(long x, long y) {
        long result = 1;
        long base = x;
        for (long e = y; e != 0; e >>>= 1) {
            if ((e & 1) != 0) {
                result *= base;
            }
            base *= base;
        }
        return result;
    }

    private static ValueException negativePower() {
        return new ValueException("Integers to negative integer powers are not allowed.");
    }
}
