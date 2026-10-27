package org.numjvm.dtype;

/**
 * Helpers for {@link DType#UINT64}, which Java has no primitive for.
 *
 * <p>{@code getLong} on a uint64 array returns the raw 64 bits, so values from
 * 2<sup>63</sup> upwards appear negative. Use these methods to interpret them.
 */
public final class Unsigned {

    private static final double TWO_POW_63 = 0x1p63;

    private Unsigned() {
    }

    /**
     * Converts raw uint64 bits to the nearest {@code double}.
     *
     * @param bits the 64 bits of an unsigned value
     * @return the value, in {@code [0, 2^64)}
     */
    public static double toDouble(long bits) {
        if (bits >= 0) {
            return bits;
        }
        // Halve with the low bit folded in (round-to-odd), convert, then double: correctly rounded.
        return (double) ((bits >>> 1) | (bits & 1)) * 2.0;
    }

    /**
     * Formats raw uint64 bits as a decimal number.
     *
     * @param bits the 64 bits of an unsigned value
     * @return e.g. {@code "18446744073709551615"} for {@code -1L}
     */
    public static String toString(long bits) {
        return Long.toUnsignedString(bits);
    }

    /**
     * Converts a {@code double} to uint64 bits, truncating toward zero.
     *
     * <p>Values from 2<sup>63</sup> to 2<sup>64</sup> are represented exactly as far as
     * the double allows. Negative values wrap modulo 2<sup>64</sup> (so {@code -1.0}
     * becomes the maximum value), NaN becomes 0, and values of 2<sup>64</sup> or more
     * saturate. NumPy leaves out-of-range conversions undefined.
     *
     * @param value the value
     * @return the raw bits
     */
    public static long fromDouble(double value) {
        if (value >= TWO_POW_63) {
            if (value >= 2 * TWO_POW_63) {
                return -1L;
            }
            return (long) (value - TWO_POW_63) ^ Long.MIN_VALUE;
        }
        return (long) value;
    }
}
