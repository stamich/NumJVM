package org.numjvm.internal;

/**
 * bfloat16 conversions, and correctly rounded conversions from {@code double} to both
 * 16-bit float formats. (Between IEEE binary16 and {@code float}, use
 * {@link Float#float16ToFloat(short)} and {@link Float#floatToFloat16(float)}.)
 */
public final class HalfFloat {

    private HalfFloat() {
    }

    /**
     * Widens bfloat16 bits to a float: bfloat16 is the upper half of a binary32.
     *
     * @param bits the 16 bits
     * @return the exact value
     */
    public static float bfloat16ToFloat(short bits) {
        return Float.intBitsToFloat((bits & 0xFFFF) << 16);
    }

    /**
     * Narrows a float to bfloat16 with round-to-nearest-even; NaN stays a (quiet) NaN.
     *
     * @param value the value
     * @return the 16 bits
     */
    public static short floatToBFloat16(float value) {
        int bits = Float.floatToRawIntBits(value);
        if (Float.isNaN(value)) {
            return (short) ((bits >>> 16) | 0x0040);
        }
        int roundingBias = 0x7FFF + ((bits >>> 16) & 1);
        return (short) ((bits + roundingBias) >>> 16);
    }

    /**
     * Narrows a {@code double} to IEEE binary16 with a single round-to-nearest-even, as
     * NumPy's {@code npy_double_to_half} does. Going through {@code (float) value} would
     * round twice and differ for values just beyond a halfway point.
     *
     * @param value the value
     * @return the 16 bits
     */
    public static short doubleToFloat16(double value) {
        return Float.floatToFloat16(toFloatRoundToOdd(value));
    }

    /**
     * Narrows a {@code double} to bfloat16 with a single round-to-nearest-even.
     *
     * @param value the value
     * @return the 16 bits
     */
    public static short doubleToBFloat16(double value) {
        return floatToBFloat16(toFloatRoundToOdd(value));
    }

    /**
     * Rounds to {@code float} toward zero and sets the lowest bit when the result is
     * inexact ("round to odd"). Rounding that result to a format with at least two bits
     * less precision gives the same result as rounding {@code value} directly.
     */
    static float toFloatRoundToOdd(double value) {
        float f = (float) value;
        if (Double.isNaN(value) || (double) f == value) {
            return f;
        }
        if (Math.abs((double) f) > Math.abs(value)) {
            f = Math.nextAfter(f, 0.0);
        }
        return Float.intBitsToFloat(Float.floatToRawIntBits(f) | 1);
    }
}
