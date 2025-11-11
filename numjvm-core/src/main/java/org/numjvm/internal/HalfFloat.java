package org.numjvm.internal;

/**
 * bfloat16 conversions. (IEEE binary16 uses {@link Float#float16ToFloat(short)} and
 * {@link Float#floatToFloat16(float)}.)
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
}
