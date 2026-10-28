package org.numjvm.codegen;

import org.numjvm.codegen.DTypeSpec.Family;

/**
 * Conversion expressions between dtype families, for the generated cast loops.
 *
 * <p>The rules are those of {@code NDArray.astype}: integer and bool values keep their
 * low bits (wrapping to the target's width), every conversion to a floating-point type
 * rounds once to nearest, and floating-point values convert to integers by truncation
 * toward zero (out-of-range values follow Java's saturating conversion to {@code long},
 * then wrap; NumPy leaves them undefined).
 */
final class Casts {

    private Casts() {
    }

    /**
     * An expression of the target family's compute type converting {@code x}, a value of
     * {@code from}'s compute type.
     *
     * @param from source dtype
     * @param to   target family
     * @param x    the source value expression
     * @return the conversion expression
     */
    static String convert(DTypeSpec from, Family to, String x) {
        Family f = from.family();
        return switch (to) {
            case BOOL -> switch (f) {
                case BOOL -> x;
                case INT -> x + " != 0";
                case LONG, ULONG -> x + " != 0L";
                case FLOAT -> x + " != 0f";
                case DOUBLE -> x + " != 0.0";
            };
            case INT -> switch (f) {
                case BOOL -> "(" + x + " ? 1 : 0)";
                case INT -> x;
                case LONG, ULONG -> "(int) " + x;
                case FLOAT, DOUBLE -> "(int) (long) " + x;
            };
            case LONG -> switch (f) {
                case BOOL -> "(" + x + " ? 1L : 0L)";
                case INT -> "(long) " + x;
                case LONG, ULONG -> x;
                case FLOAT, DOUBLE -> "(long) " + x;
            };
            case ULONG -> switch (f) {
                case BOOL -> "(" + x + " ? 1L : 0L)";
                case INT -> "(long) " + x;
                case LONG, ULONG -> x;
                case FLOAT, DOUBLE -> "Unsigned.fromDouble(" + x + ")";
            };
            case FLOAT -> switch (f) {
                case BOOL -> "(" + x + " ? 1f : 0f)";
                case INT, LONG -> "(float) " + x;
                case ULONG -> "Unsigned.toFloat(" + x + ")";
                case FLOAT -> x;
                case DOUBLE -> "(float) " + x;
            };
            case DOUBLE -> switch (f) {
                case BOOL -> "(" + x + " ? 1.0 : 0.0)";
                case INT, LONG -> "(double) " + x;
                case ULONG -> "Unsigned.toDouble(" + x + ")";
                case FLOAT -> "(double) " + x;
                case DOUBLE -> x;
            };
        };
    }

    /**
     * The statement writing {@code x} (of {@code from}'s compute type) as dtype {@code to}.
     * Conversions to float16 and bfloat16 from anything wider than {@code float} go through
     * {@code double} with a single rounding.
     */
    static String write(DTypeSpec from, DTypeSpec to, String segment, String offset, String x) {
        boolean half = to.dtype().equals("FLOAT16") || to.dtype().equals("BFLOAT16");
        if (half && from.family() != Family.FLOAT && from.family() != Family.BOOL) {
            String wide = from.family() == Family.DOUBLE ? x : convert(from, Family.DOUBLE, x);
            String fn = to.dtype().equals("FLOAT16") ? "HalfFloat.doubleToFloat16" : "HalfFloat.doubleToBFloat16";
            return segment + ".set(I16, " + offset + ", " + fn + "(" + wide + "))";
        }
        return to.write(segment, offset, convert(from, to.family(), x));
    }

    /**
     * The statements writing {@code x} as a complex number with a zero imaginary part.
     *
     * @param single {@code true} for complex64 (float components), {@code false} for complex128
     */
    static String writeComplex(DTypeSpec from, boolean single, String segment, String offset, String x) {
        if (single) {
            String re = convert(from, Family.FLOAT, x);
            return segment + ".set(F32, " + offset + ", " + re + "); " + segment + ".set(F32, " + offset + " + 4, 0f)";
        }
        String re = convert(from, Family.DOUBLE, x);
        return segment + ".set(F64, " + offset + ", " + re + "); " + segment + ".set(F64, " + offset + " + 8, 0.0)";
    }
}
