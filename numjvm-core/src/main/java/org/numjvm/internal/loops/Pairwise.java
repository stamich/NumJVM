package org.numjvm.internal.loops;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.numjvm.internal.HalfFloat;

/**
 * NumPy's pairwise summation ({@code @TYPE@_pairwise_sum} in {@code loops_utils.h.src}),
 * used by {@code add} reductions over floating-point runs: blocks of up to 128 elements are
 * summed with eight accumulators and blocks are combined recursively, which bounds the
 * rounding error by O(log n) instead of O(n). float16, bfloat16 and float32 accumulate in
 * {@code float}, float64 in {@code double}, as in NumPy.
 */
public final class Pairwise {

    private static final ValueLayout.OfShort I16 = ValueLayout.JAVA_SHORT_UNALIGNED;
    private static final ValueLayout.OfFloat F32 = ValueLayout.JAVA_FLOAT_UNALIGNED;
    private static final ValueLayout.OfDouble F64 = ValueLayout.JAVA_DOUBLE_UNALIGNED;

    /**
     * Block size below which elements are summed with eight accumulators, as in NumPy.
     */
    static final int BLOCK = 128;

    private Pairwise() {
    }

    /**
     * Sums {@code n} float64 elements at byte stride {@code stride}.
     *
     * @param s      memory
     * @param off    byte offset of the first element
     * @param stride byte stride
     * @param n      number of elements
     * @return the pairwise sum
     */
    public static double float64(MemorySegment s, long off, long stride, long n) {
        if (n < 8) {
            double res = 0.0;
            for (long i = 0; i < n; i++) {
                res += s.get(F64, off + i * stride);
            }
            return res;
        } else if (n <= BLOCK) {
            double r0 = s.get(F64, off);
            double r1 = s.get(F64, off + stride);
            double r2 = s.get(F64, off + 2 * stride);
            double r3 = s.get(F64, off + 3 * stride);
            double r4 = s.get(F64, off + 4 * stride);
            double r5 = s.get(F64, off + 5 * stride);
            double r6 = s.get(F64, off + 6 * stride);
            double r7 = s.get(F64, off + 7 * stride);
            long limit = n - n % 8;
            long i;
            for (i = 8; i < limit; i += 8) {
                long p = off + i * stride;
                r0 += s.get(F64, p);
                r1 += s.get(F64, p + stride);
                r2 += s.get(F64, p + 2 * stride);
                r3 += s.get(F64, p + 3 * stride);
                r4 += s.get(F64, p + 4 * stride);
                r5 += s.get(F64, p + 5 * stride);
                r6 += s.get(F64, p + 6 * stride);
                r7 += s.get(F64, p + 7 * stride);
            }
            double res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
            for (; i < n; i++) {
                res += s.get(F64, off + i * stride);
            }
            return res;
        }
        long n2 = n / 2;
        n2 -= n2 % 8;
        return float64(s, off, stride, n2) + float64(s, off + n2 * stride, stride, n - n2);
    }

    /**
     * Sums {@code n} float32 elements in {@code float}.
     *
     * @param s      memory
     * @param off    byte offset of the first element
     * @param stride byte stride
     * @param n      number of elements
     * @return the pairwise sum
     */
    public static float float32(MemorySegment s, long off, long stride, long n) {
        return sum(s, off, stride, n, 0);
    }

    /**
     * Sums {@code n} float16 elements in {@code float}.
     *
     * @param s      memory
     * @param off    byte offset of the first element
     * @param stride byte stride
     * @param n      number of elements
     * @return the pairwise sum, not yet rounded to float16
     */
    public static float float16(MemorySegment s, long off, long stride, long n) {
        return sum(s, off, stride, n, 1);
    }

    /**
     * Sums {@code n} bfloat16 elements in {@code float}.
     *
     * @param s      memory
     * @param off    byte offset of the first element
     * @param stride byte stride
     * @param n      number of elements
     * @return the pairwise sum, not yet rounded to bfloat16
     */
    public static float bfloat16(MemorySegment s, long off, long stride, long n) {
        return sum(s, off, stride, n, 2);
    }

    private static float get(MemorySegment s, long off, int type) {
        return switch (type) {
            case 0 -> s.get(F32, off);
            case 1 -> Float.float16ToFloat(s.get(I16, off));
            default -> HalfFloat.bfloat16ToFloat(s.get(I16, off));
        };
    }

    /**
     * The pairwise sum in {@code float}; {@code type} 0 float32, 1 float16, 2 bfloat16.
     */
    private static float sum(MemorySegment s, long off, long stride, long n, int type) {
        if (n < 8) {
            float res = 0.0f;
            for (long i = 0; i < n; i++) {
                res += get(s, off + i * stride, type);
            }
            return res;
        } else if (n <= BLOCK) {
            float r0 = get(s, off, type);
            float r1 = get(s, off + stride, type);
            float r2 = get(s, off + 2 * stride, type);
            float r3 = get(s, off + 3 * stride, type);
            float r4 = get(s, off + 4 * stride, type);
            float r5 = get(s, off + 5 * stride, type);
            float r6 = get(s, off + 6 * stride, type);
            float r7 = get(s, off + 7 * stride, type);
            long limit = n - n % 8;
            long i;
            for (i = 8; i < limit; i += 8) {
                long p = off + i * stride;
                r0 += get(s, p, type);
                r1 += get(s, p + stride, type);
                r2 += get(s, p + 2 * stride, type);
                r3 += get(s, p + 3 * stride, type);
                r4 += get(s, p + 4 * stride, type);
                r5 += get(s, p + 5 * stride, type);
                r6 += get(s, p + 6 * stride, type);
                r7 += get(s, p + 7 * stride, type);
            }
            float res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
            for (; i < n; i++) {
                res += get(s, off + i * stride, type);
            }
            return res;
        }
        long n2 = n / 2;
        n2 -= n2 % 8;
        return sum(s, off, stride, n2, type) + sum(s, off + n2 * stride, stride, n - n2, type);
    }

    /**
     * One part (real or imaginary) of NumPy's pairwise sum of {@code n} complex128 elements:
     * {@code CDOUBLE_pairwise_sum} sums both parts with independent accumulators, so each part
     * is summed on its own here with the same blocking (four accumulators, blocks of 64
     * elements). {@code off} addresses the part of the first element.
     *
     * @param s      memory
     * @param off    byte offset of the part of the first element
     * @param stride byte stride between elements
     * @param n      number of elements
     * @return the sum of that part
     */
    public static double complex128Part(MemorySegment s, long off, long stride, long n) {
        if (n < 4) {
            double res = 0.0;
            for (long i = 0; i < n; i++) {
                res += s.get(F64, off + i * stride);
            }
            return res;
        } else if (n <= BLOCK / 2) {
            double r0 = s.get(F64, off);
            double r1 = s.get(F64, off + stride);
            double r2 = s.get(F64, off + 2 * stride);
            double r3 = s.get(F64, off + 3 * stride);
            long limit = n - n % 4;
            long i;
            for (i = 4; i < limit; i += 4) {
                long p = off + i * stride;
                r0 += s.get(F64, p);
                r1 += s.get(F64, p + stride);
                r2 += s.get(F64, p + 2 * stride);
                r3 += s.get(F64, p + 3 * stride);
            }
            double res = (r0 + r1) + (r2 + r3);
            for (; i < n; i++) {
                res += s.get(F64, off + i * stride);
            }
            return res;
        }
        long n2 = (n - n % 8) / 2;
        return complex128Part(s, off, stride, n2) + complex128Part(s, off + n2 * stride, stride, n - n2);
    }

    /**
     * One part of NumPy's pairwise sum of {@code n} complex64 elements, in {@code float}; see
     * {@link #complex128Part}.
     *
     * @param s      memory
     * @param off    byte offset of the part of the first element
     * @param stride byte stride between elements
     * @param n      number of elements
     * @return the sum of that part
     */
    public static float complex64Part(MemorySegment s, long off, long stride, long n) {
        if (n < 4) {
            float res = 0.0f;
            for (long i = 0; i < n; i++) {
                res += s.get(F32, off + i * stride);
            }
            return res;
        } else if (n <= BLOCK / 2) {
            float r0 = s.get(F32, off);
            float r1 = s.get(F32, off + stride);
            float r2 = s.get(F32, off + 2 * stride);
            float r3 = s.get(F32, off + 3 * stride);
            long limit = n - n % 4;
            long i;
            for (i = 4; i < limit; i += 4) {
                long p = off + i * stride;
                r0 += s.get(F32, p);
                r1 += s.get(F32, p + stride);
                r2 += s.get(F32, p + 2 * stride);
                r3 += s.get(F32, p + 3 * stride);
            }
            float res = (r0 + r1) + (r2 + r3);
            for (; i < n; i++) {
                res += s.get(F32, off + i * stride);
            }
            return res;
        }
        long n2 = (n - n % 8) / 2;
        return complex64Part(s, off, stride, n2) + complex64Part(s, off + n2 * stride, stride, n - n2);
    }
}
